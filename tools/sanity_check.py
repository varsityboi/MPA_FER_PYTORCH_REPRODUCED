"""
Sanity check for the model + losses, using FAKE images (no dataset needed).

Run in Colab (GPU runtime), from the repo folder:
    python tools/sanity_check.py
    python tools/sanity_check.py --set train.batch_size=16      # any config override

What it checks:
     1. config loads, --set overrides work, typos raise errors
     2. hard prompts fit in CLIP's 77-token limit
     3. model builds, CLIP frozen, only the prompt tensors train
     4. our encoders give the same features as original CLIP (we didn't break CLIP)
     5. forward pass shapes
     6. every enabled loss is present and finite
     7. gradients reach the prompts (text + first/last visual layer) and nothing else
     8. training steps with fp16: no NaN, CLIP unchanged, prompts change, loss goes down
     9. checkpoint save -> load gives back identical prompts
    10. GPU memory + speed at the real batch size (+ epoch time estimate)
    11. ablation configs (CoOp baseline, hard prompt type 1) still run
"""
import argparse
import copy
import math
import os
import sys
import tempfile
import time
import traceback

# Make "from models ..." work when running "python tools/sanity_check.py"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # relative paths in the config (prompts/...) need the repo root

import clip
import torch
import torch.nn.functional as F

from losses import build_criterion, prototype_loss, soft_hard_contrastive
from models.mpa_fer import build_model, get_hard_template
from models.text_prompts import CONTEXT_LENGTH, _tokenizer, build_hard_prompts, load_descriptions
from utils.misc import (apply_overrides, get_device, load_checkpoint, load_config,
                        param_report, save_checkpoint, set_seed)


# ------------------------------------------------------------------
# Pass / fail printing
# ------------------------------------------------------------------
class Checker:
    def __init__(self):
        self.n_pass = self.n_fail = self.n_warn = 0
        self.current = "startup"

    def section(self, title):
        self.current = title
        print(f"\n=== {title} ===", flush=True)

    def ok(self, name, condition, detail=""):
        condition = bool(condition)
        if condition:
            self.n_pass += 1
        else:
            self.n_fail += 1
        tag = "PASS" if condition else "FAIL"
        print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
        return condition

    def warn(self, name, detail=""):
        self.n_warn += 1
        print(f"  [WARN] {name}" + (f"  ({detail})" if detail else ""), flush=True)

    def info(self, msg):
        print(f"  [INFO] {msg}", flush=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def make_fake_batch(batch_size, n_cls, device, seed=0):
    """Easy fake data: each class has its own fixed random picture + a little noise."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n_cls, 3, 224, 224, generator=g)
    labels = torch.arange(batch_size) % n_cls
    images = base[labels] + 0.1 * torch.randn(batch_size, 3, 224, 224, generator=g)
    return images.to(device), labels.to(device)


def expected_trainable(cfg, model):
    """What the trainable tensors SHOULD be, computed from the config."""
    n_cls = len(cfg["data"]["class_names"])
    tp = cfg["model"]["text_prompts"]
    text_width = model.text_prompts.token_prefix.shape[-1]
    expected = {
        "text_prompts.ctx": (n_cls if tp["class_specific_ctx"] else 1) * tp["n_ctx"] * text_width
    }
    if cfg["model"]["visual_prompts"]["enabled"]:
        ve = model.visual_encoder
        expected["visual_encoder.prompts"] = (
            ve.n_layers * cfg["model"]["visual_prompts"]["n_prompts"] * ve.conv1.out_channels
        )
    return expected


@torch.no_grad()
def frozen_checksums(model):
    return [p.double().sum().item() for p in model.parameters() if not p.requires_grad]


def train_step(model, criterion, optimizer, scaler, images, labels, device, use_amp):
    """Same pattern train.py will use: fp16 forward, float32 losses, scaled backward."""
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        out = model(images)
    loss, logs = criterion(out, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return logs


def fmt(logs):
    return "  ".join(f"{k} {v:.4f}" for k, v in logs.items())


def parse_args():
    p = argparse.ArgumentParser(description="MPA-FER sanity check with fake data")
    p.add_argument("--config", default="configs/rafdb.yaml")
    p.add_argument("--set", dest="overrides", nargs="*", default=[],
                   help="config overrides, e.g. --set train.batch_size=16")
    p.add_argument("--overfit_steps", type=int, default=30)
    p.add_argument("--train_size", type=int, default=12271,
                   help="number of training images, only for the epoch time estimate")
    p.add_argument("--skip_memory", action="store_true")
    p.add_argument("--skip_ablation", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(ck, args):
    device = get_device()
    use_amp = device == "cuda"
    print(f"torch {torch.__version__} | device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if use_amp else ""))
    if not use_amp:
        print("WARNING: no GPU. This will be very slow. In Colab: Runtime > Change runtime type > T4 GPU")

    # ---------------- 1. config ----------------
    ck.section("1. Config")
    cfg = load_config(args.config)
    class_names = cfg["data"]["class_names"]
    n_cls = len(class_names)
    ck.ok("config loads", True, args.config)
    ck.ok("7 class names", n_cls == 7, str(class_names))

    test = apply_overrides(copy.deepcopy(cfg), [
        "train.epochs=3", "model.visual_prompts.enabled=false", "hard_prompts.templates.1=hello"])
    ck.ok("--set overrides work (int, bool, number keys)",
          test["train"]["epochs"] == 3
          and test["model"]["visual_prompts"]["enabled"] is False
          and test["hard_prompts"]["templates"][1] == "hello")
    try:
        apply_overrides(copy.deepcopy(cfg), ["train.epoch=3"])
        ck.ok("typo in --set key raises an error", False)
    except KeyError:
        ck.ok("typo in --set key raises an error", True)

    if args.overrides:
        apply_overrides(cfg, args.overrides)
        ck.info(f"applied your overrides: {args.overrides}")

    # ---------------- 2. hard prompt length ----------------
    ck.section("2. Hard prompts fit in CLIP's 77 tokens")
    template = get_hard_template(cfg)
    ck.info(f"template (type {cfg['hard_prompts']['type']}): {template}")
    descriptions = None
    if "{desc}" in template:
        descriptions = load_descriptions(cfg["hard_prompts"]["descriptions_file"])
        missing = [c for c in class_names if c not in descriptions]
        ck.ok("every class has a description", not missing, f"missing: {missing}" if missing else "")
    hard_texts = build_hard_prompts(class_names, template, descriptions)
    lengths = {c: len(_tokenizer.encode(t)) + 2 for c, t in zip(class_names, hard_texts)}
    for c, n in lengths.items():
        ck.info(f"{c:<10} {n} tokens")
    longest = max(lengths.values())
    ck.ok(f"all hard prompts <= {CONTEXT_LENGTH} tokens", longest <= CONTEXT_LENGTH, f"longest {longest}")

    # ---------------- 3. build + frozen ----------------
    ck.section("3. Build model, CLIP frozen")
    set_seed(cfg["seed"])
    t0 = time.time()
    model = build_model(cfg, device)
    ck.info(f"built in {time.time() - t0:.1f}s (first run downloads CLIP, about 335 MB)")
    for line in param_report(model).splitlines():
        ck.info(line)

    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    expected = expected_trainable(cfg, model)
    ck.ok("only the prompt tensors are trainable", set(trainable) == set(expected),
          f"got {sorted(trainable)}")
    n_train = sum(p.numel() for p in trainable.values())
    ck.ok("trainable count matches config", n_train == sum(expected.values()),
          f"{n_train:,} vs expected {sum(expected.values()):,}")
    ck.ok("all weights are float32", all(p.dtype == torch.float32 for p in model.parameters()))
    #ck.info(f"paper reports 0.218M trainable for ViT-B/16, ours is {n_train / 1e6:.3f}M (known gap, goes in README)")
    ck.info(f"{n_train:,} params = {n_train * 4 / 1024**2:.3f} MB in fp32 "
            f"(paper quotes 0.218 MB for ViT-B/16; known gap, goes in README)")
    # ---------------- 4. match original CLIP ----------------
    ck.section("4. Our encoders match original CLIP")
    images, labels = make_fake_batch(4, n_cls, device)
    ref, _ = clip.load(cfg["model"]["clip_backbone"], device="cpu", jit=False)
    ref = ref.float().to(device).eval()
    model.eval()
    with torch.no_grad():
        ours = model.encode_frozen_image(images)
        theirs = ref.encode_image(images)
        d = (ours - theirs).abs().max().item()
        ck.ok("image features (prompts off) == CLIP encode_image", d < 1e-3, f"max diff {d:.1e}")

        hard_tokens = clip.tokenize(model.text_prompts.hard_texts, truncate=True).to(device)
        d = (model.text_prompts.hard_text_features - ref.encode_text(hard_tokens)).abs().max().item()
        ck.ok("hard prompt text features == CLIP encode_text", d < 1e-3, f"max diff {d:.1e}")

        if model.visual_encoder.prompts is not None:
            prompted = model.visual_encoder(images, use_prompts=True)["global"]
            d = (prompted - ours).abs().max().item()
            ck.ok("visual prompts change the features (really inserted)", d > 1e-6, f"max diff {d:.1e}")
    del ref
    if use_amp:
        torch.cuda.empty_cache()
    model.train()

    # ---------------- 5. shapes ----------------
    ck.section("5. Forward pass shapes")
    E = model.text_prompts.hard_text_features.shape[1]
    W = model.text_prompts.token_prefix.shape[-1]
    B = images.shape[0]
    with torch.no_grad():
        out = model(images)
    expected_shapes = {
        "logits": (B, n_cls),
        "image_global": (B, E),
        "text_features": (n_cls, E),
        "hard_text_features": (n_cls, E),
        "soft_token_emb": (n_cls, W),
        "hard_token_emb": (n_cls, W),
    }
    for key, shape in expected_shapes.items():
        got = tuple(out[key].shape)
        ck.ok(f"{key} shape {shape}", got == shape, "" if got == shape else f"got {got}")
    ck.ok("logits are finite", torch.isfinite(out["logits"]).all().item())

    # ---------------- 6. losses ----------------
    ck.section("6. Losses")
    fake_protos = F.normalize(torch.randn(n_cls, E), dim=-1)
    sh_on = cfg["losses"]["soft_hard_align"]["enabled"]
    proto_on = cfg["losses"]["proto_align"]["enabled"]
    criterion = build_criterion(cfg, fake_protos if proto_on else None, device)
    loss, logs = criterion(out, labels)
    ck.info(fmt(logs))
    expected_keys = {"loss_vt", "loss_total"}
    if sh_on:
        expected_keys |= {"loss_ta", "loss_pa"}
    if proto_on:
        expected_keys |= {"loss_v"}
    ck.ok("all enabled loss parts present", set(logs) == expected_keys, str(sorted(logs)))
    ck.ok("all losses finite", all(math.isfinite(v) for v in logs.values()))
    if proto_on:
        try:
            build_criterion(cfg, None, device)
            ck.ok("missing prototypes gives a clear error", False)
        except ValueError:
            ck.ok("missing prototypes gives a clear error", True)

    # ---------------- 7. gradients ----------------
    ck.section("7. Gradients reach the prompts, and only the prompts")
    model.zero_grad(set_to_none=True)
    out = model(images)
    loss, _ = criterion(out, labels)
    loss.backward()

    ctx = model.text_prompts.ctx
    ck.ok("text ctx gets gradient", ctx.grad is not None and ctx.grad.abs().sum().item() > 0)
    vp = model.visual_encoder.prompts
    if vp is not None:
        has = vp.grad is not None
        ck.ok("visual prompts get gradient (layer 1)", has and vp.grad[0].abs().sum().item() > 0)
        ck.ok(f"visual prompts get gradient (layer {vp.shape[0]})", has and vp.grad[-1].abs().sum().item() > 0)
    leaked = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    ck.ok("no gradients on frozen CLIP weights", not leaked, f"{leaked[:3]}" if leaked else "")

    # each extra loss is connected to the right prompts
    model.zero_grad(set_to_none=True)
    out = model(images)
    tau = cfg["losses"]["soft_hard_align"]["tau"]
    l_t = (soft_hard_contrastive(out["soft_token_emb"], out["hard_token_emb"], tau)
           + soft_hard_contrastive(out["text_features"], out["hard_text_features"], tau))
    g = torch.autograd.grad(l_t, ctx, retain_graph=True)[0]
    ck.ok("L_t (soft-hard alignment) reaches text ctx", g.abs().sum().item() > 0)
    if vp is not None:
        l_v = prototype_loss(out["image_global"], labels, fake_protos.to(device),
                             cfg["losses"]["proto_align"]["match"])
        g = torch.autograd.grad(l_v, vp)[0]
        ck.ok("L_v (prototype) reaches visual prompts", g.abs().sum().item() > 0)
    model.zero_grad(set_to_none=True)

    # ---------------- 8. training steps ----------------
    ck.section(f"8. {args.overfit_steps} training steps on easy fake data (fp16 AMP: {use_amp})")
    tcfg = cfg["train"]
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=tcfg["lr"], momentum=tcfg["momentum"],
                                weight_decay=tcfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    frozen_before = frozen_checksums(model)
    prompts_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    small_images, small_labels = make_fake_batch(2 * n_cls, n_cls, device, seed=1)
    history = []
    for step in range(args.overfit_steps):
        logs = train_step(model, criterion, optimizer, scaler, small_images, small_labels, device, use_amp)
        history.append(logs)
        if step % 10 == 0 or step == args.overfit_steps - 1:
            print(f"    step {step:3d}  {fmt(logs)}", flush=True)

    ck.ok("all losses stayed finite (no NaN)",
          all(math.isfinite(v) for h in history for v in h.values()))
    ck.ok("frozen CLIP weights unchanged", frozen_checksums(model) == frozen_before)
    changed = all(not torch.equal(p, prompts_before[n])
                  for n, p in model.named_parameters() if p.requires_grad)
    ck.ok("prompts were updated", changed)
    n = max(1, min(3, len(history) // 2))
    first = sum(h["loss_vt"] for h in history[:n]) / n
    last = sum(h["loss_vt"] for h in history[-n:]) / n
    if last < first:
        ck.ok("loss_vt went down", True, f"{first:.3f} -> {last:.3f}")
    else:
        ck.warn("loss_vt did not go down on fake data", f"{first:.3f} -> {last:.3f}, tell Claude")
    if use_amp:
        ck.info(f"AMP grad scale now {scaler.get_scale():.0f}")

    # ---------------- 9. checkpoint ----------------
    ck.section("9. Checkpoint save -> load")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "last.pt")
        save_checkpoint(path, model, optimizer, scaler=scaler, epoch=3, best_acc=55.5, cfg=cfg)
        size_kb = os.path.getsize(path) / 1024
        saved = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(1.0)  # mess up the prompts on purpose
        state = load_checkpoint(path, model, optimizer, scaler=scaler)
        same = all(torch.equal(p, saved[n]) for n, p in model.named_parameters() if p.requires_grad)
        ck.ok("prompts restored exactly", same)
        ck.ok("epoch and best_acc restored", state["epoch"] == 3 and state["best_acc"] == 55.5)
        ck.info(f"checkpoint size {size_kb:.0f} KB (prompts + optimizer only, not CLIP)")

    # ---------------- 10. memory + speed ----------------
    if use_amp and not args.skip_memory:
        bs = tcfg["batch_size"]
        ck.section(f"10. GPU memory + speed at batch size {bs}")
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        big_images, big_labels = make_fake_batch(bs, n_cls, device, seed=2)
        try:
            times = []
            for _ in range(5):
                torch.cuda.synchronize()
                t = time.time()
                train_step(model, criterion, optimizer, scaler, big_images, big_labels, device, use_amp)
                torch.cuda.synchronize()
                times.append(time.time() - t)
            peak = torch.cuda.max_memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if peak < 0.9 * total:
                ck.ok(f"batch {bs} fits in GPU memory", True, f"peak {peak:.1f} GB of {total:.1f} GB")
            else:
                ck.warn(f"batch {bs} is very close to the memory limit", f"peak {peak:.1f} GB of {total:.1f} GB")
            per_iter = sum(times[2:]) / len(times[2:])  # skip 2 warm-up steps
            epoch_min = per_iter * math.ceil(args.train_size / bs) / 60
            ck.info(f"{per_iter:.2f} s/iter -> about {epoch_min:.1f} min per epoch "
                    f"on {args.train_size} images (model only, data loading adds more)")
            ck.info(f"{tcfg['epochs']} epochs would take about {epoch_min * tcfg['epochs'] / 60:.1f} h")
        except torch.cuda.OutOfMemoryError:
            ck.ok(f"batch {bs} fits in GPU memory", False, "out of memory, try --set train.batch_size=16")
            torch.cuda.empty_cache()

    # ---------------- 11. ablations ----------------
    if not args.skip_ablation:
        ck.section("11. Ablation configs still run")
        del model, criterion, optimizer, scaler, out, loss
        if use_amp:
            torch.cuda.empty_cache()
        variants = {
            "CoOp baseline (all 4 switches off)": [
                "model.visual_prompts.enabled=false",
                "model.local_align.enabled=false",
                "losses.soft_hard_align.enabled=false",
                "losses.proto_align.enabled=false",
            ],
            "hard prompt type 1 (no descriptions)": ["hard_prompts.type=1"],
        }
        for name, overrides in variants.items():
            vcfg = apply_overrides(copy.deepcopy(cfg), overrides)
            m = build_model(vcfg, device)
            protos = fake_protos if vcfg["losses"]["proto_align"]["enabled"] else None
            crit = build_criterion(vcfg, protos, device)
            opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad],
                                  lr=vcfg["train"]["lr"], momentum=0.9)
            sc = torch.amp.GradScaler("cuda", enabled=use_amp)
            logs = train_step(m, crit, opt, sc, images, labels, device, use_amp)
            ck.ok(f"{name}: one step, losses finite",
                  all(math.isfinite(v) for v in logs.values()), fmt(logs))
            n_m = m.count_trainable_params()
            exp = sum(expected_trainable(vcfg, m).values())
            ck.ok(f"{name}: trainable params", n_m == exp, f"{n_m:,}")
            del m, crit, opt, sc
            if use_amp:
                torch.cuda.empty_cache()

    # ---------------- summary ----------------
    print(f"\n=== SUMMARY: {ck.n_pass} passed, {ck.n_fail} failed, {ck.n_warn} warnings ===")
    if ck.n_fail:
        print("Something is wrong. Paste the FAIL lines (or the whole output) to Claude.")
        sys.exit(1)
    print("All good. Model + losses are ready for real data.")


if __name__ == "__main__":
    args = parse_args()
    checker = Checker()
    try:
        main(checker, args)
    except Exception:
        traceback.print_exc()
        print(f"\nCRASHED during: {checker.current}. Paste the error above to Claude.")
        sys.exit(1)