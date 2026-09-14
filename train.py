"""
Train MPA-FER.

    python train.py                                  # full run from configs/rafdb.yaml
    python train.py --set train.epochs=3             # quick 3-epoch check
    python train.py --set train.epochs=1 train.max_iters_per_epoch=20   # 2-minute smoke run

Only the text context vectors and the visual prompt tokens are optimised;
CLIP itself never moves. Checkpoints therefore hold a few hundred KB, and
train.py resumes from outputs/<name>/last.pt automatically after a Colab
disconnect (set output.resume=false to start clean).
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import build_loaders
from losses import build_criterion
from models.mpa_fer import build_model
from tools.build_prototypes import load_or_build
from utils.misc import (Logger, MeterDict, find_resume_checkpoint, format_time,
                        get_device, load_checkpoint, load_config, param_report,
                        save_checkpoint, save_config, set_seed)


# ------------------------------------------------------------------
# LR schedule: linear warmup, then cosine decay, stepped every iteration
# ------------------------------------------------------------------
def build_scheduler(optimizer, cfg, iters_per_epoch):
    total = max(1, cfg["train"]["epochs"] * iters_per_epoch)
    warmup = cfg["train"]["warmup_epochs"] * iters_per_epoch

    def lr_lambda(step):
        if warmup > 0 and step < warmup:
            return 0.01 + 0.99 * step / warmup  # ramp up from 1% of base lr
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    if cfg["train"]["scheduler"] != "cosine":
        raise ValueError("Only scheduler='cosine' is implemented")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, n_classes, use_amp=True, desc="eval"):
    """Overall accuracy and mean per-class accuracy, both in percent."""
    model.eval()
    correct = torch.zeros(n_classes, device=device)
    total = torch.zeros(n_classes, device=device)
    loss_sum, n_seen = 0.0, 0

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            logits = model(images)["logits"]
        logits = logits.float()
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        n_seen += labels.numel()

        preds = logits.argmax(dim=1)
        ones = torch.ones_like(labels, dtype=torch.float)
        total.index_add_(0, labels, ones)
        correct.index_add_(0, labels, (preds == labels).float())

    model.train()
    overall = 100.0 * correct.sum().item() / max(n_seen, 1)
    seen = total > 0
    mean_class = 100.0 * (correct[seen] / total[seen]).mean().item()
    per_class = (100.0 * correct / total.clamp(min=1)).tolist()
    return {
        "acc": overall,
        "mean_class_acc": mean_class,
        "test_loss": loss_sum / max(n_seen, 1),
        "per_class_acc": [round(v, 2) for v in per_class],
    }


# ------------------------------------------------------------------
# One epoch
# ------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, cfg, epoch, logger):
    model.train()
    meters = MeterDict()
    max_iters = cfg["train"]["max_iters_per_epoch"]
    n_iters = len(loader) if max_iters is None else min(max_iters, len(loader))
    use_amp = scaler.is_enabled()
    log_every = cfg["train"]["log_every"]
    t0 = time.time()

    pbar = tqdm(loader, total=n_iters, desc=f"epoch {epoch}", leave=False)
    for it, (images, labels) in enumerate(pbar):
        if it >= n_iters:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            out = model(images)
        loss, logs = criterion(out, labels)  # losses computed in float32

        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss became {loss.item()} at epoch {epoch} iter {it}. Parts: {logs}")

        scaler.scale(loss).backward()
        clip_value = cfg["train"].get("grad_clip")
        if clip_value:
            scaler.unscale_(optimizer)  # undo AMP scaling before clipping
            params = [p for g in optimizer.param_groups for p in g["params"]]
            logs["grad_norm"] = torch.nn.utils.clip_grad_norm_(params, clip_value).item()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            acc = (out["logits"].argmax(dim=1) == labels).float().mean().item() * 100
        logs["train_acc"] = acc
        meters.update(logs, n=labels.size(0))
        pbar.set_postfix(loss=f"{logs['loss_total']:.3f}", acc=f"{acc:.1f}")

        if log_every and it % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.log(f"epoch {epoch} iter {it}/{n_iters} lr {lr:.5f}  {meters}")

    stats = meters.averages()
    stats["epoch_time_s"] = time.time() - t0
    return stats


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train MPA-FER")
    p.add_argument("--config", default="configs/rafdb.yaml")
    p.add_argument("--set", dest="overrides", nargs="*", default=[],
                   help="config overrides, e.g. --set train.epochs=3 train.batch_size=16")
    p.add_argument("--eval_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    cfg = load_config(args.config, args.overrides)
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    logger = Logger(out_dir)
    save_config(cfg, out_dir)

    set_seed(cfg["seed"])
    device = get_device()
    use_amp = cfg["train"]["amp"] and device == "cuda"
    n_classes = len(cfg["data"]["class_names"])

    logger.log(f"experiment: {cfg['experiment_name']}")
    logger.log(f"device: {device}"
               + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else "")
               + f" | amp: {use_amp}")
    if args.overrides:
        logger.log(f"overrides: {args.overrides}")

    # ---------------- data ----------------
    train_loader, test_loader = build_loaders(cfg, seed=cfg["seed"])
    logger.log(f"train images: {len(train_loader.dataset)}  test images: {len(test_loader.dataset)}")
    logger.log(f"train class counts: {train_loader.dataset.class_counts()}")

    # ---------------- model ----------------
    model = build_model(cfg, device)
    for line in param_report(model).splitlines():
        logger.log(line)
    n_train_params = model.count_trainable_params()
    logger.log(f"trainable size: {n_train_params * 4 / 1024**2:.3f} MB in fp32 "
               f"(paper quotes 0.218 MB for ViT-B/16)")

    # ---------------- prototypes + loss ----------------
    prototypes = None
    if cfg["losses"]["proto_align"]["enabled"]:
        prototypes = load_or_build(cfg, model, device, logger=logger)
    criterion = build_criterion(cfg, prototypes, device)

    # ---------------- optimiser ----------------
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    max_iters = cfg["train"]["max_iters_per_epoch"]
    iters_per_epoch = len(train_loader) if max_iters is None else min(max_iters, len(train_loader))
    scheduler = build_scheduler(optimizer, cfg, iters_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---------------- resume ----------------
    start_epoch, best_acc = 1, 0.0
    resume_path = find_resume_checkpoint(out_dir) if cfg["output"]["resume"] else None
    if resume_path:
        state = load_checkpoint(resume_path, model, optimizer, scheduler, scaler)
        start_epoch = state["epoch"] + 1
        best_acc = state["best_acc"]
        logger.log(f"resumed from {resume_path} at epoch {state['epoch']} (best acc {best_acc:.2f})")

    # ---------------- eval only ----------------
    if args.eval_only:
        stats = evaluate(model, test_loader, device, n_classes, use_amp)
        logger.log(f"eval: acc {stats['acc']:.2f}  mean-class {stats['mean_class_acc']:.2f}")
        logger.log(f"per class: {dict(zip(cfg['data']['class_names'], stats['per_class_acc']))}")
        return

    # ---------------- train ----------------
    epochs = cfg["train"]["epochs"]
    logger.log(f"training for {epochs} epochs, {iters_per_epoch} iters/epoch")
    t_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        stats = train_one_epoch(model, train_loader, criterion, optimizer,
                                scheduler, scaler, device, cfg, epoch, logger)
        msg = "  ".join(f"{k} {v:.4f}" for k, v in stats.items() if k != "epoch_time_s")
        logger.log(f"epoch {epoch}/{epochs} done in {format_time(stats['epoch_time_s'])}  {msg}")

        record = {"epoch": epoch, **stats, "lr": optimizer.param_groups[0]["lr"]}

        if epoch % cfg["eval"]["every_n_epochs"] == 0 or epoch == epochs:
            eval_stats = evaluate(model, test_loader, device, n_classes, use_amp)
            record.update(eval_stats)
            logger.log(f"epoch {epoch} TEST acc {eval_stats['acc']:.2f}  "
                       f"mean-class {eval_stats['mean_class_acc']:.2f}  "
                       f"loss {eval_stats['test_loss']:.4f}")
            if eval_stats["acc"] > best_acc:
                best_acc = eval_stats["acc"]
                save_checkpoint(os.path.join(out_dir, "best.pt"), model, optimizer,
                                scheduler, scaler, epoch, best_acc, cfg)
                logger.log(f"new best: {best_acc:.2f} -> best.pt")

        logger.log_metrics(record)
        if epoch % cfg["output"]["save_every_n_epochs"] == 0:
            save_checkpoint(os.path.join(out_dir, "last.pt"), model, optimizer,
                            scheduler, scaler, epoch, best_acc, cfg)

    logger.log(f"finished in {format_time(time.time() - t_start)}. best test acc {best_acc:.2f}")


if __name__ == "__main__":
    main()