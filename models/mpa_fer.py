"""
Full MPA-FER model (paper Fig. 2, Eq. 8-10).

Loads CLIP, freezes it, joins the text prompts (text_prompts.py) and the
visual prompts (visual_prompts.py), and computes class logits:

    sim(image, class) = cos(z^g, θ(t_c))                         global
                      + mean of top-k cos(z^l_i, θ(t_c)) over i  local (Eq. 8-9)
    logits = logit_scale * sim                                   (Eq. 10)

Only 2 tensors are trainable: text_prompts.ctx and visual_encoder.prompts.
"""
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.text_prompts import TextPromptLearner, load_descriptions
from models.visual_prompts import VisualPromptEncoder

TRAINABLE_PARAMS = ("text_prompts.ctx", "visual_encoder.prompts")


class MPAFER(nn.Module):
    def __init__(
        self,
        clip_model,
        class_names,
        hard_template,
        descriptions=None,
        n_ctx=10,
        text_init_std=0.02,
        class_specific_ctx=False,
        token_pooling="mean",
        visual_prompts_enabled=True,
        n_visual_prompts=8,
        visual_init_std=0.02,
        local_align_enabled=True,
        top_k=16,
        local_weight=1.0,
        logit_scale=100.0,
    ):
        super().__init__()
        self.class_names = list(class_names)
        self.local_align_enabled = local_align_enabled
        self.top_k = top_k
        self.local_weight = local_weight
        self.logit_scale = logit_scale

        self.text_prompts = TextPromptLearner(
            clip_model,
            class_names,
            n_ctx=n_ctx,
            init_std=text_init_std,
            class_specific_ctx=class_specific_ctx,
            hard_template=hard_template,
            descriptions=descriptions,
            token_pooling=token_pooling,
        )
        self.visual_encoder = VisualPromptEncoder(
            clip_model,
            n_prompts=n_visual_prompts,
            init_std=visual_init_std,
            enabled=visual_prompts_enabled,
        )
        self.freeze_clip()

    # --------------------------------------------------------------
    # Freezing and saving only what we train
    # --------------------------------------------------------------
    def freeze_clip(self):
        """Everything frozen except the prompts."""
        for name, p in self.named_parameters():
            p.requires_grad_(name in TRAINABLE_PARAMS)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def trainable_state_dict(self):
        """Only the prompts. Keeps checkpoints tiny instead of saving all of CLIP."""
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state_dict(self, state):
        own = dict(self.named_parameters())
        for name, value in state.items():
            if name not in own:
                raise KeyError(f"Checkpoint has '{name}' but the model doesn't. Config mismatch?")
            own[name].data.copy_(value.to(own[name].device, own[name].dtype))

    # --------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------
    def compute_logits(self, image_out, text_features):
        """Eq. 8-10: global similarity + top-k local similarity."""
        img_g = F.normalize(image_out["global"].float(), dim=-1)     # (B, D)
        txt = F.normalize(text_features.float(), dim=-1)             # (C, D)
        sim = img_g @ txt.t()                                        # (B, C)

        if self.local_align_enabled:
            # ASSUMPTION: patch features are L2-normalized, so <.,.> in Eq. 8 is cosine
            img_l = F.normalize(image_out["local"].float(), dim=-1)  # (B, N, D)
            local_sim = torch.einsum("bnd,cd->bnc", img_l, txt)      # (B, N, C)
            k = min(self.top_k, local_sim.shape[1])
            topk_sim = local_sim.topk(k, dim=1).values.mean(dim=1)   # (B, C), top-k picked per class
            sim = sim + self.local_weight * topk_sim

        return self.logit_scale * sim

    def forward(self, images):
        text_out = self.text_prompts()

        if self.visual_encoder.prompts is None:
            # No visual prompts = nothing to train on image side, skip gradients (faster)
            with torch.no_grad():
                image_out = self.visual_encoder(images, use_prompts=False)
        else:
            image_out = self.visual_encoder(images, use_prompts=True)

        return {
            "logits": self.compute_logits(image_out, text_out["text_features"]),  # (B, C)
            "image_global": image_out["global"],   # prompted z^g, for prototype loss (Eq. 7)
            **text_out,                            # text_features, soft/hard token emb, hard_text_features
        }

    @torch.no_grad()
    def encode_frozen_image(self, images):
        """Plain CLIP CLS feature, no prompts. Used to build prototypes (Eq. 6)."""
        return self.visual_encoder(images, use_prompts=False)["global"]


# ------------------------------------------------------------------
# Build the model from the YAML config
# ------------------------------------------------------------------
def get_hard_template(cfg):
    hp = cfg["hard_prompts"]
    templates = {int(k): v for k, v in hp["templates"].items()}
    t = int(hp["type"])
    if t not in templates:
        raise ValueError(f"hard_prompts.type={t} but templates only has {list(templates)}")
    return templates[t]


def build_model(cfg, device="cuda"):
    mcfg = cfg["model"]

    # Load on CPU in float32 (loading on GPU would turn CLIP into fp16; we use AMP instead)
    clip_model, _ = clip.load(mcfg["clip_backbone"], device="cpu", jit=False)
    clip_model = clip_model.float()

    template = get_hard_template(cfg)
    descriptions = None
    if "{desc}" in template:
        descriptions = load_descriptions(cfg["hard_prompts"]["descriptions_file"])

    model = MPAFER(
        clip_model,
        class_names=cfg["data"]["class_names"],
        hard_template=template,
        descriptions=descriptions,
        n_ctx=mcfg["text_prompts"]["n_ctx"],
        text_init_std=mcfg["text_prompts"]["init_std"],
        class_specific_ctx=mcfg["text_prompts"]["class_specific_ctx"],
        token_pooling=cfg["losses"]["soft_hard_align"]["token_pooling"],
        visual_prompts_enabled=mcfg["visual_prompts"]["enabled"],
        n_visual_prompts=mcfg["visual_prompts"]["n_prompts"],
        visual_init_std=mcfg["visual_prompts"]["init_std"],
        local_align_enabled=mcfg["local_align"]["enabled"],
        top_k=mcfg["local_align"]["top_k"],
        local_weight=mcfg["local_align"].get("local_weight", 1.0),
        logit_scale=mcfg["logit_scale"],
    )
    return model.to(device)