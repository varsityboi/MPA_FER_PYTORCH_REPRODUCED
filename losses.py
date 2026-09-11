"""
Losses for MPA-FER (paper Eq. 1-4, 7, 11, 12).

L_total = L_v_t + beta * L_t + gamma * L_v
    L_v_t : cross entropy on global + top-k local logits        (Eq. 11)
    L_t   : soft-hard prompt alignment = L_ta + L_pa            (Eq. 1-4)
    L_v   : prompted CLS feature pulled to its class prototype  (Eq. 7)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_hard_contrastive(soft, hard, tau):
    """
    Eq. 1-3: for each hard prompt d, the correct soft prompt must win out of all C.
        sim[d, c] = cos(soft_c, hard_d)
        loss      = cross_entropy(sim / tau, target = d)

    soft: (C, D)   hard: (C, D)
    Used twice: on pooled token embeddings (L_ta) and on text encoder outputs (L_pa).
    """
    soft = F.normalize(soft.float(), dim=-1)
    hard = F.normalize(hard.float(), dim=-1)
    sim = hard @ soft.t()                                   # (C, C): rows = hard d, cols = soft c
    targets = torch.arange(sim.shape[0], device=sim.device)
    return F.cross_entropy(sim / tau, targets)


def prototype_loss(image_global, labels, prototypes, match="cosine"):
    """
    Eq. 7: distance between each prompted CLS feature and the prototype of its true class.
    ASSUMPTION: averaged over the batch (the paper writes a sum, which grows with batch size).

    image_global: (B, D)   labels: (B,)   prototypes: (C, D)
    """
    z = image_global.float()
    p = prototypes[labels].float()                          # (B, D): prototype for each image's class
    if match == "cosine":
        return (1.0 - F.cosine_similarity(z, p, dim=-1)).mean()
    if match == "l1":
        return F.l1_loss(z, p)
    raise ValueError(f"Unknown match '{match}', use 'cosine' or 'l1'")


class MPAFERLoss(nn.Module):
    def __init__(
        self,
        soft_hard_enabled=True,
        beta=1.0,
        tau=0.01,
        proto_enabled=True,
        gamma=1.0,
        match="cosine",
        prototypes=None,
    ):
        super().__init__()
        self.soft_hard_enabled = soft_hard_enabled
        self.beta = beta
        self.tau = tau
        self.proto_enabled = proto_enabled
        self.gamma = gamma
        self.match = match

        if proto_enabled:
            if prototypes is None:
                raise ValueError(
                    "proto_align is enabled but no prototypes were given. "
                    "Run tools/build_prototypes.py first."
                )
            self.register_buffer("prototypes", prototypes.float())
        else:
            self.register_buffer("prototypes", None)

    def forward(self, out, labels):
        """
        out:    dict returned by MPAFER.forward()
        labels: (B,) class indices
        Returns the total loss (for backward) and a dict of plain numbers (for logging).
        """
        # (1) classification loss, Eq. 11
        loss_vt = F.cross_entropy(out["logits"].float(), labels)
        total = loss_vt
        logs = {"loss_vt": loss_vt.item()}

        # (2) soft-hard prompt alignment, Eq. 1-4
        if self.soft_hard_enabled:
            loss_ta = soft_hard_contrastive(out["soft_token_emb"], out["hard_token_emb"], self.tau)
            loss_pa = soft_hard_contrastive(out["text_features"], out["hard_text_features"], self.tau)
            total = total + self.beta * (loss_ta + loss_pa)
            logs["loss_ta"] = loss_ta.item()
            logs["loss_pa"] = loss_pa.item()

        # (3) prototype-guided visual alignment, Eq. 7
        if self.proto_enabled:
            loss_v = prototype_loss(out["image_global"], labels, self.prototypes, self.match)
            total = total + self.gamma * loss_v
            logs["loss_v"] = loss_v.item()

        logs["loss_total"] = total.item()
        return total, logs


def build_criterion(cfg, prototypes=None, device="cuda"):
    """Build the loss from the YAML config."""
    sh = cfg["losses"]["soft_hard_align"]
    pa = cfg["losses"]["proto_align"]
    criterion = MPAFERLoss(
        soft_hard_enabled=sh["enabled"],
        beta=sh["beta"],
        tau=sh["tau"],
        proto_enabled=pa["enabled"],
        gamma=pa["gamma"],
        match=pa["match"],
        prototypes=prototypes,
    )
    return criterion.to(device)