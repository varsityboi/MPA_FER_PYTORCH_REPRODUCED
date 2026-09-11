"""
Text side of MPA-FER (paper Sec. 3.2 - 3.3).

Soft prompt:  [SOS] [ctx_1 ... ctx_10] [class name] . [EOS]    (ctx = learnable)
Hard prompt:  template + class name (+ LLM description)        (frozen, encoded once)

Gives back everything the soft-hard alignment losses need:
    soft_token_emb       t_c        pooled token embeddings of soft prompts   (Eq. 1-2)
    hard_token_emb       t*_c       pooled token embeddings of hard prompts   (Eq. 1-2)
    text_features        θ(t_c)     soft prompts after CLIP text encoder      (Eq. 3)
    hard_text_features   θ(t*_c)    hard prompts after CLIP text encoder      (Eq. 3)
"""
import json
import warnings

import torch
import torch.nn as nn
import clip
from clip.simple_tokenizer import SimpleTokenizer

CONTEXT_LENGTH = 77  # CLIP text encoder can only read 77 tokens
_tokenizer = SimpleTokenizer()


# ------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------
def load_descriptions(path):
    """Read the LLM descriptions from prompts/rafdb_descriptions.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["descriptions"]


def build_hard_prompts(class_names, template, descriptions=None):
    """Fill the template for every class, e.g. 'a photo of {cls}, {desc}'."""
    needs_desc = "{desc}" in template
    prompts = []
    for name in class_names:
        if needs_desc:
            if descriptions is None or name not in descriptions:
                raise KeyError(f"No description found for class '{name}'")
            prompts.append(template.format(cls=name, desc=descriptions[name]))
        else:
            prompts.append(template.format(cls=name))
    return prompts


def tokenize_with_check(texts):
    """Tokenize, and warn if a sentence is too long for CLIP (it gets cut)."""
    for t in texts:
        n_tokens = len(_tokenizer.encode(t)) + 2  # +2 for SOS and EOS
        if n_tokens > CONTEXT_LENGTH:
            warnings.warn(
                f"Prompt has {n_tokens} tokens (max {CONTEXT_LENGTH}) and will be cut: '{t[:60]}...'"
            )
    return clip.tokenize(texts, context_length=CONTEXT_LENGTH, truncate=True)


def mean_pool_tokens(token_emb, tokens):
    """
    Average the embeddings of the real word tokens (skip SOS, EOS, padding).
    ASSUMPTION: the paper compares token embeddings in Eq. 1 but soft and hard
    prompts have different lengths, so we average them into one vector each.

    token_emb: (C, 77, D)   tokens: (C, 77)   ->   (C, D)
    """
    eos_idx = tokens.argmax(dim=-1)  # EOS has the largest token id in CLIP
    positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    mask = (positions > 0) & (positions < eos_idx.unsqueeze(1))
    mask = mask.unsqueeze(-1).to(token_emb.dtype)
    return (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


# ------------------------------------------------------------------
# Frozen CLIP text encoder that accepts embeddings instead of words
# ------------------------------------------------------------------
class TextEncoder(nn.Module):
    """
    Same as CLIP's encode_text, except it takes token *embeddings* as input.
    Needed because our soft prompts are learnable vectors, not real words.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, token_emb, tokens):
        x = token_emb + self.positional_embedding.to(token_emb.dtype)
        x = x.permute(1, 0, 2)  # (N, L, D) -> (L, N, D): CLIP's transformer wants sequence first
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # back to (N, L, D)
        x = self.ln_final(x)
        eos_idx = tokens.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eos_idx]  # take the EOS token
        return x @ self.text_projection  # (N, embed_dim)


# ------------------------------------------------------------------
# Soft prompts (learnable) + hard prompts (frozen)
# ------------------------------------------------------------------
class TextPromptLearner(nn.Module):
    def __init__(
        self,
        clip_model,
        class_names,
        n_ctx=10,
        init_std=0.02,
        class_specific_ctx=False,
        hard_template="a photo of {cls}",
        descriptions=None,
        token_pooling="mean",
    ):
        super().__init__()
        if token_pooling != "mean":
            raise ValueError("Only token_pooling='mean' is implemented")

        self.class_names = list(class_names)
        self.n_cls = len(self.class_names)
        self.n_ctx = n_ctx
        self.text_encoder = TextEncoder(clip_model)

        token_embedding = clip_model.token_embedding
        width = token_embedding.weight.shape[1]
        device = token_embedding.weight.device

        # ---------------- soft prompts ----------------
        # Learnable context vectors, Gaussian init (mean 0, std 0.02) as in the paper
        if class_specific_ctx:
            ctx = torch.empty(self.n_cls, n_ctx, width)  # one set per class
        else:
            ctx = torch.empty(n_ctx, width)  # one set shared by all classes
        nn.init.normal_(ctx, std=init_std)
        self.ctx = nn.Parameter(ctx)

        # Trick from CoOp: tokenize "X X X ... X happiness." so we know where
        # the class name, EOS and padding sit. The X's get replaced by self.ctx.
        placeholder = " ".join(["X"] * n_ctx)
        soft_texts = [f"{placeholder} {name}." for name in self.class_names]
        soft_tokens = tokenize_with_check(soft_texts).to(device)
        with torch.no_grad():
            soft_emb = token_embedding(soft_tokens)
        self.register_buffer("soft_tokens", soft_tokens, persistent=False)
        self.register_buffer("token_prefix", soft_emb[:, :1, :].float(), persistent=False)  # SOS
        self.register_buffer("token_suffix", soft_emb[:, 1 + n_ctx:, :].float(), persistent=False)  # class . EOS pad

        # ---------------- hard prompts ----------------
        # Frozen, so encode once here and store the results
        self.hard_texts = build_hard_prompts(self.class_names, hard_template, descriptions)
        hard_tokens = tokenize_with_check(self.hard_texts).to(device)
        with torch.no_grad():
            hard_emb = token_embedding(hard_tokens)
            hard_features = self.text_encoder(hard_emb, hard_tokens)
        self.register_buffer("hard_token_emb", mean_pool_tokens(hard_emb, hard_tokens).float(), persistent=False)
        self.register_buffer("hard_text_features", hard_features.float(), persistent=False)

    def build_soft_prompts(self):
        """Glue [SOS] + learnable ctx + [class name . EOS padding] -> (C, 77, width)."""
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix.to(ctx.dtype)
        suffix = self.token_suffix.to(ctx.dtype)
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self):
        soft_emb = self.build_soft_prompts()
        return {
            "text_features": self.text_encoder(soft_emb, self.soft_tokens),  # θ(t_c)   (C, embed_dim)
            "soft_token_emb": mean_pool_tokens(soft_emb, self.soft_tokens),  # t_c      (C, width)
            "hard_text_features": self.hard_text_features,                  # θ(t*_c)  (C, embed_dim), frozen
            "hard_token_emb": self.hard_token_emb,                          # t*_c     (C, width), frozen
        }