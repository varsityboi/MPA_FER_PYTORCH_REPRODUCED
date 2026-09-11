"""
Image side of MPA-FER (paper Sec. 3.4, Eq. 5).

Frozen CLIP ViT image encoder with deep visual prompts:
at every layer l, N_p learnable prompt tokens are appended to the sequence,
the frozen layer runs, and the prompt outputs are thrown away:

    [z^l, _] = layer_l([z^(l-1), p^l])

Gives back:
    global   z^g   final CLS token, projected    (B, embed_dim)       used in Eq. 7 and Eq. 10
    local    Z^l   final patch tokens, projected (B, 196, embed_dim)  used in Eq. 8
"""
import torch
import torch.nn as nn


class VisualPromptEncoder(nn.Module):
    def __init__(self, clip_model, n_prompts=8, init_std=0.02, enabled=True):
        super().__init__()
        visual = clip_model.visual

        # Frozen CLIP parts (frozen centrally in mpa_fer.py)
        self.conv1 = visual.conv1                              # cuts image into 16x16 patches
        self.class_embedding = visual.class_embedding          # CLS token
        self.positional_embedding = visual.positional_embedding
        self.ln_pre = visual.ln_pre
        self.resblocks = visual.transformer.resblocks          # the 12 transformer layers
        self.ln_post = visual.ln_post
        self.proj = visual.proj                                # 768 -> 512, same space as text

        self.enabled = enabled
        self.n_prompts = n_prompts
        self.n_layers = len(self.resblocks)
        width = self.conv1.out_channels                        # 768 for ViT-B/16

        # Learnable prompts: one set of N_p tokens per layer, Gaussian init (paper)
        if enabled:
            prompts = torch.empty(self.n_layers, n_prompts, width)
            nn.init.normal_(prompts, std=init_std)
            self.prompts = nn.Parameter(prompts)
        else:
            self.prompts = None

    def forward(self, images, use_prompts=True):
        """
        images: (B, 3, 224, 224)
        use_prompts=False gives plain frozen CLIP features (used to build prototypes).
        """
        use_prompts = use_prompts and self.prompts is not None

        # ---- image -> patch tokens ----
        x = self.conv1(images.to(self.conv1.weight.dtype))     # (B, 768, 14, 14)
        B = x.shape[0]
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)      # (B, 196, 768)

        # ---- add CLS token + position info ----
        cls = self.class_embedding.to(x.dtype).expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)                         # (B, 197, 768)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        n_tokens = x.shape[1]                                  # 197

        # ---- 12 frozen layers, fresh prompts at each one ----
        for i, block in enumerate(self.resblocks):
            if use_prompts:
                p = self.prompts[i].to(x.dtype).unsqueeze(0).expand(B, -1, -1)  # (B, 8, 768)
                x = torch.cat([x, p], dim=1)                   # (B, 205, 768)

            x = x.permute(1, 0, 2)                             # CLIP layers want (seq, batch, dim)
            x = block(x)
            x = x.permute(1, 0, 2)                             # back to (batch, seq, dim)

            if use_prompts:
                x = x[:, :n_tokens, :]                         # throw away prompt outputs

        # ---- project CLS and patches into the shared image-text space ----
        # ASSUMPTION: patches go through the same ln_post + proj as CLS,
        # so they can be compared with text features (paper does not say how).
        x = self.ln_post(x)
        x = x @ self.proj                                      # (B, 197, 512)

        return {
            "global": x[:, 0, :],                              # z^g  (B, 512)
            "local": x[:, 1:, :],                              # Z^l  (B, 196, 512)
        }