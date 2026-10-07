"""
DINO-ViT self-similarity structure guidance -- the second implementation of
"structure guidance" (see GenericDDPMWrapper._structure_guidance_correct),
replacing the first (raw high-frequency PIXEL residual matching,
structure_guidance_loss="pixel_high_freq") after diagnosing why that one
produced a "two images overlaid, opacity varying with scale" artifact rather
than genuine structure-preserving generation: matching raw pixel values
(even a high-pass-filtered residual) is mechanically a soft pixel COPY/BLEND
operation -- its gradient can literally inject source color/luminance
content into x, competing with (rather than reshaping) whatever the target
model is independently drawing. This is a known failure mode in the guided-
diffusion literature (pixel-space losses cause "pixel-blending" artifacts;
feature/semantic-space losses do not, since they have no channel through
which raw color can leak in).

DINO-ViT self-similarity (Tumanyan et al. "Splicing ViT Features for
Semantic Appearance Transfer", CVPR 2022; used identically for structure-
preserving diffusion guidance in DiffuseIT, Kwon & Ye, ICLR 2023) is
appearance-invariant BY CONSTRUCTION: the loss compares cosine-similarity
relationships between patch KEY vectors (which parts of the image resemble
which other parts), never raw pixel/color values, so its gradient cannot
inject source color -- it can only push the model to reshape its own
drawing so that ITS patches relate to each other the way the source
photo's patches do.

Model: facebookresearch/dino's ViT-S/8 (torch.hub, downloaded and cached at
~/.cache/torch/hub on first use). The feature-extraction path here is
identical to the (now-removed) diagnostics/dino_structure_metric.py used
earlier this session purely as an evaluation metric -- reused here because
it is already fully differentiable pure-PyTorch (F.interpolate, tensor
normalization, a forward hook, reshape/normalize -- no numpy round-trip),
so it needed no changes to also serve as a gradient-guidance loss.
"""
import torch
import torch.nn.functional as F

_dino_model = None
_dino_device = None
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_DINO_INPUT_SIZE = 224


def _get_dino_model(device):
    global _dino_model, _dino_device
    if _dino_model is None or _dino_device != device:
        model = torch.hub.load("facebookresearch/dino:main", "dino_vits8", pretrained=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _dino_model = model.to(device)
        _dino_device = device
    return _dino_model


def _extract_last_layer_keys(model, img01_batch, device):
    """img01_batch: (B, 3, H, W) in [0, 1]. Returns (B, N_patches, embed_dim)
    L2-normalized key vectors (CLS token excluded), from the last attention
    block's qkv projection via a forward hook. Fully differentiable w.r.t.
    img01_batch -- every op here (interpolate, normalize, the hooked linear
    layer, reshape, F.normalize) supports autograd; DINO's own parameters
    are frozen (requires_grad=False) but that only stops gradients from
    accumulating ON them, not gradients flowing THROUGH them back to the
    input, so this is safe to call inside a grad-enabled block without the
    generator's gradient-checkpointing workaround this session needed
    elsewhere (plain ViT forward here, no custom checkpoint Function)."""
    x = F.interpolate(img01_batch, size=(_DINO_INPUT_SIZE, _DINO_INPUT_SIZE), mode="bilinear", align_corners=False)
    x = (x - _IMAGENET_MEAN.to(device)) / _IMAGENET_STD.to(device)

    captured = {}

    def hook(module, inp, out):
        captured["qkv"] = out

    attn = model.blocks[-1].attn
    handle = attn.qkv.register_forward_hook(hook)
    try:
        model(x)
    finally:
        handle.remove()

    qkv = captured["qkv"]
    bsz, n_tokens, three_c = qkv.shape
    num_heads = attn.num_heads
    head_dim = three_c // 3 // num_heads
    qkv = qkv.reshape(bsz, n_tokens, 3, num_heads, head_dim).permute(2, 0, 1, 3, 4)
    keys = qkv[1]  # (B, N+1, num_heads, head_dim)
    keys = keys[:, 1:]  # drop CLS token -> (B, N, num_heads, head_dim)
    keys = keys.reshape(bsz, keys.shape[1], num_heads * head_dim)  # concat heads -> (B, N, embed_dim)
    keys = F.normalize(keys, dim=-1)
    return keys


def self_similarity(img01_batch, device):
    """img01_batch: (B, 3, H, W) in [0, 1]. Returns (B, N, N) cosine
    self-similarity matrices."""
    model = _get_dino_model(device)
    keys = _extract_last_layer_keys(model, img01_batch, device)
    return torch.bmm(keys, keys.transpose(1, 2))


def structure_guidance_loss(pred01, ref01, device):
    """pred01: (1, 3, H, W) in [0, 1], WITH gradients (this is what
    autograd.grad will differentiate w.r.t.). ref01: (1, 3, H, W) in [0, 1],
    the fixed source reference -- computed under no_grad since we never
    need gradients w.r.t. it, only w.r.t. pred01. Returns a scalar MSE loss
    between the two self-similarity matrices."""
    sim_pred = self_similarity(pred01, device)
    with torch.no_grad():
        sim_ref = self_similarity(ref01, device)
    return F.mse_loss(sim_pred, sim_ref)
