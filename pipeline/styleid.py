"""
StyleID-style self-attention K/V injection (Chung et al., "Style Injection in
Diffusion: A Training-free Approach for Adapting Large-scale Diffusion Models
for Style Transfer", CVPR 2024, https://github.com/jiwoogit/StyleID), adapted
to this repo's pixel-space EDM2 MPUNetModel (model/lib/ddpm_ddim/models/
improved_ddpm/unet_mp.py) instead of Stable Diffusion's latent U-Net.

Mechanism (all three parts from the paper):
  1. Both the content photo and an anime style reference are DDIM-inverted
     (eta=0) to t_0. At every step the target (anime) model's decoder
     self-attention Q of the content trajectory and K/V of the style
     trajectory are recorded, keyed by (timestep, block name).
  2. Generation starts from the content's inverted latent and, in the chosen
     decoder blocks, replaces attention's K/V with the style's, while the
     query is "preserved": q = gamma * Q_content + (1 - gamma) * q_current.
     Q decides WHERE each position attends (layout), K/V decide WHAT gets
     pulled in (appearance), so the photo's layout is kept while the anime
     reference's appearance is copied in.
  3. Attention temperature tau > 1 sharpens the softmax, compensating for the
     lower q.k similarity between content queries and style keys. The MP
     U-Net's attention is cosine (q, k pixel-normalized per head), so this is
     applied as q * tau, i.e. softmax(tau * q.k / sqrt(c)).
  Plus optional initial-latent AdaIN: x_T's per-channel mean/std is set to the style
  latent's, which the paper uses to fix color harmony (off by default here --
  see translate_styleid.py's --adain for why).

Why the TARGET model records the content Q too (not the FFHQ model): Q and
K/V must live in the same projection space for q.k to be meaningful, so all
injected features come from the target model's own attn_qkv. The content
trajectory itself can still come from either model's DDIM inversion -- see
invert_and_record()'s record_model/invert_model split.
"""
import numpy as np
import torch

from model.gan_wrapper.ddpm_ddim_wrapper import denoising_step_with_eps
from model.lib.ddpm_ddim.utils.diffusion_utils import denoising_step


class StyleIDInjector:
    """Installs MPBlock.attn_hook on every attention-bearing decoder block of
    `generator` whose name ("dec.<res>x<res>_<block>") starts with one of
    `layer_prefixes`. Callers drive it by setting `mode` and `t`:
      - "record_content": store each hooked block's q under (t, name)
      - "record_style":   store each hooked block's k, v under (t, name)
      - "inject":         replace q/k/v as described in the module docstring
      - "off":            pass through unchanged
    """

    def __init__(self, generator, layer_prefixes, gamma=0.75, tau=1.5):
        self.gamma = gamma
        self.tau = tau
        self.mode = "off"
        self.t = None
        self.content = {}
        self.style = {}
        self._names = {}
        for name, blk in generator.dec.items():
            full = f"dec.{name}"
            if getattr(blk, "num_heads", 0) and any(full.startswith(p) for p in layer_prefixes):
                self._names[blk] = full
                blk.attn_hook = self._hook
        if not self._names:
            raise ValueError(
                f"no attention block in generator.dec matches {layer_prefixes} -- available: "
                f"{[f'dec.{n}' for n, b in generator.dec.items() if getattr(b, 'num_heads', 0)]}"
            )

    @property
    def layer_names(self):
        return sorted(self._names.values())

    def _hook(self, blk, q, k, v):
        key = (self.t, self._names[blk])
        if self.mode == "record_content":
            self.content[key] = q.detach()
        elif self.mode == "record_style":
            self.style[key] = (k.detach(), v.detach())
        elif self.mode == "inject":
            k, v = self.style[key]
            q = self.gamma * self.content[key] + (1.0 - self.gamma) * q
            q = q * self.tau
        return q, k, v

    def remove(self):
        for blk in self._names:
            blk.attn_hook = None
        self._names = {}


def timestep_sequence(t_0, num_steps):
    """Increasing ints 0 .. t_0, shared by inversion and generation so every
    generation step finds features recorded at exactly its own t."""
    return [int(s) for s in np.linspace(0, t_0, num_steps)]


def _t(value, bsz, device):
    return (torch.ones(bsz) * value).to(device)


@torch.no_grad()
def invert_and_record(invert_model, record_model, injector, mode, x0, seq, betas, logvar, learn_sigma):
    """Deterministic DDIM inversion (eta=0) of x0 ([-1, 1]) along `seq` with
    `invert_model`, recording `record_model`'s features (hooked by `injector`)
    at every t in `seq`. When the two are the same model, the inversion's own
    forward passes do the recording; otherwise `record_model` gets one extra
    forward per step on the same latent (only the target model is hooked, so
    the inverting model's passes never record). Returns x at t = seq[-1]."""
    bsz, device = x0.shape[0], x0.device
    x = x0
    injector.mode = mode
    try:
        for idx in range(len(seq) - 1):
            injector.t = seq[idx]
            t_cur = _t(seq[idx], bsz, device)
            if record_model is not invert_model:
                record_model(x, t_cur)
            x = denoising_step_with_eps(
                x, eps=torch.zeros_like(x), t=t_cur, t_next=_t(seq[idx + 1], bsz, device),
                models=invert_model, logvars=logvar, b=betas, sampling_type="ddim", eta=0.0,
                learn_sigma=learn_sigma,
            )
        # Generation's first step evaluates the model at seq[-1], which the
        # loop above never did.
        injector.t = seq[-1]
        record_model(x, _t(seq[-1], bsz, device))
    finally:
        injector.mode = "off"
    return x


def adain(content, style, eps=1e-5):
    """Per-channel mean/std of `content` replaced with `style`'s."""
    c_mean, c_std = content.mean(dim=(2, 3), keepdim=True), content.std(dim=(2, 3), keepdim=True)
    s_mean, s_std = style.mean(dim=(2, 3), keepdim=True), style.std(dim=(2, 3), keepdim=True)
    return (content - c_mean) / (c_std + eps) * s_std + s_mean


@torch.no_grad()
def generate(target_model, injector, x_T, seq, betas, logvar, learn_sigma, inject=True):
    """Deterministic DDIM decode (eta=0) from x_T at seq[-1] down to seq[0]
    with `target_model`, injecting recorded features at every step when
    `inject`. Returns x at t = seq[0] (= 0), in [-1, 1]."""
    bsz, device = x_T.shape[0], x_T.device
    x = x_T
    injector.mode = "inject" if inject else "off"
    try:
        for idx in range(len(seq) - 1, 0, -1):
            injector.t = seq[idx]
            x = denoising_step(
                x, t=_t(seq[idx], bsz, device), t_next=_t(seq[idx - 1], bsz, device),
                models=target_model, logvars=logvar, b=betas, sampling_type="ddim", eta=0.0,
                learn_sigma=learn_sigma,
            )
    finally:
        injector.mode = "off"
    return x
