"""
FBSDiff frequency-band replacement (Gao & Liu, "FBSDiff: Plug-and-Play
Frequency Band Substitution of Diffusion Features for Highly Controllable
Text-Driven Image Translation", ACM MM 2024, arXiv:2408.00998) -- Task 3 of
the ILVR-failure-mode-driven diagnosis plan.

Where ILVR (diagnostics/generic_ddpm_wrapper.py's _ilvr_correct) forces a
static PIXEL-SPACE low-pass replacement at every free decode step -- shown by
Task 0's diagnosis (diagnostics/ilvr_failure_mode.md) to suppress target-model
stylization to well under half of baseline at every constraint strength
tested, even the loosest -- FBSDiff instead substitutes a selected FREQUENCY
BAND (via 2D-DCT, not a spatial blur) between two decode trajectories that
run in lockstep from a SHARED deterministically-inverted latent (see
GenericDDPMWrapper.deterministic_invert, Task 4):
  - the "reconstruction trajectory": the SOURCE model decoding that shared
    latent (eta=0), which (if inversion is accurate) closely retraces the
    source photo's own structure.
  - the "sampling trajectory": the TARGET model decoding the same shared
    latent, which is what actually gets stylized and returned.
Only during an initial "calibration phase" (the first `1 - fbs_lambda`
fraction of decode steps, i.e. while noise is still high) is a frequency band
of the sampling trajectory's current latent overwritten with that same band
from the reconstruction trajectory; after the calibration phase, the target
decodes completely freely, so stylization is not fought at every step the way
ILVR fights it.

Threshold scale note (REVISED after empirical testing -- see
diagnostics/outputs/step_g4..g16): this module originally rescaled the
paper's th_lp=80/th_hp=5 (calibrated for a 64x64 LDM latent, max DCT
coordinate-sum (u+v)=126) by the same ~4x ratio as this project's 256x256
pixel-space decode (max coordinate-sum=510), i.e. th_lp=324, preserving the
paper's FRACTION of the spectrum (80/126 = 324/510 = 63.5%). That rescaling
turned out to be the wrong analogy: a VAE latent's low frequencies encode
mostly SEMANTIC/structural content (fine repeating texture is discarded by
the encoder before the diffusion model ever sees it), so a 63%-of-spectrum
swap there only ever touches structure. A raw 256x256 PHOTO's low frequencies
(same 63% cutoff) still carry real pattern/texture energy (a patterned
garment or leafy background has fundamental frequencies well within that
band) -- substituting that into the anime target's flat-shaded generative
process for hundreds of consecutive steps compounds into a persistent
hatched/textured artifact (confirmed reproducible regardless of hard vs.
soft mask edges or per-step blend strength -- see _BLEND_STRENGTH below --
ruling those out; only a much NARROWER absolute threshold removed it).
DEFAULT_THRESHOLDS below therefore uses the paper's raw th_lp/th_hp
MAGNITUDES directly (not rescaled by resolution), which empirically gave a
clean result on both a low-detail photo and a heavily patterned one.
Residual grain on ONE image (data/ffhq/50494.png) persisted at every
threshold tested (10 through 324) -- that image was independently flagged
earlier this session (diagnostics/outputs/summary_report_stepE_translation.md)
as a structurally hard case for this checkpoint pair regardless of method
(collapses the same way under plain CycleDiffusion and under ILVR too), so
this looks like that same known difficulty surfacing under FBS, not a new bug.
"mid" has no paper-given default and was NOT part of this validation (see
frequency_band_mask's mid branch -- it excludes low frequencies entirely,
so it does not transfer structure and should be considered unvalidated).
"""
import numpy as np
import torch
from scipy.fft import dctn, idctn

# Paper's raw th_lp=80/th_hp=5, used directly (NOT rescaled by resolution --
# see the threshold scale note above for why the earlier rescaled version,
# th_lp=324, caused a persistent hatched artifact on detailed/patterned
# photos). "mid" has no paper-given default and is unvalidated (see module
# docstring).
DEFAULT_THRESHOLDS = {"low": 10.0, "high": 5.0, "mid": 60.0}


def _coord_sum_grid(h, w):
    u = np.arange(h).reshape(-1, 1)
    v = np.arange(w).reshape(1, -1)
    return u + v


# Width (in coordinate-sum units) of the smooth sigmoid taper at each mask
# edge -- see frequency_band_mask's docstring for why this must not be 0
# (a hard cutoff). Not exposed as a CLI knob: like the ILVR bicubic->
# area/bilinear interpolation fix earlier this session, this addresses a
# ringing ARTIFACT, not a tunable creative parameter, so there is no
# legitimate reason to want the hard-edged (buggy) version back.
_SOFT_EDGE_WIDTH = 15.0


def frequency_band_mask(h, w, mode, threshold, soft_width=_SOFT_EDGE_WIDTH):
    """Float (h, w) numpy mask in [0, 1] selecting how much of each 2D-DCT
    coefficient comes from the reconstruction trajectory (1) vs. the
    sampling trajectory (0) for the requested band. `threshold` is a
    coordinate-sum (u+v) cutoff -- NOT a raw frequency in cycles/pixel --
    consistent with the paper's th_lp/th_hp/th_mp convention (low
    frequencies cluster near u=v=0, i.e. small coordinate-sum; high
    frequencies near the opposite corner).

    Uses a smooth logistic taper across `soft_width`, not a hard boolean
    cutoff: a hard cutoff is a brick-wall filter in frequency space, which
    causes Gibbs-phenomenon ringing when transformed back to the spatial
    domain -- reapplied every calibration-phase step (hundreds of times),
    that ringing compounds into visible periodic grain/texture, exactly
    parallel to the bicubic-interpolation ringing bug found and fixed in
    ILVR's _ilvr_low_pass (diagnostics/generic_ddpm_wrapper.py) earlier this
    session. Confirmed empirically: the hard-cutoff version produced the
    same grain artifact at threshold=324 AND threshold=40 alike (ruling out
    "band too wide" as the cause) -- see diagnostics/outputs/step_g5..g7.
    """
    coord_sum = _coord_sum_grid(h, w).astype(np.float64)
    max_sum = (h - 1) + (w - 1)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    if mode == "low":
        mask = sigmoid((threshold - coord_sum) / soft_width)
    elif mode == "high":
        mask = sigmoid((coord_sum - (max_sum - threshold)) / soft_width)
    elif mode == "mid":
        mask = sigmoid((coord_sum - threshold) / soft_width) * sigmoid(((max_sum - threshold) - coord_sum) / soft_width)
    else:
        raise ValueError(f"unknown fbs mode: {mode!r}")
    return mask


# Per-step blend strength for the selected band (0 = no effect, 1 = full
# per-paper replacement). NOTE on what this did and didn't fix: originally
# added while chasing the grain artifact under the (WRONG) rescaled
# threshold=324 -- at that threshold, neither this dampening nor annealing
# it to 0 across the calibration window (see translate_one_deterministic)
# removed the grain; only narrowing the threshold itself did (see the
# module docstring's "Threshold scale note"). Left at a fractional value
# anyway as a conservative safety margin against the same general class of
# failure (strong per-step diffusion guidance held at full strength for
# hundreds of consecutive steps fighting the model's own trajectory) on
# images/thresholds not covered by this session's testing -- not because
# it was the confirmed fix. Not exposed as a CLI knob for the same reason
# the threshold defaults aren't tunable-by-default: this is a bias
# correction, not a creative tradeoff.
_BLEND_STRENGTH = 0.3


def replace_frequency_band(sampling_x, reconstruction_x, mode, threshold, strength=_BLEND_STRENGTH):
    """sampling_x, reconstruction_x: (C, H, W) torch tensors on any
    device/dtype, same shape. `strength` overrides _BLEND_STRENGTH (e.g. for
    an annealed per-step schedule -- see translate_one_deterministic, which
    tapers strength across the calibration window rather than holding it
    constant then cutting it off abruptly, to give the model more room to
    reconcile the substituted band with its own trajectory before free
    decoding resumes). Returns (out, mask_is_all_true):
      out: (C, H, W) tensor, same device/dtype as sampling_x, blended
        towards reconstruction_x's selected DCT frequency band (see
        frequency_band_mask for the band shape), per channel independently
        (each channel's 2D-DCT/IDCT is computed separately -- the standard
        per-channel treatment for RGB frequency-domain operations, avoiding
        cross-channel leakage).
      mask_is_all_true: True if the BAND SHAPE (before strength scaling) is
        >=0.999 EVERYWHERE -- i.e. threshold is misconfigured to select the
        entire spectrum, not that any single step fully overwrites
        sampling_x (blend strength is a separate, intentional design
        choice, not a misconfiguration). Caller should warn on this exactly
        once.
    """
    assert sampling_x.shape == reconstruction_x.shape
    c, h, w = sampling_x.shape
    shape_mask = frequency_band_mask(h, w, mode, threshold)
    mask_is_all_true = bool(np.all(shape_mask >= 0.999))
    mask = strength * shape_mask

    device, dtype = sampling_x.device, sampling_x.dtype
    samp_np = sampling_x.detach().cpu().float().numpy()
    recon_np = reconstruction_x.detach().cpu().float().numpy()
    out_np = np.empty_like(samp_np)
    for ch in range(c):
        samp_dct = dctn(samp_np[ch], norm="ortho")
        recon_dct = dctn(recon_np[ch], norm="ortho")
        merged_dct = mask * recon_dct + (1.0 - mask) * samp_dct
        out_np[ch] = idctn(merged_dct, norm="ortho")

    out = torch.from_numpy(out_np).to(device=device, dtype=dtype)
    return out, mask_is_all_true
