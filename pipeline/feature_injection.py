"""
Plug-and-Play-style decoder ResBlock feature injection -- "A-1" of the
structure-preserving-translation plan (Tumanyan et al., "Plug-and-Play
Diffusion Features for Text-Driven Image-to-Image Translation", CVPR 2023,
https://github.com/MichalGeyer/plug-and-play).

Root motivation: ILVR (pixel low-pass) and FBSDiff (DCT frequency-band
substitution) both operate on RAW SPATIAL/FREQUENCY content, with no notion
of image semantics -- shown this session to either suppress stylization
(ILVR, diagnostics/ilvr_failure_mode.md) or, when tuned to actually transfer
structure, still let facial geometry drift more than the user wants for
this project's goal (surface texture/color-fill/edge-emphasis transfer with
the exact face/hair shape held fixed). Plug-and-Play instead injects the
SOURCE model's own decoder ResBlock activations -- an intermediate
representation that already encodes precise spatial layout -- directly into
the TARGET model's corresponding layer, for a calibration-phase fraction of
steps, matching this project's already-established --deterministic_inversion
+ dual-trajectory decode pattern (see
diagnostics/step_e12_translate_matched_arch.py's translate_one_deterministic,
originally built for Task 3/FBSDiff and reused here). Because both models
share IDENTICAL architecture (confirmed earlier this session via direct
checkpoint tensor-shape inspection), the same output_blocks[idx] index names
the same role/resolution in both models, so injecting one model's ResBlock
output directly into the other's forward pass is well-defined -- no
resizing or remapping needed.

Scope note: only ResBlock spatial-feature injection is implemented here.
The original paper also supports self-attention MAP injection (injecting
just the attention weights while keeping the target's own values) as a
secondary refinement; that requires intercepting QKVAttentionLegacy's
internal Q/K/V split (see model/lib/ddpm_ddim/models/improved_ddpm/unet.py's
QKVAttentionLegacy.forward), which is not implemented in this pass -- the
paper itself treats ResBlock feature injection as carrying most of the
structural-preservation effect, with attention injection as an additional
refinement, so this is a reasonable place to stop for a first validated
version rather than an accuracy gap.
"""
import torch


class ResBlockFeatureInjector:
    """Hooks a SOURCE and a TARGET GenericDDPMWrapper's identical-architecture
    `.generator` (a UNetModel) at the same `output_blocks` (decoder) layer
    indices. While `injection_enabled` is True, the target's ResBlock output
    at each hooked layer is REPLACED by whatever the source's same-layer
    ResBlock most recently produced (captured automatically on every source
    forward pass) -- both models must be run in lockstep (same t at the same
    step) by the caller, exactly as translate_one_deterministic's dual
    source/target decode loop already does for FBSDiff.

    injection_enabled defaults to False and is the caller's responsibility to
    toggle per decode step (e.g. only during a calibration-phase window) --
    this class only performs the capture/replace mechanics, not scheduling.

    injection_strength (default 1.0 = full replacement, matching this
    class's original behavior) blends towards the source's captured feature
    rather than fully overwriting the target's own: out = (1-strength)*target
    + strength*source. Added after a parameter search found NO (layer,
    lambda) combination using full-strength (1.0) replacement that preserved
    both contour AND facial-feature formation at once -- every duration long
    enough to hold contour (lambda <~0.5) also fully suppressed facial detail
    (diagnostics/outputs/step_h5_sweep3) -- so a longer HOLD at PARTIAL
    strength (rather than a short hold at full strength) is the one lever in
    this mechanism not yet tried at the time of writing."""

    def __init__(self, source_model, target_model, layer_indices):
        self.layer_indices = list(layer_indices)
        self.injection_enabled = False
        self.injection_strength = 1.0
        self._captured = {}
        self._handles = []
        for idx in self.layer_indices:
            source_resblock = source_model.generator.output_blocks[idx][0]
            target_resblock = target_model.generator.output_blocks[idx][0]
            assert type(source_resblock).__name__ == "ResBlock", (
                f"output_blocks[{idx}][0] is {type(source_resblock).__name__}, expected ResBlock -- "
                f"check --pnp_feature_layers against the model's actual output_blocks layout."
            )
            assert type(target_resblock).__name__ == "ResBlock"
            self._handles.append(source_resblock.register_forward_hook(self._make_capture_hook(idx)))
            self._handles.append(target_resblock.register_forward_hook(self._make_inject_hook(idx)))

    def _make_capture_hook(self, idx):
        def hook(module, inp, out):
            self._captured[idx] = out.detach()
        return hook

    def _make_inject_hook(self, idx):
        def hook(module, inp, out):
            if not self.injection_enabled:
                return out
            captured = self._captured.get(idx)
            if captured is None or captured.shape != out.shape:
                # Shape mismatch would mean source/target ran at different
                # spatial resolutions this step -- should not happen given
                # matched architectures and lockstep t, but fail soft rather
                # than crash a long-running translation.
                return out
            captured = captured.to(device=out.device, dtype=out.dtype)
            if self.injection_strength >= 1.0:
                return captured
            return (1.0 - self.injection_strength) * out + self.injection_strength * captured
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
