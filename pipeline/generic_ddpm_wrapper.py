"""
GenericDDPMWrapper: a full (encode + decode) counterpart to
model.gan_wrapper.ddpm_ddim_wrapper.DDPMDDIMWrapper for checkpoints whose
architecture doesn't match any i_DDPM() preset in
model/lib/ddpm_ddim/models/improved_ddpm/script_util.py.

Supersedes diagnostics/generic_ddpm_target.py (decode-only): both the FFHQ
side (`~/improved-diffusion/logs/ffhq512/model*.pt`) and the anime side
(`~/improved-diffusion/logs/anime-aligned-curated/model*.pt`) turned out to
use the SAME non-preset architecture (num_channels=256,
attention_resolutions="32,16,8", learn_sigma=True -- confirmed by directly
inspecting both checkpoints' tensor shapes: 486 tensors,
time_embed.0.weight=(1024,256), out.2.weight=(6,256,3,3), identical file
size 2,004,877,596 bytes), so a source-side wrapper needs the same
architecture-construction workaround as the target side, plus `.encode()`.

This class is an ADAPTED COPY of DDPMDDIMWrapper's __init__ (only the
model-construction part, i.e. i_DDPM(...) -> build_unet_from_args(...);
everything else copied verbatim), `.encode()`, `.generate()`, and
`.forward()` (copied verbatim -- none of these three methods reference
i_DDPM or any dataset-name dispatch, only self.generator/self.betas/
self.logvar/self.resolution/self.channels/self.sample_type/self.eta/
self.custom_steps/self.es_steps/self.refine_steps/self.t_0, which this
class sets up the same way model/gan_wrapper/ddpm_ddim_wrapper.py does).

See diagnostics/generic_ddpm_target.py's docstring for the architecture
cross-check against the user-provided DDPM_ARCHITECTURE.md (attention QKV
layout, rescale_timesteps no-op, normalization/output conversion, LEARNED_
RANGE channel split) -- all of that reasoning applies identically here.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.gan_wrapper.ddpm_ddim_wrapper import (  # noqa: E402
    denoising_step_with_eps, compute_eps, sample_xt, sample_xt_next,
)
from model.lib.ddpm_ddim.utils.diffusion_utils import denoising_step, get_beta_schedule, extract  # noqa: E402
from pipeline import model_loader  # noqa: E402


class GenericDDPMWrapper(torch.nn.Module):
    def __init__(self, model_path, image_size, channels, arch_kwargs,
                 beta_schedule, beta_start, beta_end, num_diffusion_timesteps,
                 sample_type, custom_steps, es_steps, eta=None, t_0=None,
                 refine_steps=0, refine_iterations=1, enforce_class_input=None,
                 strict_load=True, free_tail_steps=1,
                 ilvr_enabled=False, ilvr_downsample_factor=8, allow_eta_zero=False):
        super().__init__()
        # ILVR (Choi et al., "ILVR: Conditioning Method for Denoising Diffusion
        # Probabilistic Models") low-frequency structure guidance -- OFF by
        # default, matching original generate() behavior bit-for-bit when
        # ilvr_enabled=False (the correction step is skipped entirely, see
        # _ilvr_correct below). See diagnostics/outputs/ for the design doc
        # this implements. `ilvr_reference` (the source photo, [-1,1], same
        # resolution) is supplied per-call to generate()/forward(), not here,
        # since it's per-image data rather than wrapper configuration.
        self.ilvr_enabled = ilvr_enabled
        self.ilvr_downsample_factor = ilvr_downsample_factor
        self.enforce_class_input = enforce_class_input
        self.custom_steps = custom_steps
        self.refine_steps = refine_steps
        self.refine_iterations = refine_iterations
        self.sample_type = sample_type
        self.eta = eta
        self.t_0 = t_0 if t_0 is not None else 999
        self.es_steps = es_steps
        # free_tail_steps: number of FINAL decode iterations (closest to t=0,
        # where compute_eps's c1 -> 0 causes the transcribed eps to blow up
        # numerically -- see diagnostics/outputs from the eps-norm comparison
        # experiment this session) that use the target's own free prediction
        # (denoising_step, no injected eps) instead of denoising_step_with_eps.
        # DDPMDDIMWrapper.generate() effectively hard-codes this to 1 (only
        # the very last iteration, it == es_steps-1, skips eps injection).
        # Raising it trades some fidelity to the source's exact trajectory for
        # avoiding the numerically unstable tail, without shortening es_steps
        # itself (which was found to hurt otherwise-good images).
        assert 1 <= free_tail_steps <= es_steps
        self.free_tail_steps = free_tail_steps
        if ilvr_enabled and free_tail_steps < 20:
            # ILVR is only applied on "free" (non-eps-injected) steps -- see
            # _ilvr_correct's call site in generate() -- so with the default
            # free_tail_steps=1 it would only ever run once, making it close to
            # a no-op. Not a hard error (the caller may genuinely want a very
            # light touch), just a heads-up.
            print(f"[GenericDDPMWrapper] NOTE: ilvr_enabled=True with free_tail_steps={free_tail_steps} "
                  f"-- ILVR only applies during free (non-eps-injected) steps, so its effect will be "
                  f"minimal unless free_tail_steps is also raised.")

        if self.sample_type == "ddim":
            # Original CycleDiffusion encode()/generate() genuinely requires eta > 0
            # (compute_eps/sample_xt_next divide by an eta-derived c1). Task 4's
            # deterministic_invert()/decode_one_step() never go through those
            # functions with a nonzero eta requirement -- decode_one_step's target
            # branch legitimately wants eta=0 to be selectable (fully deterministic
            # target decode, e.g. to isolate whether Task 3's --fbs artifacts are
            # caused by interaction with target-side stochastic noise injection) --
            # so allow_eta_zero (set by the caller only when deterministic_inversion
            # is requested) relaxes this to >=0 without changing the default path.
            if allow_eta_zero:
                assert self.eta is not None and self.eta >= 0
            else:
                assert self.eta is not None and self.eta > 0
        elif self.sample_type == "ddpm":
            assert self.eta is None
        else:
            raise ValueError(self.sample_type)

        betas = get_beta_schedule(
            beta_start=beta_start, beta_end=beta_end,
            num_diffusion_timesteps=num_diffusion_timesteps, beta_schedule=beta_schedule,
        )
        self.register_buffer("betas", torch.from_numpy(betas).float())
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        self.register_buffer("alphas_cumprod", torch.from_numpy(alphas_cumprod).float())
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.logvar = np.log(np.maximum(posterior_variance, 1e-20))
        # Matches DDPMDDIMWrapper.__init__'s hard-coded choice for every dataset
        # (see diagnostics/outputs/step_c2/step_b2 from an earlier session): the
        # model's learned variance channels are still split off correctly
        # regardless of this flag; only the (DDIM-branch-unused)
        # posterior-variance term is affected.
        self.learn_sigma = False

        class _Args:
            pass
        a = _Args()
        for k, v in arch_kwargs.items():
            setattr(a, k, v)
        self.generator = model_loader.build_unet_from_args(a, image_size=image_size)
        state_dict = torch.load(model_path, map_location="cpu")
        missing, unexpected = self.generator.load_state_dict(state_dict, strict=strict_load)
        if missing or unexpected:
            print(f"[GenericDDPMWrapper] WARNING load mismatch -- missing={missing} unexpected={unexpected}")

        self.resolution = image_size
        self.channels = channels
        self.latent_dim = self.resolution ** 2 * self.channels * self.es_steps
        for p in self.generator.parameters():
            p.requires_grad_(False)

        self.post_process = transforms.Compose(
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
        )

    # --- Copied verbatim (algorithm-wise) from DDPMDDIMWrapper.encode() ---
    def encode(self, image, class_label=None):
        self.generator.eval()

        if (self.t_0 + 1) % self.custom_steps == 0:
            seq_inv = range(0, self.t_0 + 1, (self.t_0 + 1) // self.custom_steps)
            assert len(seq_inv) == self.custom_steps
        else:
            seq_inv = np.linspace(0, 1, self.custom_steps) * self.t_0
        seq_inv = [int(s) for s in list(seq_inv)][:self.es_steps]
        seq_inv_next = ([-1] + list(seq_inv[:-1]))[:self.es_steps]

        image = (image - 0.5) * 2.0
        assert image.shape[2] == image.shape[3] == self.resolution

        with torch.no_grad():
            x0 = image
            bsz = x0.shape[0]

            if self.enforce_class_input:
                assert class_label is not None
                raise NotImplementedError()
            else:
                T = (torch.ones(bsz) * (self.es_steps - 1)).to(self.device)
                xT = sample_xt(x0=x0, t=T, b=self.betas)
                z_list = [xT, ]

                xt = xT
                for it, (i, j) in enumerate(zip(reversed(seq_inv), reversed(seq_inv_next))):
                    t = (torch.ones(bsz) * i).to(self.device)
                    t_next = (torch.ones(bsz) * j).to(self.device)

                    if it < self.es_steps - 1:
                        xt_next = sample_xt_next(
                            x0=x0, xt=xt, t=t, t_next=t_next,
                            sampling_type=self.sample_type, b=self.betas, eta=self.eta,
                        )
                        eps = compute_eps(
                            xt=xt, xt_next=xt_next, t=t, t_next=t_next, models=self.generator,
                            sampling_type=self.sample_type, b=self.betas, logvars=self.logvar,
                            eta=self.eta, learn_sigma=self.learn_sigma,
                        )
                        xt = xt_next
                        z_list.append(eps)
                    else:
                        break

            z = torch.stack(z_list, dim=1).view(bsz, -1)
            assert z.shape[1] == self.latent_dim

        return z

    # --- Task 4: deterministic DDIM inversion (Song et al., eta=0) -- an
    # alternative to encode()'s CycleDiffusion transcribed-eps latent. Unlike
    # encode(), which samples xt_next via the closed-form forward process
    # q(x_t|x0) (using fresh randomness) and then solves for whatever eps
    # denoising_step_with_eps would need to reproduce that exact sample,
    # deterministic_invert walks FORWARD through noise levels using the
    # model's OWN eps prediction at each step -- no randomness anywhere.
    # denoising_step_with_eps's ddim/eta=0 branch (`xt_next = sqrt(at_next)*
    # x0_t + sqrt(1-at_next)*et`) is direction-agnostic: it only depends on
    # which of t/t_next is "current" (evaluated by the model) vs "target", not
    # on whether at_next is larger or smaller than at, so it can be reused
    # unmodified for the forward (noise-increasing) direction by simply
    # passing t=current, t_next=further-along-the-forward-sequence. This is
    # required by FBSDiff (Task 3): its "reconstruction trajectory" only
    # makes sense as a genuine deterministic inverse of the source photo,
    # which CycleDiffusion's stochastic-transcription encode() is not.
    def deterministic_invert(self, image, inversion_steps=1000):
        self.generator.eval()
        seq = [int(s) for s in np.linspace(0, self.t_0, inversion_steps)]
        image = (image - 0.5) * 2.0
        assert image.shape[2] == image.shape[3] == self.resolution
        x = image
        bsz = x.shape[0]
        with torch.no_grad():
            for idx in range(len(seq) - 1):
                t_cur = (torch.ones(bsz) * seq[idx]).to(self.device)
                t_fwd = (torch.ones(bsz) * seq[idx + 1]).to(self.device)
                x = denoising_step_with_eps(
                    x, eps=torch.zeros_like(x), t=t_cur, t_next=t_fwd, models=self.generator,
                    logvars=self.logvar, b=self.betas, sampling_type="ddim", eta=0.0,
                    learn_sigma=self.learn_sigma,
                )
        return x

    def decode_one_step(self, x, t, t_next, sampling_type=None, eta=None):
        """One free decode (reverse, noise-decreasing) step using this
        wrapper's own model and its own randomness (denoising_step, not the
        eps-injection variant) -- the primitive Task 3's dual source/target
        trajectory decode needs, exposed so the translation script can step
        both wrappers in lockstep without reaching into model internals.
        Defaults to this wrapper's own configured sample_type/eta (so target
        decodes exactly as it normally would); pass sampling_type='ddim',
        eta=0.0 explicitly to get the deterministic reconstruction trajectory
        (this exactly reverses deterministic_invert's forward path -- both
        use denoising_step's/denoising_step_with_eps's ddim eta=0 branch,
        which computes the same x0_t formula and is direction-agnostic)."""
        sampling_type = sampling_type if sampling_type is not None else self.sample_type
        eta = eta if eta is not None else (self.eta if self.eta is not None else 0.0)
        return denoising_step(
            x, t=t, t_next=t_next, models=self.generator, logvars=self.logvar,
            sampling_type=sampling_type, b=self.betas, eta=eta, learn_sigma=self.learn_sigma,
        )

    def _ilvr_low_pass(self, x):
        """φ_N(x): downsample by ilvr_downsample_factor then upsample back,
        i.e. a simple low-pass filter. factor<=1 is a no-op.
        Uses 'area' (proper anti-aliased box averaging, no ringing) for the
        downsample and 'bilinear' for the upsample -- NOT bicubic, which
        overshoots/rings at edges; applied every free step for up to
        free_tail_steps iterations, that ringing compounds (each step's
        ringing becomes part of the next step's model input) and was the
        likely cause of the residual scribble-noise corruption seen with
        bicubic in diagnostics/outputs/step_e21_ilvr_fixed/."""
        factor = self.ilvr_downsample_factor
        if factor <= 1:
            return x
        h, w = x.shape[-2], x.shape[-1]
        down = F.interpolate(x, size=(h // factor, w // factor), mode="area")
        up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
        return up

    def _ilvr_correct(self, x, t_next, ilvr_reference):
        """ILVR (Choi et al.) low-frequency structure guidance: replace x's
        low-frequency component with that of the reference image noised to
        the same level (t_next). No-op unless ilvr_enabled and a reference
        was supplied -- callers always go through this method rather than
        branching inline, so ilvr_enabled=False is guaranteed to leave the
        original (pre-ILVR) generate() behavior completely unchanged."""
        if not self.ilvr_enabled or ilvr_reference is None:
            return x
        bsz = x.shape[0]
        if t_next.sum() == -t_next.shape[0]:  # t_next == -1 (final step): treat as the clean reference itself
            at_next = torch.ones((bsz,) + (1,) * (len(x.shape) - 1), device=x.device, dtype=x.dtype)
        else:
            at_next = extract(self.alphas_cumprod, t_next.long(), x.shape)
        noise = torch.randn_like(ilvr_reference)
        y_next = at_next.sqrt() * ilvr_reference + (1 - at_next).sqrt() * noise
        return x - self._ilvr_low_pass(x) + self._ilvr_low_pass(y_next)

    # --- Copied verbatim (algorithm-wise) from DDPMDDIMWrapper.generate(),
    # plus the optional ILVR correction (see _ilvr_correct above; a no-op
    # when ilvr_enabled=False, which is the default). ---
    def generate(self, z, class_label=None, ilvr_reference=None):
        if self.enforce_class_input:
            assert class_label is not None
            raise NotImplementedError()
        else:
            assert class_label is None
        if self.ilvr_enabled:
            assert ilvr_reference is not None, "ilvr_enabled=True requires generate()/forward() to receive ilvr_reference"

        if (self.t_0 + 1) % self.custom_steps == 0:
            seq_inv = range(0, self.t_0 + 1, (self.t_0 + 1) // self.custom_steps)
            assert len(seq_inv) == self.custom_steps
        else:
            seq_inv = np.linspace(0, 1, self.custom_steps) * self.t_0
        seq_inv = [int(s) for s in list(seq_inv)][:self.es_steps]
        seq_inv_next = ([-1] + list(seq_inv[:-1]))[:self.es_steps]

        bsz = z.shape[0]
        eps_list = z.view(bsz, self.es_steps, self.channels, self.resolution, self.resolution)
        x_T = eps_list[:, 0]
        eps_list = eps_list[:, 1:]

        x = x_T
        for it, (i, j) in enumerate(zip(reversed(seq_inv), reversed(seq_inv_next))):
            t = (torch.ones(bsz) * i).to(self.device)
            t_next = (torch.ones(bsz) * j).to(self.device)
            if it < self.es_steps - self.free_tail_steps:
                # Eps-injected (transcribed-source) step: compute_eps computed this
                # eps assuming an ILVR-free trajectory, so applying the ILVR
                # low-frequency swap here would fight the injected eps -- this was
                # exactly the cause of the noise-collapse failure observed when
                # ILVR was (incorrectly) applied unconditionally at every step. Do
                # NOT call _ilvr_correct in this branch.
                eps = eps_list[:, it]
                x = denoising_step_with_eps(
                    x, eps=eps, t=t, t_next=t_next, models=self.generator, logvars=self.logvar,
                    sampling_type=self.sample_type, b=self.betas, eta=self.eta, learn_sigma=self.learn_sigma,
                )
            else:
                # Free (target's own prediction, no injected eps) step: safe to
                # apply ILVR here, since there's no transcribed-trajectory
                # assumption to conflict with.
                x = denoising_step(
                    x, t=t, t_next=t_next, models=self.generator, logvars=self.logvar,
                    sampling_type=self.sample_type, b=self.betas, eta=self.eta, learn_sigma=self.learn_sigma,
                )
                x = self._ilvr_correct(x, t_next, ilvr_reference)

        if self.refine_steps == 0:
            img = x
        else:
            for _ in range(self.refine_iterations):
                refine_eta = 1
                t = (torch.ones(bsz) * self.refine_steps - 1).to(self.device)
                xt = sample_xt(x0=x, t=t, b=self.betas)
                x = xt
                assert self.refine_steps < self.custom_steps
                seq_inv_refine = seq_inv[:self.refine_steps]
                seq_inv_next_refine = seq_inv_next[:self.refine_steps]
                for i, j in zip(reversed(seq_inv_refine), reversed(seq_inv_next_refine)):
                    t = (torch.ones(bsz) * i).to(self.device)
                    t_next = (torch.ones(bsz) * j).to(self.device)
                    x = denoising_step(
                        x, t=t, t_next=t_next, models=self.generator, logvars=self.logvar,
                        sampling_type=self.sample_type, b=self.betas, eta=refine_eta, learn_sigma=self.learn_sigma,
                    )
                    x = self._ilvr_correct(x, t_next, ilvr_reference)
            img = x
        return img

    def forward(self, z, class_label=None, ilvr_reference=None):
        self.generator.eval()
        img = self.generate(z, class_label, ilvr_reference=ilvr_reference)
        img = self.post_process(img)
        return img

    @property
    def device(self):
        return next(self.parameters()).device
