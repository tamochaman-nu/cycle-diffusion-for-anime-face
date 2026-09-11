"""
Step E-12: CycleDiffusion FFHQ -> anime translation with BOTH source and
target on the matching new architecture (num_channels=256,
attention_resolutions="32,16,8", learn_sigma=True).

Supersedes step_e4_translate_curated_target.py for this pairing:
step_e4 used ckpts/ddpm/ffhq400000.pt (the OLD FFHQ checkpoint, CUSTOM_FACE256
architecture: num_channels=128) as the source while anime-aligned-curated
uses the new 256-channel architecture -- an unintentional architecture
mismatch between source and target caught mid-run by the user. The correctly
paired source is `~/improved-diffusion/logs/ffhq512/model*.pt`, confirmed by
directly inspecting its checkpoint tensors: 486 tensors, dtype float32,
time_embed.0.weight=(1024,256), out.2.weight=(6,256,3,3) -- byte-identical
file size (2,004,877,596) and shape signature to the anime-aligned-curated
checkpoints. ("ffhq512" names the source dataset -- ffhq512x512 -- not the
model's operating resolution; the checkpoint is architecturally identical to
the 256px anime-aligned-curated one.)

Both sides now use diagnostics/generic_ddpm_wrapper.py's GenericDDPMWrapper
(new; DDPMDDIMWrapper's i_DDPM()-based construction has no preset for this
architecture) instead of model.gan_wrapper.ddpm_ddim_wrapper.DDPMDDIMWrapper.
GenericDDPMWrapper's .encode()/.generate()/.forward() are adapted copies of
DDPMDDIMWrapper's (see that file's docstring), so this script's translation
logic is otherwise identical to step_e4/step_d1.

Usage:
  python diagnostics/step_e12_translate_matched_arch.py \\
      --source_model_path ckpts/ddpm/ffhq512_080000.pt \\
      --target_model_path ckpts/ddpm/anime_aligned_curated_130000.pt \\
      --num_images 4
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline.generic_ddpm_wrapper import GenericDDPMWrapper  # noqa: E402
from pipeline.common import (  # noqa: E402
    set_all_seeds, get_device, ensure_dir, load_image_as_tensor, save_image_grid,
    save_side_by_side, mean_saturation, laplacian_variance, write_csv, psnr, ssim,
)
from pipeline import fbsdiff  # noqa: E402
from pipeline import feature_injection  # noqa: E402
from pipeline import face_parsing  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _str2bool(x):
    return str(x).lower() in ("1", "true", "yes", "y")


def add_arch_args(parser, prefix):
    p = f"--{prefix}_"
    parser.add_argument(f"{p}image_size", type=int, default=256)
    parser.add_argument(f"{p}channels", type=int, default=3)
    parser.add_argument(f"{p}arch_num_channels", type=int, default=256)
    parser.add_argument(f"{p}arch_num_res_blocks", type=int, default=2)
    parser.add_argument(f"{p}arch_attention_resolutions", type=str, default="32,16,8")
    parser.add_argument(f"{p}arch_num_heads", type=int, default=4)
    parser.add_argument(f"{p}arch_num_head_channels", type=int, default=-1)
    parser.add_argument(f"{p}arch_channel_mult", type=str, default="")
    parser.add_argument(f"{p}arch_dropout", type=float, default=0.0)
    parser.add_argument(f"{p}arch_resblock_updown", type=_str2bool, default=False)
    parser.add_argument(f"{p}arch_use_scale_shift_norm", type=_str2bool, default=True)
    parser.add_argument(f"{p}arch_class_cond", type=_str2bool, default=False)
    parser.add_argument(f"{p}arch_learn_sigma", type=_str2bool, default=True)
    parser.add_argument(f"{p}beta_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument(f"{p}beta_start", type=float, default=0.0001)
    parser.add_argument(f"{p}beta_end", type=float, default=0.02)
    parser.add_argument(f"{p}num_diffusion_timesteps", type=int, default=1000)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_model_path", type=str, required=True)
    parser.add_argument("--target_model_path", type=str, required=True)
    parser.add_argument("--source_sample_type", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--target_sample_type", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--eta", type=float, default=0.1)
    add_arch_args(parser, "source")
    add_arch_args(parser, "target")
    parser.add_argument("--custom_steps", type=int, default=1000)
    parser.add_argument("--es_steps", type=int, default=850)
    parser.add_argument("--refine_steps", type=int, default=0)
    parser.add_argument("--target_free_tail_steps", type=int, default=1,
                         help="Number of final target-side decode iterations that use the target's own "
                              "free prediction instead of injected source eps (default 1, matching "
                              "DDPMDDIMWrapper's production behavior). Raise this to skip the "
                              "numerically unstable near-t=0 tail (see diagnostics/generic_ddpm_wrapper.py "
                              "GenericDDPMWrapper's free_tail_steps docstring) without shortening es_steps.")
    parser.add_argument("--ilvr_enabled", type=_str2bool, default=False,
                         help="ON/OFF switch for ILVR (Choi et al.) low-frequency structure guidance: "
                              "at each target-side decode step, replace the low-frequency component of "
                              "the current sample with that of the FFHQ source photo noised to the same "
                              "level, leaving high-frequency (style/texture) detail free for the target "
                              "model. Default OFF -- when False, GenericDDPMWrapper.generate() is "
                              "bit-for-bit identical to before this feature was added.")
    parser.add_argument("--ilvr_downsample_factor", type=int, default=8,
                         help="Only used when --ilvr_enabled. Controls how aggressive the low-pass filter "
                              "is (downsample-then-upsample by this factor). Larger = weaker structural "
                              "constraint (more style freedom); smaller = stronger constraint (closer to "
                              "the source photo, less stylization). 1 disables filtering (full copy).")
    # --- Task 4: deterministic DDIM inversion (see generic_ddpm_wrapper.py's
    # deterministic_invert docstring). Default OFF -- when False this whole
    # code path is untouched and source.encode() behaves exactly as before.
    parser.add_argument("--deterministic_inversion", type=_str2bool, default=False,
                         help="Task 4. Default OFF. When True, replaces source.encode()'s CycleDiffusion "
                              "transcribed-eps latent with a genuine deterministic DDIM inversion (eta=0, "
                              "no injected eps at any step) -- a prerequisite for --fbs (Task 3), which needs "
                              "a real source-model reconstruction trajectory to substitute frequency bands "
                              "from. Also decodes the target with a plain DDIM loop (eta=0 unless --fbs is "
                              "also set with an --eta override) instead of CycleDiffusion's eps-injection "
                              "scheme, since there is no per-step transcribed eps in this mode.")
    parser.add_argument("--inversion_steps", type=int, default=1000,
                         help="Number of discretized forward steps used by --deterministic_inversion "
                              "(0 -> t_0). Independent of --custom_steps/--es_steps, which only affect the "
                              "original CycleDiffusion encode()/generate() path.")
    parser.add_argument("--reconstruction_warn_ssim", type=float, default=0.85,
                         help="--deterministic_inversion self-check: after inverting, the source model "
                              "immediately re-decodes (eta=0) back to an image and SSIM is compared against "
                              "the original photo. Below this threshold a warning is printed/logged -- poor "
                              "reconstruction here means Task 3's frequency-band substitution is working from "
                              "an already-unreliable reconstruction trajectory.")
    # --- Task 3: FBSDiff frequency-band replacement (see diagnostics/fbsdiff.py).
    # Requires --deterministic_inversion (validated at startup, not just documented
    # here -- see main()). Default "off" -- zero effect on the existing pipeline.
    parser.add_argument("--fbs", type=str, default="off", choices=["off", "low", "mid", "high"],
                         help="Task 3. Default 'off' (no effect). Which 2D-DCT frequency band of the target's "
                              "decode trajectory to substitute with the source reconstruction trajectory's "
                              "same band, during the calibration phase only (see --fbs_lambda). Requires "
                              "--deterministic_inversion true.")
    parser.add_argument("--fbs_threshold", type=float, default=None,
                         help="DCT coordinate-sum (u+v) cutoff defining the band for --fbs -- small values "
                              "(paper's own raw magnitudes, see diagnostics/fbsdiff.py DEFAULT_THRESHOLDS) "
                              "confirmed to work at this project's 256x256 pixel-space resolution; a naive "
                              "resolution-proportional rescaling (~4x larger) was tried and empirically "
                              "produces a persistent hatched artifact on detailed/patterned photos -- see "
                              "the 'Threshold scale note' in diagnostics/fbsdiff.py's module docstring. "
                              "Default: None, meaning use the mode-specific default.")
    parser.add_argument("--fbs_lambda", type=float, default=0.45,
                         help="Fraction of decode steps left FREE (no substitution) at the end of decoding, "
                              "matching the paper's lambda; substitution only happens during the first "
                              "(1 - fbs_lambda) fraction of steps (the 'calibration phase'), while noise is "
                              "still high. Default 0.45, matching the FBSDiff paper.")
    # --- A-1: Plug-and-Play decoder ResBlock feature injection (see
    # diagnostics/feature_injection.py). Requires --deterministic_inversion
    # (validated at startup -- see validate_args). Default OFF -- zero effect
    # on the existing pipeline (including on --fbs, which it can be combined
    # with, since they act on different representations -- see
    # translate_one_deterministic).
    parser.add_argument("--pnp_enabled", type=_str2bool, default=False,
                         help="A-1 (Tumanyan et al. CVPR2023 Plug-and-Play Diffusion Features, adapted). "
                              "Default OFF. When True, injects the source model's decoder ResBlock activations "
                              "(at --pnp_feature_layers) directly into the target model's same layer during "
                              "the calibration phase (see --pnp_lambda), preserving spatial/structural layout "
                              "at a much richer representation than --fbs's raw DCT bands or ILVR's raw pixels. "
                              "Requires --deterministic_inversion true.")
    parser.add_argument("--pnp_feature_layers", type=int, nargs="+", default=[0],
                         help="Indices into the UNet's output_blocks (decoder) ModuleList to inject at -- index "
                              "0 is the coarsest/deepest decoder layer (same resolution as the bottleneck), "
                              "increasing indices move toward the final full-resolution output (this "
                              "architecture has 18 output_blocks total; 0-8 have self-attention, 9-17 do not -- "
                              "see diagnostics/feature_injection.py). Default [0]: a parameter sweep this "
                              "session (diagnostics/outputs/step_h2_sweep) found layer 4 (an earlier default) "
                              "erases facial-feature formation entirely (eyes/nose/mouth never form -- "
                              "diagnostics/outputs/step_h1_pnp_rcc_1000), while layer 0 consistently produced "
                              "clean anime line art WITH well-formed eyes/nose/mouth across every --pnp_lambda "
                              "tested (0.7-0.95); layers 1-2 fell back towards a photo-realistic hybrid look "
                              "(diagnostics/outputs/step_h2_sweep/L1_lam0.8, L2_lam0.8). Validated at full "
                              "inversion_steps=1000 on 3 images (diagnostics/outputs/step_h3_pnp_L0_lam09_1000): "
                              "2/3 clean, 1/3 (the pose-difficult 50494.png already flagged this session) still "
                              "struggles.")
    parser.add_argument("--pnp_lambda", type=float, default=0.9,
                         help="Fraction of decode steps left FREE (no injection) at the end of decoding, same "
                              "convention as --fbs_lambda. Default 0.9 (only the first 10% of steps injected) -- "
                              "the Plug-and-Play paper's own ~80%%-injected guideline (originally used as this "
                              "default) was found this session to erase facial-feature formation entirely on "
                              "this project's much longer 1000-step pixel-space schedule (see "
                              "--pnp_feature_layers's help and diagnostics/outputs/step_h1_pnp_rcc_1000); at "
                              "--pnp_feature_layers 0, values from 0.7 to 0.95 all gave similarly good results "
                              "(diagnostics/outputs/step_h2_sweep) -- 0.9 was picked as a middle-of-range "
                              "default, not because it was uniquely best.")
    parser.add_argument("--pnp_strength", type=float, default=1.0,
                         help="Blend strength for the injected feature (1.0 = full replacement, matching the "
                              "original Plug-and-Play paper's design; 0.0 = no effect). Added after a parameter "
                              "search (diagnostics/outputs/step_h2..h5) found NO (layer, lambda) combination at "
                              "full strength that preserved both contour AND facial-feature formation at once -- "
                              "every calibration window long enough to hold contour also fully suppressed facial "
                              "detail. A longer hold at partial strength is the untried alternative to a short "
                              "hold at full strength; default 1.0 keeps prior behavior unchanged.")
    # --- A-2: region-aware color correction (see diagnostics/face_parsing.py
    # and region_color_correct's docstring for why this is a POST-PROCESS,
    # not an in-loop guidance mechanism). Default OFF. Independent of
    # --deterministic_inversion -- works with the original CycleDiffusion
    # path too, since it only needs the finished image and the source photo.
    parser.add_argument("--region_color_correct", type=_str2bool, default=False,
                         help="A-2. Default OFF. When True, segments the SOURCE photo into background/skin/"
                              "hair/hat/clothing regions (BiSeNet face parsing, diagnostics/face_parsing.py) "
                              "and nudges each region's average color in the translated output towards that "
                              "region's average color in the source, by --region_color_strength. Targets the "
                              "reported failure mode of background/clothing color bleeding into hair color. "
                              "Works with or without --deterministic_inversion.")
    parser.add_argument("--region_color_strength", type=float, default=0.5,
                         help="0 = no correction, 1 = fully match each region's source mean color. Only used "
                              "when --region_color_correct is true.")
    # --- Diagnostic: remove the source photo's background before translation
    # (BiSeNet face parsing, diagnostics/face_parsing.py). Default OFF, zero
    # effect on the default pipeline. Investigates whether background content
    # contributes to the contour-misidentification failure mode found while
    # tuning --pnp_enabled (background occupies a large fraction of the
    # spatial extent especially at coarse decoder layers, and could dilute or
    # confuse what gets captured/injected as "structure").
    parser.add_argument("--remove_background", type=_str2bool, default=False,
                         help="Default OFF. When True, segments the source photo's background (BiSeNet face "
                              "parsing) and replaces it with flat mid-gray BEFORE encoding/inversion -- applied "
                              "to both translate_one and translate_one_deterministic's input, so it affects the "
                              "encode()/deterministic_invert() latent itself, not just a post-process. Isolates "
                              "whether background content is contributing to incorrect contour transfer, "
                              "independent of --pnp_enabled/--fbs/--ilvr_enabled (works with any of them).")
    parser.add_argument("--t_0", type=int, default=None)
    parser.add_argument("--image_dir", type=str, default=os.path.join(_REPO_ROOT, "data/ffhq"))
    parser.add_argument("--image_paths", type=str, nargs="+", default=None)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default=os.path.join(_REPO_ROOT, "diagnostics/outputs/step_e12"))
    return parser.parse_args()


def validate_args(args):
    """4th fundamental principle of the diagnosis plan: options that are
    logically contradictory or that mutually require each other must be
    validated/rejected at startup, not discovered mid-run."""
    if args.fbs != "off" and not args.deterministic_inversion:
        raise SystemExit(
            f"--fbs={args.fbs} requires --deterministic_inversion true (FBSDiff substitutes frequency bands "
            f"between the source model's own reconstruction trajectory and the target's sampling trajectory -- "
            f"both must start from the SAME deterministically-inverted latent; CycleDiffusion's transcribed-eps "
            f"encode() has no such shared, model-agnostic latent to offer)."
        )
    if args.fbs != "off":
        print("[step_e12] WARNING: --fbs is enabled without identity-guidance (Task 6, not yet implemented in "
              "this codebase). The FBSDiff paper's plan calls for face-recognition-based identity guidance "
              "alongside frequency-band substitution; proceeding without it, per the plan's own 'warn if "
              "missing' (not hard-block) requirement.")
    if not (0.0 <= args.fbs_lambda <= 1.0):
        raise SystemExit(f"--fbs_lambda must be in [0, 1], got {args.fbs_lambda}")
    if args.pnp_enabled and not args.deterministic_inversion:
        raise SystemExit(
            "--pnp_enabled requires --deterministic_inversion true (feature injection needs the source and "
            "target to decode in lockstep from the SAME shared deterministically-inverted latent, exactly like "
            "--fbs -- CycleDiffusion's transcribed-eps encode() has no such shared latent)."
        )
    if not (0.0 <= args.pnp_lambda <= 1.0):
        raise SystemExit(f"--pnp_lambda must be in [0, 1], got {args.pnp_lambda}")
    if not (0.0 <= args.pnp_strength <= 1.0):
        raise SystemExit(f"--pnp_strength must be in [0, 1], got {args.pnp_strength}")
    if not (0.0 <= args.region_color_strength <= 1.0):
        raise SystemExit(f"--region_color_strength must be in [0, 1], got {args.region_color_strength}")


def resolve_output_dir(args):
    """3rd fundamental principle: record every run's active option set in
    the output directory name (in addition to the metadata JSON written by
    write_run_metadata). Only the Task 3/4/A-1/A-2 options are tagged here
    since they're the newest/most likely to be run in varying combinations
    this session; older options (--ilvr_enabled etc.) already have their own
    dedicated --output_dir conventions established in earlier runs."""
    tags = []
    if args.deterministic_inversion:
        tags.append(f"definv{args.inversion_steps}")
    if args.fbs != "off":
        thr = args.fbs_threshold if args.fbs_threshold is not None else fbsdiff.DEFAULT_THRESHOLDS[args.fbs]
        tags.append(f"fbs-{args.fbs}-thr{thr:g}-lam{args.fbs_lambda:g}")
    if args.pnp_enabled:
        layers = "-".join(str(i) for i in args.pnp_feature_layers)
        tags.append(f"pnp-L{layers}-lam{args.pnp_lambda:g}-s{args.pnp_strength:g}")
    if args.region_color_correct:
        tags.append(f"rcc{args.region_color_strength:g}")
    if args.remove_background:
        tags.append("nobg")
    if not tags:
        return args.output_dir
    return args.output_dir.rstrip("/") + "_" + "_".join(tags)


def write_run_metadata(args, output_dir):
    """Principle 3 of the diagnosis plan: record every active CLI option to
    a metadata JSON alongside the run's outputs, so any run can be
    reproduced later from its output directory alone."""
    import json
    ensure_dir(output_dir)
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)


def pick_random_images(image_dir, num_images, seed):
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTS))
    rng = random.Random(seed)
    return [os.path.join(image_dir, f) for f in rng.sample(files, min(num_images, len(files)))]


def eta_for(sample_type, eta):
    return eta if sample_type == "ddim" else None


def build_wrapper(args, prefix, model_path, sample_type, eta, free_tail_steps=1,
                   ilvr_enabled=False, ilvr_downsample_factor=8):
    def g(name):
        return getattr(args, f"{prefix}_{name}")
    arch_kwargs = dict(
        arch_num_channels=g("arch_num_channels"), arch_num_res_blocks=g("arch_num_res_blocks"),
        arch_attention_resolutions=g("arch_attention_resolutions"), arch_num_heads=g("arch_num_heads"),
        arch_num_head_channels=g("arch_num_head_channels"), arch_channel_mult=g("arch_channel_mult"),
        arch_dropout=g("arch_dropout"), arch_resblock_updown=g("arch_resblock_updown"),
        arch_use_scale_shift_norm=g("arch_use_scale_shift_norm"), arch_class_cond=g("arch_class_cond"),
        arch_learn_sigma=g("arch_learn_sigma"),
    )
    return GenericDDPMWrapper(
        model_path=model_path, image_size=g("image_size"), channels=g("channels"),
        arch_kwargs=arch_kwargs, beta_schedule=g("beta_schedule"), beta_start=g("beta_start"),
        beta_end=g("beta_end"), num_diffusion_timesteps=g("num_diffusion_timesteps"),
        sample_type=sample_type, custom_steps=args.custom_steps, es_steps=args.es_steps,
        eta=eta, t_0=args.t_0, refine_steps=args.refine_steps, free_tail_steps=free_tail_steps,
        ilvr_enabled=ilvr_enabled, ilvr_downsample_factor=ilvr_downsample_factor,
        allow_eta_zero=args.deterministic_inversion,
    )


@torch.no_grad()
def translate_one(source, target, img01, device, remove_bg=False):
    low_res = F.interpolate(
        img01.unsqueeze(0).to(device), size=(source.resolution, source.resolution),
        mode="bilinear", align_corners=False,
    )
    if remove_bg:
        low_res = remove_background(low_res, device)
    z = source.encode(low_res)
    # ILVR reference must be in the model's native [-1,1] space (same convention
    # encode() normalizes into via (image-0.5)*2) -- a no-op unless target.ilvr_enabled.
    ilvr_reference = (low_res - 0.5) * 2.0
    styled = target(z=z, ilvr_reference=ilvr_reference)
    return torch.clamp(styled, 0.0, 1.0).squeeze(0).cpu(), low_res.squeeze(0).cpu()


@torch.no_grad()
def translate_one_deterministic(source, target, img01, device, args, log):
    """Tasks 3+4 path: deterministic DDIM inversion (Task 4) of the source
    photo into a single shared latent x_T, optionally followed by FBSDiff
    frequency-band substitution (Task 3) between the source's own
    reconstruction trajectory and the target's sampling trajectory as both
    decode from that shared x_T in lockstep. Only reached when
    args.deterministic_inversion is True -- translate_one (CycleDiffusion's
    transcribed-eps path) is completely untouched and remains the default."""
    assert source.t_0 == target.t_0, (
        f"deterministic inversion requires source and target to share the same t_0 (got "
        f"source.t_0={source.t_0}, target.t_0={target.t_0}) -- both trajectories must walk the same "
        f"noise-level schedule for frequency-band substitution to line up step-for-step."
    )
    low_res = F.interpolate(
        img01.unsqueeze(0).to(device), size=(source.resolution, source.resolution),
        mode="bilinear", align_corners=False,
    )
    if args.remove_background:
        low_res = remove_background(low_res, device)

    x_T = source.deterministic_invert(low_res, inversion_steps=args.inversion_steps)

    seq = [int(s) for s in np.linspace(0, source.t_0, args.inversion_steps)]
    total_steps = len(seq) - 1
    # Decode walks the same schedule backwards: (seq[-1] -> seq[-2]), ..., (seq[1] -> seq[0]), (seq[0] -> -1).
    decode_pairs = list(zip(reversed(seq[1:]), reversed(seq[:-1]))) + [(seq[0], -1)]
    assert len(decode_pairs) == total_steps + 1

    # --- Reconstruction-fidelity self-check (Task 4): source model decoding
    # its own inverted latent (eta=0) should approximately reproduce the input.
    x_recon = x_T
    for t_i, t_next_i in decode_pairs:
        bsz = x_recon.shape[0]
        t = (torch.ones(bsz) * t_i).to(device)
        t_next = (torch.ones(bsz) * t_next_i).to(device)
        x_recon = source.decode_one_step(x_recon, t, t_next, sampling_type="ddim", eta=0.0)
    recon_img = torch.clamp(source.post_process(x_recon), 0.0, 1.0)
    recon_ssim = ssim(low_res.squeeze(0).cpu(), recon_img.squeeze(0).cpu())
    recon_psnr = psnr(low_res.squeeze(0).cpu(), recon_img.squeeze(0).cpu())
    log(f"[translate_one_deterministic] reconstruction fidelity: SSIM={recon_ssim:.4f} PSNR={recon_psnr:.2f} "
        f"(warn threshold SSIM<{args.reconstruction_warn_ssim})")
    if recon_ssim < args.reconstruction_warn_ssim:
        log(f"[translate_one_deterministic] WARNING: reconstruction SSIM {recon_ssim:.4f} is below "
            f"--reconstruction_warn_ssim={args.reconstruction_warn_ssim} -- deterministic_invert's inverted "
            f"latent does not reliably reproduce the source photo via the source model itself, so anything "
            f"built on top of this reconstruction trajectory (Task 3's --fbs) is working from an unreliable "
            f"reference.")

    if args.fbs == "off" and not args.pnp_enabled:
        x_sample = x_T
        for t_i, t_next_i in decode_pairs:
            bsz = x_sample.shape[0]
            t = (torch.ones(bsz) * t_i).to(device)
            t_next = (torch.ones(bsz) * t_next_i).to(device)
            x_sample = target.decode_one_step(x_sample, t, t_next)
        styled = torch.clamp(target.post_process(x_sample), 0.0, 1.0)
        return _finish(styled, low_res, args, device, log)

    # --- Task 3 (FBSDiff) and/or A-1 (Plug-and-Play ResBlock feature injection):
    # both need the source's reconstruction trajectory decoded in lockstep with
    # the target's sampling trajectory from the SAME shared x_T, so they share
    # one dual-trajectory loop. Each has its own independent calibration window
    # (its own lambda) since they act on different representations (FBS: DCT
    # frequency bands of the pixel-space latent; PnP: U-Net decoder ResBlock
    # activations) and there is no reason the same schedule should suit both.
    fbs_threshold = args.fbs_threshold if args.fbs_threshold is not None else fbsdiff.DEFAULT_THRESHOLDS.get(args.fbs)
    fbs_calib_steps = int(round((1.0 - args.fbs_lambda) * len(decode_pairs))) if args.fbs != "off" else 0
    if args.fbs != "off":
        log(f"[translate_one_deterministic] FBS mode={args.fbs} threshold={fbs_threshold:g} "
            f"lambda={args.fbs_lambda} -> substituting during the first {fbs_calib_steps}/{len(decode_pairs)} "
            f"decode steps")

    injector = None
    pnp_calib_steps = 0
    if args.pnp_enabled:
        injector = feature_injection.ResBlockFeatureInjector(source, target, args.pnp_feature_layers)
        injector.injection_strength = args.pnp_strength
        pnp_calib_steps = int(round((1.0 - args.pnp_lambda) * len(decode_pairs)))
        log(f"[translate_one_deterministic] PnP feature injection layers={args.pnp_feature_layers} "
            f"lambda={args.pnp_lambda} strength={args.pnp_strength} -> injecting during the first "
            f"{pnp_calib_steps}/{len(decode_pairs)} decode steps")

    try:
        x_recon2 = x_T.clone()
        x_sample = x_T.clone()
        warned_all_true = False
        for step_idx, (t_i, t_next_i) in enumerate(decode_pairs):
            bsz = x_sample.shape[0]
            t = (torch.ones(bsz) * t_i).to(device)
            t_next = (torch.ones(bsz) * t_next_i).to(device)
            # Source's own forward pass here also feeds the injector's capture
            # hooks (a no-op if --pnp_enabled is False -- no hooks registered).
            x_recon2 = source.decode_one_step(x_recon2, t, t_next, sampling_type="ddim", eta=0.0)
            if injector is not None:
                injector.injection_enabled = step_idx < pnp_calib_steps
            x_sample = target.decode_one_step(x_sample, t, t_next)
            if args.fbs != "off" and step_idx < fbs_calib_steps:
                # Anneal strength linearly from full at step 0 down to ~0 at the end of the
                # calibration window, rather than holding it constant then cutting it off
                # abruptly -- an abrupt cutoff was found to compound into a persistent
                # hatched/textured artifact (see fbsdiff.py's _BLEND_STRENGTH docstring and
                # diagnostics/outputs/step_g8..g12): the model needs room to reconcile the
                # substituted band with its own trajectory before being left to decode freely.
                anneal_strength = fbsdiff._BLEND_STRENGTH * (1.0 - step_idx / fbs_calib_steps)
                merged, mask_all_true = fbsdiff.replace_frequency_band(
                    x_sample.squeeze(0), x_recon2.squeeze(0), mode=args.fbs, threshold=fbs_threshold,
                    strength=anneal_strength,
                )
                x_sample = merged.unsqueeze(0).to(device)
                if mask_all_true and not warned_all_true:
                    log(f"[translate_one_deterministic] WARNING: --fbs_threshold={fbs_threshold:g} for mode="
                        f"'{args.fbs}' selects EVERY DCT coefficient -- this is equivalent to fully overwriting "
                        f"the target's latent with the source reconstruction at every calibration step, not "
                        f"band-selective substitution. Consider lowering (low/mid) or raising (high) the threshold.")
                    warned_all_true = True
    finally:
        if injector is not None:
            injector.remove()

    styled = torch.clamp(target.post_process(x_sample), 0.0, 1.0)
    return _finish(styled, low_res, args, device, log)


def _finish(styled, low_res, args, device, log):
    """Shared tail for both translate_one_deterministic branches: A-2's
    region-aware color correction (see region_color_correct's docstring for
    why this is a single post-process rather than an in-loop guidance
    mechanism) applies here, independent of which decode path produced
    `styled` -- it needs no --deterministic_inversion machinery itself."""
    if args.region_color_correct:
        styled = region_color_correct(styled, low_res, device, args.region_color_strength, log)
    return styled.squeeze(0).cpu(), low_res.squeeze(0).cpu()


def remove_background(low_res, device):
    """Diagnostic: replace the source photo's background (BiSeNet face
    parsing, diagnostics/face_parsing.py) with flat mid-gray, BEFORE
    encoding/inversion -- unlike region_color_correct (a post-process on the
    finished output), this changes what the source model itself sees, so it
    affects encode()/deterministic_invert()'s latent and therefore every
    downstream mechanism (ILVR/FBS/PnP/plain translation) uniformly.
    low_res: (1, 3, H, W) tensor in [0, 1]. Returns a same-shape tensor."""
    mask = face_parsing.region_masks(low_res.squeeze(0), device)["background"]
    out = low_res.clone()
    out.squeeze(0)[:, mask] = 0.5
    return out


def region_color_correct(styled, low_res, device, strength, log):
    """A-2: nudge each semantic region's (background/skin/hair/hat/clothing,
    via diagnostics/face_parsing.py's BiSeNet segmentation of the SOURCE
    photo) average color in `styled` towards that same region's average
    color in the source, by `strength` (0 = no change, 1 = fully match the
    source region's mean color). Applied ONCE on the finished image rather
    than injected into the decode loop -- every guidance mechanism this
    session that intervened repeatedly inside the loop (ILVR, FBSDiff)
    produced compounding artifacts needing significant debugging to resolve;
    a single corrective pass on the final image cannot compound, and this
    directly targets the user's reported problem of background/clothing
    color bleeding into hair color, which is a property of the FINAL
    image's per-region statistics, not of the generative trajectory.
    styled, low_res: (1, 3, H, W) tensors in [0, 1] (same convention as
    translate_one_deterministic's other tensors). Segmentation runs on
    low_res (the source photo) only -- styled's own content is not
    segmented, since its semantic layout is exactly what we are trying to
    keep faithful to the source's, not re-derive independently."""
    masks = face_parsing.region_masks(low_res.squeeze(0), device)
    corrected = styled.clone()
    for name, mask in masks.items():
        pixel_count = int(mask.sum().item())
        if pixel_count == 0:
            continue
        mask_f = mask.to(device=device, dtype=styled.dtype)
        source_mean = (low_res.squeeze(0) * mask_f).sum(dim=(-2, -1)) / pixel_count
        styled_mean = (styled.squeeze(0) * mask_f).sum(dim=(-2, -1)) / pixel_count
        shift = (source_mean - styled_mean) * strength
        corrected.squeeze(0)[:, mask] += shift.unsqueeze(-1)
        log(f"[region_color_correct] {name}: {pixel_count}px, mean RGB shift {shift.tolist()}")
    return torch.clamp(corrected, 0.0, 1.0)


def main():
    args = parse_args()
    validate_args(args)
    args.output_dir = resolve_output_dir(args)
    set_all_seeds(args.seed)
    device = get_device(args.device)
    ensure_dir(args.output_dir)

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    write_run_metadata(args, args.output_dir)

    image_paths = args.image_paths or pick_random_images(args.image_dir, args.num_images, args.seed)
    log(f"[step_e12] source: {args.source_model_path} sample_type={args.source_sample_type}")
    log(f"[step_e12] target: {args.target_model_path} sample_type={args.target_sample_type}")
    log(f"[step_e12] custom_steps={args.custom_steps} es_steps={args.es_steps} eta={args.eta} "
        f"refine_steps={args.refine_steps} target_free_tail_steps={args.target_free_tail_steps} "
        f"ilvr_enabled={args.ilvr_enabled} ilvr_downsample_factor={args.ilvr_downsample_factor} "
        f"deterministic_inversion={args.deterministic_inversion} inversion_steps={args.inversion_steps} "
        f"fbs={args.fbs} fbs_threshold={args.fbs_threshold} fbs_lambda={args.fbs_lambda} "
        f"pnp_enabled={args.pnp_enabled} pnp_feature_layers={args.pnp_feature_layers} pnp_lambda={args.pnp_lambda} "
        f"pnp_strength={args.pnp_strength} "
        f"region_color_correct={args.region_color_correct} region_color_strength={args.region_color_strength} "
        f"seed={args.seed} device={device}")
    log(f"[step_e12] images: {[os.path.basename(p) for p in image_paths]}")

    log("[step_e12] loading source model (GenericDDPMWrapper)...")
    source = build_wrapper(
        args, "source", args.source_model_path, args.source_sample_type,
        eta_for(args.source_sample_type, args.eta),
    ).to(device).eval()

    log("[step_e12] loading target model (GenericDDPMWrapper)...")
    target = build_wrapper(
        args, "target", args.target_model_path, args.target_sample_type,
        eta_for(args.target_sample_type, args.eta), free_tail_steps=args.target_free_tail_steps,
        ilvr_enabled=args.ilvr_enabled, ilvr_downsample_factor=args.ilvr_downsample_factor,
    ).to(device).eval()

    assert source.resolution == target.resolution
    log(f"[step_e12] resolution={source.resolution} latent_dim={source.latent_dim}")

    rows, grid_tiles = [], []
    t0 = time.time()
    for idx, path in enumerate(image_paths):
        img01 = load_image_as_tensor(path, source.resolution)
        if args.deterministic_inversion:
            translated, original_resized = translate_one_deterministic(source, target, img01, device, args, log)
        else:
            translated, original_resized = translate_one(source, target, img01, device, remove_bg=args.remove_background)
            if args.region_color_correct:
                corrected = region_color_correct(
                    translated.unsqueeze(0).to(device), original_resized.unsqueeze(0).to(device),
                    device, args.region_color_strength, log,
                )
                translated = corrected.squeeze(0).cpu()

        pair_path = os.path.join(
            args.output_dir, f"pair_{idx:02d}_{os.path.splitext(os.path.basename(path))[0]}.png"
        )
        save_side_by_side(original_resized, translated, pair_path)
        grid_tiles.extend([original_resized, translated])

        rows.append({
            "image": os.path.basename(path),
            "orig_saturation": mean_saturation(original_resized),
            "translated_saturation": mean_saturation(translated),
            "orig_laplacian_variance": laplacian_variance(original_resized),
            "translated_laplacian_variance": laplacian_variance(translated),
        })
        log(f"[step_e12] [{idx+1}/{len(image_paths)}] {os.path.basename(path)} -> {pair_path} "
            f"(sat {rows[-1]['orig_saturation']:.3f} -> {rows[-1]['translated_saturation']:.3f}, "
            f"{time.time() - t0:.1f}s elapsed)")

    grid_path = os.path.join(args.output_dir, "comparison_grid.png")
    save_image_grid(torch.stack(grid_tiles, dim=0), grid_path, nrow=2)
    log(f"[step_e12] saved comparison grid (left=original, right=translated) to {grid_path}")

    csv_path = os.path.join(args.output_dir, "translation_metrics.csv")
    write_csv(csv_path, rows, fieldnames=list(rows[0].keys()))

    with open(os.path.join(args.output_dir, "run_log.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
