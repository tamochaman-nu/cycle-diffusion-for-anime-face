"""
FFHQ (photo) -> anime translation by StyleID-style self-attention K/V
injection (see pipeline/styleid.py for the mechanism): the photo supplies the
attention queries (layout), an anime reference image supplies the keys/values
(appearance), both read from the anime model's own decoder self-attention.

Unlike translate_ffhq_to_anime.py (CycleDiffusion), this needs one or more
anime STYLE REFERENCE images; every input photo is translated once per
reference. Currently supports MPUNetModel (arch: "mp") configs only, since the
injection hooks into MPBlock.attn_hook.

Usage (via the diagA service, whose entrypoint is plain `python`):
  docker compose run --rm diagA translate_styleid.py \\
      --input path/to/photos/ --style path/to/anime_refs/ --output path/to/out_dir/ \\
      --target_model_type guided_diffusion_mp128_face256_cosine \\
      --target_model_path ckpts/ddpm/anime-mp-100000-128-ema09995.pt

Outputs: <output>/<style stem>/<input filename> for every (style, input)
pair, plus <output>/comparison.png (rows = inputs, columns = input then one
output per style, with the style images along the top row).
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)  # DDPMDDIMWrapper resolves ckpts/ddpm/configs/<...>.yml relative to CWD.

from model.gan_wrapper.ddpm_ddim_wrapper import DDPMDDIMWrapper  # noqa: E402
from pipeline import styleid  # noqa: E402
from translate_ffhq_to_anime import IMAGE_EXTS, load_image01, save_image01  # noqa: E402


def _str2bool(x):
    return str(x).lower() in ("1", "true", "yes", "y")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, required=True, help="Photo file or directory of photos.")
    parser.add_argument("--style", type=str, nargs="+", required=True,
                         help="Anime reference image file(s) and/or directories of them.")
    parser.add_argument("--output", type=str, required=True, help="Output directory.")
    parser.add_argument("--target_model_type", type=str, default="guided_diffusion_mp128_face256_cosine")
    parser.add_argument("--target_model_path", type=str, required=True)
    parser.add_argument("--content_inversion", type=str, default="target", choices=["target", "source"],
                         help="Which model DDIM-inverts the photo. 'target' (default) is the paper's single-model "
                              "setup; 'source' inverts with the FFHQ model (DDIB-style) while the anime model "
                              "still supplies the recorded queries. Requires --source_model_* for 'source'.")
    parser.add_argument("--source_model_type", type=str, default="guided_diffusion_mp128_face256_cosine")
    parser.add_argument("--source_model_path", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=50,
                         help="DDIM steps for inversion and generation (StyleID uses 50).")
    # Not 999: with the cosine schedule sqrt(alphabar) is ~5e-5 at t=999 (vs
    # 0.08 at 950), so DDIM's x0 prediction amplifies eps error ~20000x there.
    # A no-injection round trip of a photo (anime-mp-100000-128) collapsed to
    # noise at 999, faded at 980, and reconstructed closely at 950.
    parser.add_argument("--t_0", type=int, default=950, help="Inversion depth.")
    parser.add_argument("--gamma", type=float, default=0.75,
                         help="Query preservation: q = gamma * Q_content + (1 - gamma) * q_current.")
    parser.add_argument("--tau", type=float, default=1.5, help="Attention temperature (q scaled by tau).")
    parser.add_argument("--layers", type=str, nargs="+", default=["dec.16x16", "dec.32x32"],
                         help="Name prefixes of the decoder attention blocks to inject into.")
    # Off by default, unlike the paper (whose latent is SD's VAE space): here
    # x_T is pixel-space, and on diagnostics/outputs/step_e18_styleid AdaIN
    # turned a line-art reference into dense scribble texture over the whole
    # face (and, without injection, into saturated noise), while injection
    # without AdaIN transferred that same reference's flat fill cleanly.
    parser.add_argument("--adain", type=_str2bool, default=False, help="AdaIN the initial latent to the style's.")
    parser.add_argument("--inject", type=_str2bool, default=True,
                         help="false = ablation: same inversion/AdaIN/decode with no K/V injection.")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def list_images(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.lower().endswith(IMAGE_EXTS)]
        else:
            files.append(p)
    if not files:
        raise FileNotFoundError(f"No images found in {paths}")
    return files


def load_wrapper(model_type, model_path, num_steps, device):
    # eta/sample_type only satisfy DDPMDDIMWrapper's own asserts -- this script
    # never calls its encode()/generate(), only uses .generator and the schedule.
    wrapper = DDPMDDIMWrapper(
        source_model_type=model_type, source_model_path=model_path,
        sample_type="ddim", custom_steps=num_steps, es_steps=num_steps, eta=0.1,
    ).to(device).eval()
    if not hasattr(wrapper.generator, "dec"):
        raise ValueError(f"{model_type} is not an MPUNetModel (arch: \"mp\") config; StyleID injection needs one.")
    return wrapper


def to_model_space(img01, resolution, device):
    """(3, H, W) in [0, 1] -> (1, 3, R, R) in [-1, 1]."""
    x = F.interpolate(img01.unsqueeze(0).to(device), size=(resolution, resolution),
                      mode="bilinear", align_corners=False)
    return (x - 0.5) * 2.0


def save_grid(rows, path, tile=192):
    """rows: list of lists of (3, H, W) [0, 1] tensors (None = blank tile)."""
    ncol = max(len(r) for r in rows)
    grid = torch.ones(3, tile * len(rows), tile * ncol)
    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            if img is not None:
                grid[:, i * tile:(i + 1) * tile, j * tile:(j + 1) * tile] = F.interpolate(
                    img.unsqueeze(0), size=(tile, tile), mode="bilinear", align_corners=False)[0]
    save_image01(grid, path)


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.content_inversion == "source" and args.source_model_path is None:
        raise ValueError("--content_inversion source requires --source_model_path.")

    inputs = list_images([args.input])
    styles = list_images(args.style)

    target = load_wrapper(args.target_model_type, args.target_model_path, args.num_steps, device)
    source = (load_wrapper(args.source_model_type, args.source_model_path, args.num_steps, device)
              if args.content_inversion == "source" else target)
    assert source.resolution == target.resolution
    res = target.resolution
    consts = dict(betas=target.betas, logvar=target.logvar, learn_sigma=target.learn_sigma)
    if source is not target:
        assert torch.equal(source.betas, target.betas), "source/target noise schedules differ"

    injector = styleid.StyleIDInjector(target.generator, args.layers, gamma=args.gamma, tau=args.tau)
    seq = styleid.timestep_sequence(args.t_0, args.num_steps)
    print(f"[styleid] layers={injector.layer_names} gamma={args.gamma} tau={args.tau} adain={args.adain} "
          f"inject={args.inject} steps={args.num_steps} t_0={args.t_0} content_inversion={args.content_inversion}")

    # Invert every photo once; recorded queries are kept per photo (they don't
    # depend on the style), so each extra style only costs its own inversion
    # plus one generation per photo.
    contents = []
    for path in inputs:
        img01 = load_image01(path)
        injector.content = {}
        x_T = styleid.invert_and_record(
            source.generator, target.generator, injector, "record_content",
            to_model_space(img01, res, device), seq, **consts,
        )
        contents.append((path, img01, x_T, injector.content))
        print(f"[styleid] inverted {os.path.basename(path)}")

    outputs = {}
    for style_path in styles:
        style01 = load_image01(style_path)
        injector.style = {}
        x_T_style = styleid.invert_and_record(
            target.generator, target.generator, injector, "record_style",
            to_model_space(style01, res, device), seq, **consts,
        )
        stem = os.path.splitext(os.path.basename(style_path))[0]
        for path, img01, x_T, content_feats in contents:
            injector.content = content_feats
            x = styleid.adain(x_T, x_T_style) if args.adain else x_T
            x0 = styleid.generate(target.generator, injector, x, seq, inject=args.inject, **consts)
            out01 = ((x0 + 1.0) / 2.0).clamp(0, 1)
            out01 = F.interpolate(out01, size=img01.shape[1:], mode="bilinear", align_corners=False)[0].cpu()
            out_path = os.path.join(args.output, stem, os.path.basename(path))
            save_image01(out01, out_path)
            outputs[(style_path, path)] = out01
            print(f"[styleid] {os.path.basename(path)} x {stem} -> {out_path}")

    rows = [[None] + [load_image01(s) for s in styles]]
    for path, img01, _, _ in contents:
        rows.append([img01] + [outputs[(s, path)] for s in styles])
    save_grid(rows, os.path.join(args.output, "comparison.png"))
    print(f"[styleid] grid -> {os.path.join(args.output, 'comparison.png')}")


if __name__ == "__main__":
    main()
