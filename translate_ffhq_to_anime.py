"""
Translate an image (or a directory of images) from the FFHQ (photo) domain to
the anime-face domain using this repo's unpaired DDPM/DDIM image-to-image
translation (model/gan_wrapper/ddpm_ddim_wrapper.py): DDIM-invert with the
source (FFHQ) model, DDIM-decode with the target (anime) model.

This is the same encode-with-source/decode-with-target pattern as
style_pipeline/ddpm_bridge.py's DDPMStyleTransfer class in the
3dgs-for-styletransfer project (which vendors this repo as a git submodule
and drives it identically for 3D-Gaussian-Splatting renders), adapted here
into a standalone script for plain image files.

By default this expects the two custom domain configs bundled in this repo
at ckpts/ddpm/configs/{ffhq_custom,anime}.yml (see those files' own header
comments for the architecture they assume) -- you still need to supply the
actual trained checkpoint weights via --source_model_path/--target_model_path
(no default path exists for a custom domain).

Usage:
  python translate_ffhq_to_anime.py \\
      --input path/to/photo.png --output path/to/out.png \\
      --source_model_path /path/to/ffhq_custom_checkpoint.pt \\
      --target_model_path /path/to/anime_checkpoint.pt

  # Or a whole directory (output must then also be a directory):
  python translate_ffhq_to_anime.py \\
      --input path/to/photos/ --output path/to/out_dir/ \\
      --source_model_path ... --target_model_path ...

Via `docker compose run ffhq2anime` (see docker-compose.yml -- its
entrypoint already runs this script, so only pass this script's own flags):
  docker compose run --rm ffhq2anime \\
      --input input.png --output output.png \\
      --source_model_path ckpts/ddpm/ffhq_custom.pt \\
      --target_model_path ckpts/ddpm/anime.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)  # DDPMDDIMWrapper resolves ckpts/ddpm/configs/<...>.yml relative to CWD.

from model.gan_wrapper.ddpm_ddim_wrapper import DDPMDDIMWrapper  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, required=True, help="Image file or directory of images.")
    parser.add_argument("--output", type=str, required=True,
                         help="Output image file (if --input is a file) or directory (if --input is a directory).")
    parser.add_argument("--source_model_type", type=str, default="ffhq_custom",
                         help="Passed to DDPMDDIMWrapper. Default resolves to "
                              "ckpts/ddpm/configs/ffhq_custom.yml. Use 'ffhq256' for the original "
                              "ILVR checkpoint instead (requires ckpts/ddpm/ffhq_10m.pt).")
    parser.add_argument("--source_model_path", type=str, default=None,
                         help="Required unless --source_model_type ffhq256.")
    parser.add_argument("--target_model_type", type=str, default="anime",
                         help="Passed to DDPMDDIMWrapper. Default resolves to ckpts/ddpm/configs/anime.yml.")
    parser.add_argument("--target_model_path", type=str, required=True, help="Path to the anime checkpoint.")
    parser.add_argument("--sample_type", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--custom_steps", type=int, default=1000)
    parser.add_argument("--es_steps", type=int, default=850)
    parser.add_argument("--eta", type=float, default=0.1, help="Only used when --sample_type ddim.")
    parser.add_argument("--refine_steps", type=int, default=100)
    parser.add_argument("--t_0", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_image01(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def save_image01(t01: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    arr = (t01.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


@torch.no_grad()
def translate_one(source: DDPMDDIMWrapper, target: DDPMDDIMWrapper, img01: torch.Tensor,
                   device: torch.device) -> torch.Tensor:
    """img01: (3, H, W) float tensor in [0, 1]. Returns a (3, H, W) tensor in
    [0, 1], resized back to (H, W). The DDPM models operate at a fixed native
    resolution (source.resolution), so the image is downsampled for
    translation and the result is upsampled back, without preserving aspect
    ratio -- same behavior as style_pipeline/ddpm_bridge.py's `.stylize()`.
    """
    _, h, w = img01.shape
    low_res = F.interpolate(
        img01.unsqueeze(0).to(device), size=(source.resolution, source.resolution),
        mode="bilinear", align_corners=False,
    )
    z = source.encode(low_res)
    styled = target(z=z)
    styled = torch.clamp(styled, 0.0, 1.0)
    styled = F.interpolate(styled, size=(h, w), mode="bilinear", align_corners=False)
    return styled.squeeze(0).cpu()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.source_model_type != "ffhq256" and args.source_model_path is None:
        raise ValueError(
            f"--source_model_type {args.source_model_type!r} has no default checkpoint path -- "
            f"pass --source_model_path."
        )

    # DDPMDDIMWrapper.__init__ asserts eta is None when sample_type='ddpm' (and >0 when 'ddim').
    wrapper_eta = args.eta if args.sample_type == "ddim" else None

    print(f"[translate] device={device} source={args.source_model_type} target={args.target_model_type} "
          f"sample_type={args.sample_type} custom_steps={args.custom_steps} es_steps={args.es_steps} "
          f"eta={wrapper_eta} refine_steps={args.refine_steps}")

    common_kwargs = dict(
        sample_type=args.sample_type, custom_steps=args.custom_steps, es_steps=args.es_steps,
        eta=wrapper_eta, refine_steps=args.refine_steps, t_0=args.t_0,
    )
    print("[translate] loading source model...")
    source = DDPMDDIMWrapper(
        source_model_type=args.source_model_type, source_model_path=args.source_model_path,
        **common_kwargs,
    ).to(device).eval()
    print("[translate] loading target model...")
    target = DDPMDDIMWrapper(
        source_model_type=args.target_model_type, source_model_path=args.target_model_path,
        **common_kwargs,
    ).to(device).eval()
    assert source.resolution == target.resolution, (
        f"source ({source.resolution}) and target ({target.resolution}) DDPM models must share the "
        f"same native resolution"
    )

    if os.path.isdir(args.input):
        files = sorted(f for f in os.listdir(args.input) if f.lower().endswith(IMAGE_EXTS))
        if not files:
            raise FileNotFoundError(f"No images found in {args.input}")
        os.makedirs(args.output, exist_ok=True)
        for fname in files:
            img01 = load_image01(os.path.join(args.input, fname))
            out01 = translate_one(source, target, img01, device)
            out_path = os.path.join(args.output, fname)
            save_image01(out01, out_path)
            print(f"[translate] {fname} -> {out_path}")
    else:
        img01 = load_image01(args.input)
        out01 = translate_one(source, target, img01, device)
        save_image01(out01, args.output)
        print(f"[translate] {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
