"""
Standalone inference script for cycle-diffusion (DDPM/DDIM unconditional models).

Usage:
  python inference.py \
    --source_image /path/to/baby.jpg \
    --source_model_type ffhq256 \
    --source_model_path ckpts/ddpm/ffhq070000.pt \
    --target_model_type anime256 \
    --target_model_path ckpts/ddpm/anime030000.pt \
    --num_inference_steps 1000 \
    --es_steps 850 \
    --eta 0.01 \
    --use_freeinv True \
    --freeinv_seed 42 \
    --taba_ratio 0.2 \
    --use_fbsdiff True \
    --fbsdiff_cutoff 0.3 \
    --output_path output/result.png

NOTE: --source_prompt / --target_prompt は DDPM（非条件付き）モデルでは無視される。
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def str_to_bool(v):
    return v.lower() in ('true', '1', 'yes')


def parse_args():
    p = argparse.ArgumentParser()

    # 入出力
    p.add_argument('--source_image',  required=True)
    p.add_argument('--output_path',   default='output/inference_result.png')
    p.add_argument('--source_prompt', default='', help='(DDPM では無視)')
    p.add_argument('--target_prompt', default='', help='(DDPM では無視)')

    # モデル設定
    p.add_argument('--source_model_type', default='ffhq256')
    p.add_argument('--source_model_path', default='ckpts/ddpm/ffhq070000.pt')
    p.add_argument('--target_model_type', default='anime256')
    p.add_argument('--target_model_path', default='ckpts/ddpm/anime030000.pt')
    p.add_argument('--sample_type',       default='ddim')
    p.add_argument('--num_inference_steps', type=int, default=1000)
    p.add_argument('--es_steps',            type=int, default=850)
    p.add_argument('--refine_steps',        type=int, default=100)
    p.add_argument('--eta',                 type=float, default=0.01)

    # FreeInv
    p.add_argument('--use_freeinv',   type=str_to_bool, default=True)
    p.add_argument('--freeinv_seed',  type=int,         default=42)

    # SimInversion（DDPM では guidance scale 概念なし、将来の拡張用）
    p.add_argument('--use_siminversion',      type=str_to_bool, default=False)
    p.add_argument('--source_guidance_scale', type=float,       default=1.0)
    p.add_argument('--target_guidance_scale', type=float,       default=1.0)

    # TABA
    p.add_argument('--taba_ratio', type=float, default=0.2)

    # FBSDiff
    p.add_argument('--use_fbsdiff',        type=str_to_bool, default=True)
    p.add_argument('--fbsdiff_cutoff',     type=float,       default=0.3)
    p.add_argument('--fbsdiff_start_step', type=int,         default=0)
    p.add_argument('--fbsdiff_end_step',   type=int,         default=30)
    p.add_argument('--fbsdiff_cache_every',type=int,         default=1)

    # デバッグ用中間出力
    p.add_argument('--save_intermediate',  type=str_to_bool, default=False)
    p.add_argument('--intermediate_dir',   default='./debug')

    # デバイス
    p.add_argument('--device', default='cuda')

    return p.parse_args()


def load_image(path: str, resolution: int) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
    return tf(img).unsqueeze(0)  # (1, C, H, W), range [0, 1]


def build_wrapper(
    source_model_type, source_model_path,
    sample_type, custom_steps, es_steps, refine_steps, eta,
    use_freeinv, freeinv_seed,
    use_siminversion, source_guidance_scale, target_guidance_scale,
    taba_ratio,
    use_fbsdiff, fbsdiff_cutoff, fbsdiff_start_step, fbsdiff_end_step, fbsdiff_cache_every,
    save_intermediate, intermediate_dir,
):
    from model.gan_wrapper.ddpm_ddim_wrapper import DDPMDDIMWrapper
    return DDPMDDIMWrapper(
        source_model_type=source_model_type,
        source_model_path=source_model_path,
        sample_type=sample_type,
        custom_steps=custom_steps,
        es_steps=es_steps,
        refine_steps=refine_steps,
        eta=eta,
        use_freeinv=use_freeinv,
        freeinv_seed=freeinv_seed,
        use_siminversion=use_siminversion,
        source_guidance_scale=source_guidance_scale,
        target_guidance_scale=target_guidance_scale,
        taba_ratio=taba_ratio,
        use_fbsdiff=use_fbsdiff,
        fbsdiff_cutoff=fbsdiff_cutoff,
        fbsdiff_start_step=fbsdiff_start_step,
        fbsdiff_end_step=fbsdiff_end_step,
        fbsdiff_cache_every=fbsdiff_cache_every,
        save_intermediate=save_intermediate,
        intermediate_dir=intermediate_dir,
    )


def main():
    args = parse_args()

    # プロジェクトルートから実行することを想定（相対パス解決のため）
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    sys.path.insert(0, project_root)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.source_prompt or args.target_prompt:
        print("NOTE: --source_prompt / --target_prompt は DDPM（非条件付き）モデルでは無視されます。")

    # --- ソースモデル（エンコーダ）の構築 ---
    print(f"\n[Source] Loading {args.source_model_type} from {args.source_model_path}")
    source_wrapper = build_wrapper(
        source_model_type=args.source_model_type,
        source_model_path=args.source_model_path,
        sample_type=args.sample_type,
        custom_steps=args.num_inference_steps,
        es_steps=args.es_steps,
        refine_steps=args.refine_steps,
        eta=args.eta,
        use_freeinv=args.use_freeinv,
        freeinv_seed=args.freeinv_seed,
        use_siminversion=args.use_siminversion,
        source_guidance_scale=args.source_guidance_scale,
        target_guidance_scale=args.target_guidance_scale,
        taba_ratio=args.taba_ratio,
        use_fbsdiff=args.use_fbsdiff,
        fbsdiff_cutoff=args.fbsdiff_cutoff,
        fbsdiff_start_step=args.fbsdiff_start_step,
        fbsdiff_end_step=args.fbsdiff_end_step,
        fbsdiff_cache_every=args.fbsdiff_cache_every,
        save_intermediate=args.save_intermediate,
        intermediate_dir=os.path.join(args.intermediate_dir, 'source'),
    ).to(device)
    source_wrapper.eval()

    # --- ターゲットモデル（デコーダ）の構築 ---
    print(f"\n[Target] Loading {args.target_model_type} from {args.target_model_path}")
    target_wrapper = build_wrapper(
        source_model_type=args.target_model_type,
        source_model_path=args.target_model_path,
        sample_type=args.sample_type,
        custom_steps=args.num_inference_steps,
        es_steps=args.es_steps,
        refine_steps=args.refine_steps,
        eta=args.eta,
        use_freeinv=args.use_freeinv,
        freeinv_seed=args.freeinv_seed,
        use_siminversion=args.use_siminversion,
        source_guidance_scale=args.source_guidance_scale,
        target_guidance_scale=args.target_guidance_scale,
        taba_ratio=args.taba_ratio,
        use_fbsdiff=args.use_fbsdiff,
        fbsdiff_cutoff=args.fbsdiff_cutoff,
        fbsdiff_start_step=args.fbsdiff_start_step,
        fbsdiff_end_step=args.fbsdiff_end_step,
        fbsdiff_cache_every=args.fbsdiff_cache_every,
        save_intermediate=args.save_intermediate,
        intermediate_dir=os.path.join(args.intermediate_dir, 'target'),
    ).to(device)
    target_wrapper.eval()

    resolution = source_wrapper.resolution
    assert source_wrapper.resolution == target_wrapper.resolution, \
        "Source and target model resolutions must match."

    # --- 入力画像の読み込み ---
    print(f"\n[Input] {args.source_image} (resized to {resolution}x{resolution})")
    image = load_image(args.source_image, resolution).to(device)

    # --- エンコード ---
    print("\n[Encode] Running DDIM inversion ...")
    with torch.no_grad():
        z = source_wrapper.encode(image)

    inversion_latents = source_wrapper._inversion_latents
    print(f"  z.shape = {z.shape}, "
          f"inversion_latents cached steps = {len(inversion_latents)}")

    # --- デコード ---
    print("\n[Decode] Generating translated image ...")
    with torch.no_grad():
        output = target_wrapper(z=z, inversion_latents=inversion_latents)

    # --- 出力保存 ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    save_image(output, args.output_path)
    print(f"\n[Done] Saved to {args.output_path}")

    # 元画像と並べた比較画像も保存
    source_resized = transforms.Resize(resolution)(
        transforms.CenterCrop(resolution)(image)
    )
    post = transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])
    comparison = torch.cat([source_resized, output], dim=-1)  # 横に並べる
    base, ext = os.path.splitext(args.output_path)
    save_image(comparison, base + '_comparison' + ext)
    print(f"[Done] Comparison saved to {base + '_comparison' + ext}")


if __name__ == '__main__':
    main()
