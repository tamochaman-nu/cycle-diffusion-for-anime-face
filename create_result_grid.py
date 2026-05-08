import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def main():
    # パラメータ設定
    # 縦軸: fbsdiff_cutoff (0.0 to 0.8, 0.1 step)
    cutoffs = [round(x, 1) for x in np.arange(0.0, 0.9, 0.1)]
    # 横軸: fbs_end_step (0 to 100, 10 step)
    end_steps = range(0, 110, 10)
    
    base_dir = "output/optimize_wvloss"
    img_filename = "eval_256_000000.png"
    
    # サンプル画像からサイズを取得
    sample_run = "translate_ffhq256_to_anime256_10000_eta0001_free_inv_fbsdiff000_000stp_100stp_020rstp_wvloss001"
    sample_img_path = os.path.join(base_dir, sample_run, img_filename)
    
    if os.path.exists(sample_img_path):
        with Image.open(sample_img_path) as tmp:
            img_w, img_h = tmp.size
    else:
        img_w, img_h = 256, 256 # デフォルト
        
    # 余白（ラベル用）
    margin_left = 150
    margin_top = 100
    
    # 全体キャンバス作成
    grid_w = margin_left + img_w * len(end_steps)
    grid_h = margin_top + img_h * len(cutoffs)
    canvas = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # フォント設定 (デフォルト)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    print("Generating grid image...")

    # 画像の配置
    for r, cutoff in enumerate(cutoffs):
        # 縦軸ラベル
        cutoff_str = f"{int(round(cutoff * 100)):03d}"
        label_y = margin_top + r * img_h + img_h // 2
        draw.text((10, label_y), f"cutoff: {cutoff}", fill=(0, 0, 0))
        
        for c, end_step in enumerate(end_steps):
            # 横軸ラベル (最初の行のみ)
            if r == 0:
                label_x = margin_left + c * img_w + img_w // 4
                draw.text((label_x, 40), f"end_step: {end_step}", fill=(0, 0, 0))
                
            step_str = f"{int(end_step):03d}"
            # フォルダ名の構築
            run_name = f"translate_ffhq256_to_anime256_10000_eta0001_free_inv_fbsdiff{cutoff_str}_{step_str}stp_100stp_020rstp_wvloss001"
            img_path = os.path.join(base_dir, run_name, img_filename)
            
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    canvas.paste(img, (margin_left + c * img_w, margin_top + r * img_h))
            else:
                # 未完了または存在しない場合はグレー
                placeholder = Image.new('RGB', (img_w, img_h), (80, 80, 80))
                canvas.paste(placeholder, (margin_left + c * img_w, margin_top + r * img_h))
                draw.text((margin_left + c * img_w + 10, margin_top + r * img_h + 10), "Pending/Missing", fill=(200, 200, 200))

    # 保存
    output_path = "fbsdiff_grid_results.png"
    canvas.save(output_path)
    print(f"Success! Grid result saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
