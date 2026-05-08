from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class CustomTrainingArguments(TrainingArguments):

    cfg: str = None

    verbose: bool = field(
        default=False,
    )

    eta: Optional[float] = field(
        default=None,
        metadata={"help": "DDIM eta (stochasticity). Overrides the value in the config file."},
    )

    guidance_scale: Optional[float] = field(
        default=None,
        metadata={"help": "Unconditional guidance scale. Overrides the value in the config file."},
    )

    # --- FreeInv ---
    use_freeinv: Optional[bool] = field(
        default=None,
        metadata={"help": "FreeInv: ランダム可逆変換で DDIM inversion 誤差を低減する。"},
    )

    freeinv_seed: Optional[int] = field(
        default=None,
        metadata={"help": "FreeInv 変換列の乱数シード（encode/generate で共有）。"},
    )

    # --- SimInversion ---
    use_siminversion: Optional[bool] = field(
        default=None,
        metadata={"help": "SimInversion: inversion 時の guidance scale を固定する（テキスト条件付きモデル向け）。"},
    )

    source_guidance_scale: Optional[float] = field(
        default=None,
        metadata={"help": "SimInversion: inversion 時の guidance scale（デフォルト 1.0）。"},
    )

    target_guidance_scale: Optional[float] = field(
        default=None,
        metadata={"help": "SimInversion: generation 時の guidance scale（デフォルト 2.0）。"},
    )

    # --- TABA ---
    taba_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "TABA: 最初の何割の inversion ステップを forward diffusion で置き換えるか（0.0-1.0）。"},
    )

    # --- FBSDiff ---
    use_fbsdiff: Optional[bool] = field(
        default=None,
        metadata={"help": "FBSDiff: 周波数分離ガイダンスで構造を保持する。"},
    )

    fbsdiff_cutoff: Optional[float] = field(
        default=None,
        metadata={"help": "FBSDiff: 低/高周波の境界（0.1-0.5 推奨）。"},
    )

    fbsdiff_start_step: Optional[int] = field(
        default=None,
        metadata={"help": "FBSDiff を適用し始めるステップ番号。"},
    )

    fbsdiff_end_step: Optional[int] = field(
        default=None,
        metadata={"help": "FBSDiff を適用し終えるステップ番号。"},
    )

    fbsdiff_cache_every: Optional[int] = field(
        default=None,
        metadata={"help": "FBSDiff: N ステップごとに inversion latent をキャッシュする（メモリ節約）。"},
    )

    # --- デバッグ用中間出力 ---
    save_intermediate: Optional[bool] = field(
        default=None,
        metadata={"help": "中間 latent を画像として保存する（構造崩壊の診断用）。"},
    )

    intermediate_dir: Optional[str] = field(
        default=None,
        metadata={"help": "中間画像の保存先ディレクトリ。"},
    )

    show_progress: bool = field(
        default=True,
        metadata={"help": "変換処理（Diffusion steps）の進捗バーを表示するかどうか。"},
    )

    use_domain_embed: Optional[bool] = field(
        default=None,
        metadata={"help": "モデルアーキテクチャで domain_embed を使用するかどうかを指定します（waveletloss など用）。"},
    )





