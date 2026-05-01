# CycleDiffusion

<br> 
<div align=center>
    <img src="docs/teaser.png" align="middle", width=900>
</div>
<br> 

論文の公式 PyTorch 実装：<br>
**Unifying Diffusion Models' Latent Space, with Applications to CycleDiffusion and Guidance** <br>
Chen Henry Wu, Fernando De la Torre <br>
Carnegie Mellon University <br>
_プレプリント, 2022年10月_ <br>

本論文の改訂版が [ICCV 2023](https://iccv2023.thecvf.com/) に採択されました：<br>
**A Latent Space of Stochastic Diffusion Models for Zero-Shot Image Editing and Guidance** <br>
Chen Henry Wu, Fernando De la Torre <br>
Carnegie Mellon University <br>
_ICCV 2023_ <br>

[**[論文リンク]**](https://arxiv.org/abs/2210.05559) | [**[ICCV版]**](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_A_Latent_Space_of_Stochastic_Diffusion_Models_for_Zero-Shot_Image_ICCV_2023_paper.pdf) | [**[Diffusers 🧨 実装]**](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cycle_diffusion) | [**[HuggingFace 🤗 デモ]**](https://huggingface.co/spaces/ChenWu98/Stable-CycleDiffusion)

## 更新履歴

**[2022年10月13日]** コード公開。最初の ArXiv 版のセクション 4.3 は [Unified Generative Zoo](https://github.com/ChenWu98/unified-generative-zoo) にてオープンソース化されています。

**[2022年11月9日]** CycleDiffusion が HuggingFace 🤗 [Diffusers](https://github.com/huggingface/diffusers) 🧨 のパイプラインとして利用可能になりました。[パイプラインのドキュメント](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cycle_diffusion) をご確認ください。

**[2022年11月10日]** HuggingFace 🤗 Spaces を使用したデモが [Stable CycleDiffusion](https://huggingface.co/spaces/ChenWu98/Stable-CycleDiffusion) で公開されました。

## 概要

拡散モデルにおけるランダム性は魔法のようなものだと考えています。「ランダムシード」を固定することで、2つの画像分布から最小限の差異で画像を生成できることが蓄積された証拠によって示されています。本論文はまさに、**この「ランダムシード」をどのように形式化するか**、そして**与えられた実画像からどのように推定するか**を扱っています。

本論文の形式化と導出は定義に基づいており、そこからいくつかの驚くべき結果が得られることを示しています。このリポジトリには **CycleDiffusion** のコードが含まれています。これは驚くほどシンプルな手法で、以下のことが可能です：

1. Stable Diffusion などのテキスト画像拡散モデルを用いたゼロショット画像変換
2. 2つの関連ドメインで学習された拡散モデルを用いた従来の非ペア画像変換

ゼロショット画像変換の結果をご確認ください。タスクの入力を $(\boldsymbol{x}, \boldsymbol{t}, \hat{\boldsymbol{t}})$ の三つ組として定義しています：

1. $\boldsymbol{x}$ はソース画像で、紫色の枠で表示されます。
2. $\boldsymbol{t}$ はソーステキストで、テキストスパンが紫色でマークされています。
3. $\hat{\boldsymbol{t}}$ はターゲットテキストで、ソーステキストと重複するスパンは $[\ldots]$ と省略されます。

実験では [Stable Diffusion](https://github.com/CompVis/stable-diffusion) を使用しています。特筆すべき点として、すべてのソース画像 $\boldsymbol{x}$ は**実写画像**です。DALL∙E 2 で生成されたものも含まれていますが、それらは Stable Diffusion にとっては「実写」とみなせます :)

<div align=center>
    <img src="docs/text.png" align="middle", width=780>
</div>

<br>

以下はベースラインとの比較です。

<div align=center>
    <img src="docs/text_baseline.png" align="middle" width=470>
</div>

## 目次

- [CycleDiffusion](#cyclediffusion)
  - [更新履歴](#更新履歴)
  - [概要](#概要)
  - [目次](#目次)
  - [アニメ顔変換（フォーク改変）](#アニメ顔変換フォーク改変)
    - [チェックポイントの準備](#チェックポイントの準備)
    - [実行コマンド](#実行コマンド)
    - [出力の構造崩壊への対処](#出力の構造崩壊への対処)
    - [CLIパラメータ一覧](#cliパラメータ一覧)
  - [依存環境](#依存環境)
  - [評価データ](#評価データ)
  - [事前学習済み拡散モデル](#事前学習済み拡散モデル)
  - [使い方](#使い方)
    - [テキスト画像拡散モデルを用いたゼロショット画像変換](#テキスト画像拡散モデルを用いたゼロショット画像変換)
    - [ゼロショット画像変換のカスタム使用](#ゼロショット画像変換のカスタム使用)
    - [2ドメイン拡散モデルを用いた非ペア画像変換](#2ドメイン拡散モデルを用いた非ペア画像変換)
  - [引用](#引用)
  - [社会的影響について](#社会的影響について)
  - [ライセンス](#ライセンス)
  - [連絡先](#連絡先)


## アニメ顔変換（フォーク改変）

このフォークは、CycleDiffusion をピクセル空間 DDPM モデルによる実写顔写真 → アニメ風顔画像への変換（FFHQ256 → Anime256）に適用したものです。出力の構造崩壊を抑制する4つの推論改善手法、スタンドアロン実行スクリプト `inference.py`、および Docker Compose 対応を追加しています。

### チェックポイントの準備

DDPM チェックポイントファイルを `ckpts/ddpm/` に配置してください。

利用可能なチェックポイントとモデルタイプの対応は以下のとおりです。

| ファイル名 | `model_type` 文字列 | 解像度 | 備考 |
|---|---|---|---|
| `ffhq_10000.pt` | `ffhq256` | 256px | FFHQ 学習初期チェックポイント |
| `ffhq070000.pt` | `ffhq256_v2` | 256px | FFHQ 70k steps チェックポイント（推奨） |
| `anime_10000.pt` | `anime256` | 256px | アニメ顔 学習初期チェックポイント |
| `anime030000.pt` | `anime512` | **512px** | アニメ顔 512px モデル（256px パイプラインとは非互換） |

> **注意**: `anime030000.pt` は 512px モデルです。256px パイプラインで使用する場合は `anime_10000.pt` を選択してください。

### 実行コマンド

プロジェクトは Docker コンテナ内で動作します。

#### メインパイプライン（ffhq_10000.pt → anime_10000.pt）

```shell
docker compose run app
```

- ソースモデル: `ckpts/ddpm/ffhq_10000.pt`（モデルタイプ: `ffhq256`）
- ターゲットモデル: `ckpts/ddpm/anime_10000.pt`（モデルタイプ: `anime256`）
- 出力先: `output/translate_ffhq256-10000_to_anime256-10000_ddim_eta001_eta001/`

#### テストパイプライン（ffhq070000.pt → anime_10000.pt）

```shell
docker compose run app-local
```

- ソースモデル: `ckpts/ddpm/ffhq070000.pt`（モデルタイプ: `ffhq256_v2`、より多く学習済み）
- ターゲットモデル: `ckpts/ddpm/anime_10000.pt`（モデルタイプ: `anime256`）
- 出力先: `output/translate_ffhq_test256_to_anime256/`

GPU を指定する場合:

```shell
GPU_ID=0 docker compose run app
GPU_ID=1 docker compose run app-local
```

デフォルト GPU は `GPU_ID=2` です。

### 出力の構造崩壊への対処

#### 根本原因: アーキテクチャ不一致の修正

出力が崩れる最大の原因は、**チェックポイントの実際のアーキテクチャと、コード側のモデル定義（`script_util.py`）が一致していないこと**です。構造が完全に一致しない状態でチェックポイントを読み込むと、ノイズ予測が破綻し出力画像が抽象的なパターンや金属的なテクスチャに崩壊します。

本フォークでは以下の修正を実施済みです。

| チェックポイント | 修正内容 |
|---|---|
| `ffhq_10000.pt` | `num_res_blocks=1`, `resblock_updown=True`, `attention_resolutions="16"` に修正 |
| `anime_10000.pt` | `image_size=256`, `num_res_blocks=1`, `resblock_updown=False` に修正 |
| `ffhq070000.pt` | 新規モデルタイプ `ffhq256_v2` として対応（`num_res_blocks=2`, `resblock_updown=False`, `attention_resolutions="16,8"`）|
| `anime030000.pt` | 新規モデルタイプ `anime512` として対応（512px モデル） |

新しいチェックポイントを追加する場合は、以下のコマンドでアーキテクチャを確認してください。

```shell
docker compose run --rm app python3 -c "
import torch
ckpt = torch.load('ckpts/ddpm/YOUR_CKPT.pt', map_location='cpu')
state = ckpt['model'] if 'model' in ckpt else ckpt
keys = list(state.keys())
indices = set(int(k.split('.')[1]) for k in keys if k.startswith('input_blocks.'))
print('input_blocks 最大インデックス:', max(indices), '→ ブロック数:', max(indices)+1)
print('op.weight あり (resblock_updown=False):', any('op.weight' in k for k in keys))
out = state.get('out.2.weight', state.get('out.weight', None))
print('出力チャンネル数 (6=learn_sigma=True):', out.shape[0] if out is not None else 'N/A')
"
```

#### etaの調整

`eta` は DDIM の確率性を制御します。0 に近いほど決定的になり、ソース画像の構造が保持されやすくなります。

```
eta = 0.001  # 推奨値（構造保持重視）
eta = 0.01   # バランス型
eta = 0.1    # アニメスタイルへの変換が強くなるが崩壊しやすい
```

`docker-compose.yml` の `--eta` 引数、または `.cfg` ファイルの `eta =` で設定します。

#### es_steps の調整

`es_steps` は符号化（inversion）に使用するタイムステップ数です。`custom_steps`（通常1000）のうち何ステップを使うかを決めます。

```
es_steps = 850   # デフォルト。T=999 → T=150 までを符号化
es_steps = 700   # より粗い符号化。変換の自由度が増す
es_steps = 950   # より精密な符号化。ソース構造を保持しやすい
```

`.cfg` ファイルの `es_steps =` で設定します。

#### 推論改善手法のパラメータ調整

以下の4つの手法を組み合わせることで、構造崩壊をさらに抑制できます。

##### FreeInv

DDIM inversion と generation の中間 latent にランダムな可逆変換（回転・反転）を適用します。同一シードで変換列を再現することで、ステップごとの蓄積誤差を低減します。

```shell
--use_freeinv True
--freeinv_seed 42   # encode と generate で同一シードを使用すること
```

シードを変えると異なる変換列が生成されます。出力が気に入らない場合はシードを変えてみてください。

##### TABA（Timestep-Adaptive Blank Attention）

inversion の最初の `taba_ratio` 割合のステップを、通常の DDIM inversion ではなく単純なノイズ付加（forward diffusion）で置き換えます。高ノイズ領域での inversion 精度を向上させます。

```shell
--taba_ratio 0.0   # 無効（デフォルト）
--taba_ratio 0.1   # 最初の10%（85ステップ）をforward diffusionに置換
--taba_ratio 0.2   # 最初の20%（170ステップ）を置換（推奨）
```

値を大きくするとソースの大域構造が失われる代わりに、ターゲットドメインへの適合が強まります。

##### FBSDiff（Frequency Band Substitution Diffusion）

ターゲット生成の各ステップで、中間 latent の低周波成分をソース inversion の latent から注入します。顔の位置・姿勢といたグローバル構造をソースから保ちつつ、スタイル（高周波）はターゲットドメインに従います。

```shell
--use_fbsdiff True
--fbsdiff_cutoff 0.3      # 空間解像度の30%を低周波と定義（0.1〜0.5）
--fbsdiff_start_step 0    # 適用開始ステップ
--fbsdiff_end_step 30     # 適用終了ステップ（全850ステップのうち最初の30ステップ）
--fbsdiff_cache_every 1   # メモリ削減のためNステップごとにキャッシュ
```

`fbsdiff_cutoff` が大きいほどソース構造の拘束が強くなります。`fbsdiff_end_step` を小さくすると高ノイズ領域のみに適用され、細部への影響を抑えられます。

##### SimInversion

inversion 時の guidance scale を 1.0 に固定します。無条件 DDPM では実質的に効果がありませんが、テキスト条件付きモデルへの将来の拡張を想定した設定です。

```shell
--use_siminversion True
--source_guidance_scale 1.0
--target_guidance_scale 1.0
```

#### 中間出力による診断

どのステップで崩壊が起きているか確認するには、以下のフラグで中間 latent を画像として保存できます。

```shell
--save_intermediate True
--intermediate_dir ./debug
```

`debug/` ディレクトリに各タイムステップの latent が PNG として保存されます。崩壊が発生するステップを特定し、`es_steps` や各手法のパラメータを調整してください。

### CLIパラメータ一覧

`docker-compose.yml` の `command:` セクション、または `main.py` の引数として指定できます。

| 引数 | 型 | デフォルト | 説明 |
|---|---|---|---|
| `--eta` | float | `.cfg`の値 | DDIM確率性（0=決定的、1=DDPM）。低いほど構造が安定する |
| `--use_freeinv` | bool | `False` | FreeInv 可逆変換の有効化 |
| `--freeinv_seed` | int | `None` | FreeInv 変換シード（encode・generate で同一値を使うこと） |
| `--use_siminversion` | bool | `False` | SimInversion の有効化 |
| `--source_guidance_scale` | float | `1.0` | inversion 時の guidance scale（無条件 DDPM では無効） |
| `--target_guidance_scale` | float | `1.0` | generation 時の guidance scale（無条件 DDPM では無効） |
| `--taba_ratio` | float | `0.0` | 最初の何割のステップを forward diffusion に置換するか（0.0〜1.0） |
| `--use_fbsdiff` | bool | `False` | FBSDiff 周波数帯域置換の有効化 |
| `--fbsdiff_cutoff` | float | `0.3` | 低/高周波の境界（空間解像度に対する割合、0.1〜0.5推奨） |
| `--fbsdiff_start_step` | int | `0` | FBSDiff 適用開始ステップ |
| `--fbsdiff_end_step` | int | `30` | FBSDiff 適用終了ステップ |
| `--fbsdiff_cache_every` | int | `1` | Nステップごとに inversion latent をキャッシュ（メモリ削減） |
| `--save_intermediate` | bool | `False` | 中間 latent を画像として保存（崩壊診断用） |
| `--intermediate_dir` | str | `./debug` | 中間画像の保存先ディレクトリ |

`main.py` のみで有効な引数:

| 引数 | 型 | デフォルト | 説明 |
|---|---|---|---|
| `--guidance_scale` | float | `None` | テキスト条件付きモデルの guidance scale を上書き |

## 依存環境

1. 以下のコマンドで環境を作成します：
```shell
conda env create -f environment.yml
conda activate generative_prompt
pip install git+https://github.com/openai/CLIP.git
```
2. ご使用の CUDA バージョンに合わせて `torch` と `torchvision` をインストールしてください。
3. 以下のコマンドで [taming-transformers](https://github.com/CompVis/taming-transformers) をインストールします：
```shell
cd ../
git clone git@github.com:CompVis/taming-transformers.git
cd taming-transformers/
pip install -e .
cd ../
```
4. ログ記録のために [wandb](https://wandb.ai/) をセットアップしてください（アカウント登録が必要です）。`main.py` の `setup_wandb` 関数を編集して自分の認証情報を設定してください。以下のコマンドも実行が必要です：
```shell
wandb login
```

## 評価データ

1. ゼロショット画像変換用のほとんどのデータは [data/](data/) にすでに含まれています。一部の画像は AFHQ 検証セットからのもので、詳細は以下を参照してください。
2. 非ペア画像変換用（またはゼロショット画像変換で使用される一部画像）の AFHQ 検証セットを準備するには、以下を実行してください：
```shell
git clone git@github.com:clovaai/stargan-v2.git
cd stargan-v2/
bash download.sh afhq-v2-dataset
```

## 事前学習済み拡散モデル

1. Stable Diffusion
```shell
cd ckpts/
mkdir stable_diffusion
cd stable_diffusion/
# ここに Stable Diffusion の事前学習済みチェックポイントをダウンロードしてください。
# 以下のバージョンをダウンロードしてください: https://huggingface.co/CompVis/stable-diffusion-v-1-4-original
# ライセンスの都合上、チェックポイントを直接共有することはできません。
```
2. Latent Diffusion Model
```shell
cd ckpts/
wget https://www.dropbox.com/s/9lpdgs83l7tjk6c/ldm_models.zip
unzip ldm_models.zip
cd ldm_models/
mkdir text2img-large
cd text2img-large/
wget https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
wget https://www.dropbox.com/s/7pdttimz78ll0km/txt2img-1p4B-eval.yaml
```
3. DDPM（AFHQ-Dog と FFHQ は ILVR より、CelebAHQ は SDEdit より、AFHQ-Cat と -Wild は自前で学習）
```shell
cd ckpts/
mkdir ddpm
cd ddpm/
# 更新 2023年8月4日: 以下のリンク（元々 SDEdit のもの）は壊れているようです。CelebA-HQ は他のソースをお探しください（issue #24 参照）
wget https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt
wget https://www.dropbox.com/s/g4h8sv07i3hj83d/ffhq_10m.pt
wget https://www.dropbox.com/s/u74w8vaw1f8lc4k/afhq_dog_4m.pt
wget https://www.dropbox.com/s/8i5aznjwdl3b5iq/cat_ema_0.9999_050000.pt
wget https://www.dropbox.com/s/tplximipy8zxaub/wild_ema_0.9999_050000.pt
wget https://www.dropbox.com/s/vqm6bxj0zslrjxv/configs.zip
unzip configs.zip
```

## 使い方

### テキスト画像拡散モデルを用いたゼロショット画像変換

1. Stable Diffusion v1-4 を用いたゼロショット画像変換。128 個のテストサンプルを 8 グループ（各グループ 16 サンプル）に分割したため、平均指標を報告しています。
```shell
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME=translate_text2img256_stable_diffusion_stochastic_1
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1405 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
export RUN_NAME=translate_text2img256_stable_diffusion_stochastic_2
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1424 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2
export RUN_NAME=translate_text2img256_stable_diffusion_stochastic_3
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1423 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
export RUN_NAME=translate_text2img256_stable_diffusion_stochastic_4
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1422 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4
export RUN_NAME=translate_text2img256_stable_diffusion_stochastic_5
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1429 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

export CUDA_VISIBLE_DEVICES=5
export RUN_NAME=translate_text2img256_stable_diffusion_stochastic_6
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1428 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

export CUDA_VISIBLE_DEVICES=6
export RUN_NAME=translate_text2img256_stable_diffusion_stochastic_7
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1427 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

export CUDA_VISIBLE_DEVICES=7
export RUN_NAME=translate_text2img256_stable_diffusion_stochastic_8
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1426 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &
```
2. LDM テキスト画像チェックポイントを用いたゼロショット画像変換。128 サンプルを 8 グループに分割したため、平均指標を報告しています。
```shell
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME=translate_text2img256_latentdiff_stochastic_1
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1465 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 16 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
export RUN_NAME=translate_text2img256_latentdiff_stochastic_2
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1485 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 16 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2
export RUN_NAME=translate_text2img256_latentdiff_stochastic_3
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1486 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 16 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
export RUN_NAME=translate_text2img256_latentdiff_stochastic_4
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1487 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 16 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4
export RUN_NAME=translate_text2img256_latentdiff_stochastic_5
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1488 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 16 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

export CUDA_VISIBLE_DEVICES=5
export RUN_NAME=translate_text2img256_latentdiff_stochastic_6
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1489 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 16 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

export CUDA_VISIBLE_DEVICES=6
export RUN_NAME=translate_text2img256_latentdiff_stochastic_7
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1411 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 16 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &

export CUDA_VISIBLE_DEVICES=7
export RUN_NAME=translate_text2img256_latentdiff_stochastic_8
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1412 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 16 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &
```

### ゼロショット画像変換のカスタム使用

1. [この JSON ファイル](./data/translate-text.json) の末尾に自分の画像パスとソース・ターゲットのテキストペアを追加してください。追加数に制限はありません。
2. [この設定ファイル](./config/experiments/translate_text2img256_stable_diffusion_stochastic_custom.cfg) でハイパーパラメータを調整することを推奨します：
  - `decoder_unconditional_guidance_scales`: 値が大きいほどターゲットテキストへの重みが増します
  - `skip_steps`: 値が大きいほど元の画像に近くなります
  - ランダムシード: 異なるシードを使用すると異なる結果が生成されます
3. `decoder_unconditional_guidance_scales` × `skip_steps` のすべての組み合わせが列挙され、最良のものが返されます。
4. 以下のコマンドで画像を生成してください。出力は `output` フォルダーに保存されます。
```shell
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME=translate_text2img256_stable_diffusion_stochastic_custom
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1426 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &
```

### 2ドメイン拡散モデルを用いた非ペア画像変換

1. AFHQ-Cat → AFHQ-Dog（DDIM $\eta=0.1$）
```shell
export CUDA_VISIBLE_DEVICES=1
export RUN_NAME=translate_afhqcat256_to_afhqdog256_ddim_eta01
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1446 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &
```
2. AFHQ-Wild → AFHQ-Dog（DDIM $\eta=0.1$）
```shell
export CUDA_VISIBLE_DEVICES=5
export RUN_NAME=translate_afhqwild256_to_afhqdog256_ddim_eta01
export SEED=42
nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1498 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &
```


## 引用

このリポジトリが役に立った場合は、以下のように引用してください：
```
@inproceedings{cyclediffusion,
  title={Unifying Diffusion Models' Latent Space, with Applications to {CycleDiffusion} and Guidance},
  author={Chen Henry Wu and Fernando De la Torre},
  booktitle={ArXiv},
  year={2022},
}
```
または
```
@inproceedings{cyclediffusion,
  title={A Latent Space of Stochastic Diffusion Models for Zero-Shot Image Editing and Guidance},
  author={Chen Henry Wu and Fernando De la Torre},
  booktitle={ICCV},
  year={2023},
}
```

## 社会的影響について

_（原文のまま保持：本節は原著者による記述のため省略）_

## ライセンス

X11 ライセンスを使用しています。このライセンスは MIT ライセンスと同一ですが、著作権者の名称（本リポジトリの場合は Carnegie Mellon University）を書面による許可なく広告や宣伝目的で使用することを禁じる一文が追加されています。

## 連絡先

コードに関するご質問は [Issues](https://github.com/ChenWu98/cycle-diffusion/issues) へお寄せください。
手法についての議論は [Chen Henry Wu](https://chenwu.io/) までご連絡ください。

<a href="https://chenwu.io/"><img src="https://avatars.githubusercontent.com/u/28187501?v=4"  width="50" /></a>
