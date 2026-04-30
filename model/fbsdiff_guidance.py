import torch


def frequency_band_substitution(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    cutoff_ratio: float = 0.3,
    mode: str = 'low_from_source',
) -> torch.Tensor:
    """
    FBSDiff: ソース画像の低周波成分をターゲット生成の高周波と合成する。

    source_features : inversion 時に保存した diffusion latent (構造情報)
    target_features : 生成時の現在の latent
    cutoff_ratio    : 低/高周波の境界 (0.0-1.0、小さいほど低周波成分が少ない)
    mode            : 'low_from_source'  -> ソースの低周波 + ターゲットの高周波
                      'high_from_source' -> ソースの高周波 + ターゲットの低周波
    """
    src_fft = torch.fft.fft2(source_features.float())
    tgt_fft = torch.fft.fft2(target_features.float())

    H, W = source_features.shape[-2:]
    h_cut = max(1, int(H * cutoff_ratio))
    w_cut = max(1, int(W * cutoff_ratio))

    # 低周波マスク: 四隅（DC 成分周辺）を 1 にする
    mask = torch.zeros(H, W, device=source_features.device, dtype=torch.float)
    mask[:h_cut, :w_cut] = 1.0
    mask[-h_cut:, :w_cut] = 1.0
    mask[:h_cut, -w_cut:] = 1.0
    mask[-h_cut:, -w_cut:] = 1.0
    # broadcast to (B, C, H, W)
    mask = mask.unsqueeze(0).unsqueeze(0)

    if mode == 'low_from_source':
        blended_fft = mask * src_fft + (1.0 - mask) * tgt_fft
    else:
        blended_fft = (1.0 - mask) * src_fft + mask * tgt_fft

    result = torch.fft.ifft2(blended_fft).real
    return result.to(target_features.dtype)
