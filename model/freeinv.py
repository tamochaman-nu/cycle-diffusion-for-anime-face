import torch
from typing import Optional, List

TRANSFORMS = ['rot90', 'rot180', 'rot270', 'flip_h', 'flip_v', 'identity']


def apply_transform(x: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == 'rot90':
        return torch.rot90(x, 1, [-2, -1])
    elif transform == 'rot180':
        return torch.rot90(x, 2, [-2, -1])
    elif transform == 'rot270':
        return torch.rot90(x, 3, [-2, -1])
    elif transform == 'flip_h':
        return torch.flip(x, [-1])
    elif transform == 'flip_v':
        return torch.flip(x, [-2])
    else:
        return x


def invert_transform(x: torch.Tensor, transform: str) -> torch.Tensor:
    inv_map = {
        'rot90': 'rot270', 'rot270': 'rot90',
        'rot180': 'rot180', 'flip_h': 'flip_h',
        'flip_v': 'flip_v', 'identity': 'identity',
    }
    return apply_transform(x, inv_map[transform])


class FreeInvDDIMInverter:
    """
    FreeInv: ランダム可逆変換で DDIM inversion/generation の誤差を統計的に低減。

    encode 側: inversion_step() でサンプル・記録しながら T 空間へ変換。
               inversion_step_apply_same() で同じ変換を xt_next にも適用し、
               compute_eps を T 空間で実行することで eps を T 空間に合わせる。
               inversion_step_invert() でステップ後に元空間へ戻す。

    generate 側: 同じシードで generate_sequence() を呼ぶと encode と同一の変換列を再現。
                 generation_step_forward() で T 変換 -> denoising -> generation_step() で T^{-1}。
    """

    def __init__(self, use_freeinv: bool = True, seed: Optional[int] = None):
        self.use_freeinv = use_freeinv
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
        self.transform_sequence: List[str] = []

    def _sample(self) -> str:
        idx = int(torch.randint(len(TRANSFORMS), (1,), generator=self.rng).item())
        return TRANSFORMS[idx]

    def generate_sequence(self, n_steps: int) -> None:
        """generate 側で encode と同一変換列を再現するために事前生成する。"""
        self.transform_sequence = [self._sample() for _ in range(n_steps)]

    # ---- encode 側 ----

    def inversion_step(self, latent: torch.Tensor, step_idx: int) -> torch.Tensor:
        """変換をサンプル・記録し、latent (xt) に適用して返す。"""
        if not self.use_freeinv:
            return latent
        t = self._sample()
        self.transform_sequence.append(t)
        return apply_transform(latent, t)

    def inversion_step_apply_same(self, latent: torch.Tensor) -> torch.Tensor:
        """直前の inversion_step と同じ変換を latent (xt_next) に適用する。"""
        if not self.use_freeinv or not self.transform_sequence:
            return latent
        return apply_transform(latent, self.transform_sequence[-1])

    def inversion_step_invert(self, latent: torch.Tensor) -> torch.Tensor:
        """直前の inversion_step の逆変換を適用し、T 空間から元空間へ戻す。"""
        if not self.use_freeinv or not self.transform_sequence:
            return latent
        return invert_transform(latent, self.transform_sequence[-1])

    # ---- generate 側 ----

    def generation_step_forward(self, latent: torch.Tensor, step_idx: int) -> torch.Tensor:
        """記録済み変換を適用して denoising 前に T 空間へ変換する。"""
        if not self.use_freeinv or step_idx >= len(self.transform_sequence):
            return latent
        return apply_transform(latent, self.transform_sequence[step_idx])

    def generation_step(self, latent: torch.Tensor, step_idx: int) -> torch.Tensor:
        """記録済み変換の逆変換を適用して denoising 後に元空間へ戻す。"""
        if not self.use_freeinv or step_idx >= len(self.transform_sequence):
            return latent
        return invert_transform(latent, self.transform_sequence[step_idx])

    def reset(self) -> None:
        self.transform_sequence = []
