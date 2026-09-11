"""
Diagnostics-only helpers for loading/constructing diffusion configs and models.

History: this module originally reimplemented DDPMDDIMWrapper's model-
construction logic (a "coded copy, generalized to an arbitrary --config/
--dataset-family") because the version of model/gan_wrapper/ddpm_ddim_wrapper.py
this repo shipped with only recognized a fixed allowlist of dataset names
(celeba256, afhqdog256, ffhq256, ...) with no "anime" entry, and had no
generic custom-domain fallback. That workaround is no longer needed:
prepare_ddpm_ddim() (in model/gan_wrapper/ddpm_ddim_wrapper.py) now treats
any unrecognized source_model_type as the name of a
ckpts/ddpm/configs/<name>.yml file (see 'anime'/'ffhq_custom' usage
throughout diagnostics/step_a1_.../step_a2_...), so diagnostics/step_a1 and
step_a2 now import and use DDPMDDIMWrapper directly, unmodified.

What's left here, and why:
  - `dict2namespace`: used by step_b1 to load a diffusion config YAML
    standalone (without constructing a full DDPMDDIMWrapper), for comparing
    two configs against each other.
  - `build_unet_from_args`: used by step_c2/c4/c5/e3/e4 (and
    diagnostics/generic_ddpm_target.py) to construct a UNetModel (the same
    architecture class real checkpoints use) with arbitrary architecture
    hyperparameters, independent of the fixed i_DDPM() presets in
    model/lib/ddpm_ddim/models/improved_ddpm/script_util.py. This is a copy
    of create_model() from that file -- kept as a copy (rather than
    importing that module's create_model directly, which isn't itself
    exported/reused elsewhere) so callers can pass arbitrary small/large
    architecture sizes without needing a matching preset or config YAML.

Note (2026-09-03): this file, along with the rest of diagnostics/ (all
step_a-e scripts and diagnostics/outputs/), was found deleted from the
working tree partway through this session (not a git-tracked deletion --
diagnostics/ was never `git add`-ed, so there is no history to recover from).
This file was reconstructed from the conversation record to unblock
diagnostics/generic_ddpm_target.py and diagnostics/step_e4_translate_
curated_target.py, which depend on it. See the chat for the full report of
what else was lost.
"""
import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.lib.ddpm_ddim.models.improved_ddpm.unet import UNetModel  # noqa: E402


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def build_unet_from_args(args, image_size: int) -> torch.nn.Module:
    """Adapted copy of create_model() from
    model/lib/ddpm_ddim/models/improved_ddpm/script_util.py, parameterized by
    an argparse.Namespace with arch_* fields instead of a fixed preset dict.
    """
    channel_mult = args.arch_channel_mult
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in args.arch_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=args.arch_num_channels,
        out_channels=(3 if not args.arch_learn_sigma else 6),
        num_res_blocks=args.arch_num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=args.arch_dropout,
        channel_mult=channel_mult,
        num_classes=(1000 if args.arch_class_cond else None),
        use_checkpoint=False,
        use_fp16=False,
        num_heads=args.arch_num_heads,
        num_head_channels=args.arch_num_head_channels,
        num_heads_upsample=-1,
        use_scale_shift_norm=args.arch_use_scale_shift_norm,
        resblock_updown=args.arch_resblock_updown,
        use_new_attention_order=False,
    )
