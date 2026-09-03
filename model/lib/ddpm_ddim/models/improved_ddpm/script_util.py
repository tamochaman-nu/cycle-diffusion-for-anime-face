from .unet import UNetModel

NUM_CLASSES = 1000

AFHQ_DICT = dict(
    attention_resolutions="16",
    class_cond=False,
    dropout=0.0,
    image_size=256,
    learn_sigma=True,
    num_channels=128,
    num_head_channels=64,
    num_res_blocks=1,
    resblock_updown=True,
    use_fp16=False,
    use_scale_shift_norm=True,
    num_heads=4,
    num_heads_upsample=-1,
    channel_mult="",
    use_checkpoint=False,
    use_new_attention_order=False,
)


# Matches a pair of self-trained openai/improved-diffusion 256x256 unconditional
# face checkpoints (e.g. ffhq/anime) whose state_dict does NOT match AFHQ_DICT:
# num_res_blocks=2 (not 1), attention at both 16x16 and 8x8 (not just 16x16), and
# plain conv up/down-sampling instead of resblock_updown. Reverse-engineered from
# the checkpoint's state_dict shapes (input_blocks.{3,6,9,12,15}.0.op.* are plain
# Downsample convs, no h_upd/x_upd keys anywhere -> resblock_updown=False; each
# channel_mult stage has exactly 2 residual blocks before its downsample; emb_layers
# output is 2x the block's out_channels -> use_scale_shift_norm=True; out channels
# = 6 = 2*3 -> learn_sigma=True).
CUSTOM_FACE256_DICT = dict(
    attention_resolutions="16,8",
    class_cond=False,
    dropout=0.0,
    image_size=256,
    learn_sigma=True,
    num_channels=128,
    num_head_channels=64,
    num_res_blocks=2,
    resblock_updown=False,
    use_fp16=False,
    use_scale_shift_norm=True,
    num_heads=4,
    num_heads_upsample=-1,
    channel_mult="",
    use_checkpoint=False,
    use_new_attention_order=False,
)


IMAGENET_DICT = dict(
    attention_resolutions="32,16,8",
    class_cond=True,
    image_size=512,
    learn_sigma=True,
    num_channels=256,
    num_head_channels=64,
    num_res_blocks=2,
    resblock_updown=True,
    use_fp16=False,
    use_scale_shift_norm=True,
    dropout=0.0,
    num_heads=4,
    num_heads_upsample=-1,
    channel_mult="",
    use_checkpoint=False,
    use_new_attention_order=False,
)


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
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
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def i_DDPM(dataset_name = 'AFHQ'):
    if dataset_name in  ['AFHQ', 'FFHQ']:
        return create_model(**AFHQ_DICT)
    elif dataset_name == 'IMAGENET':
        return create_model(**IMAGENET_DICT)
    elif dataset_name == 'CUSTOM_FACE256':
        return create_model(**CUSTOM_FACE256_DICT)
    else:
        print('Not implemented.')
        exit()
