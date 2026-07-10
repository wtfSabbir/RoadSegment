"""
Shared Model Architecture Module
==================================
Single source of truth for build_model(), imported by BOTH
train_segformer.py and inference.py. This guarantees training and
inference always construct the identical architecture, eliminating
the class of bugs where the two scripts could silently drift apart.

Do not duplicate build_model() anywhere else. If you need to change
the architecture, change it here only.
"""

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation


def build_model(model_size='b2', num_channels=6, num_labels=3):
    """
    Load pretrained SegFormer and modify the first conv layer to
    accept `num_channels` input channels instead of 3.

    Channel convention (must match prepare_dataset.py / rasterization):
      0: height_above_ground
      1: verticality
      2: intensity
      3: red
      4: green
      5: blue

    Strategy:
      - Copy pretrained RGB weights into positions 3:6 (channels 3,4,5)
      - Randomly initialize positions 0:3 (HAG, verticality, intensity),
        scaled to match the magnitude of the pretrained RGB weights
    """
    model_name = f"nvidia/segformer-{model_size}-finetuned-ade-512-512"
    print(f"Loading pretrained: {model_name}")

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    )

    first_proj = model.segformer.encoder.patch_embeddings[0].proj
    old_weight = first_proj.weight.data  # shape (out_ch, 3, k, k)
    out_ch, in_ch_old, kh, kw = old_weight.shape

    print(f"  Original first conv: {old_weight.shape}")

    new_conv = nn.Conv2d(
        num_channels, out_ch,
        kernel_size=first_proj.kernel_size,
        stride=first_proj.stride,
        padding=first_proj.padding,
    )

    with torch.no_grad():
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

        # Channels 3,4,5 = R,G,B → copy pretrained weights exactly
        new_conv.weight[:, 3:6, :, :] = old_weight

        # Channels 0,1,2 = HAG, verticality, intensity → scaled random init
        rgb_std = old_weight.std().item()
        new_conv.weight[:, 0:3, :, :] *= (
            rgb_std / new_conv.weight[:, 0:3, :, :].std().item())

        if first_proj.bias is not None:
            new_conv.bias[:] = first_proj.bias

    model.segformer.encoder.patch_embeddings[0].proj = new_conv
    print(f"  New first conv: {new_conv.weight.shape}")
    print(f"  Channels 0-2 (HAG, verticality, intensity): randomly initialized")
    print(f"  Channels 3-5 (R, G, B): pretrained weights copied")

    return model


# Channel order constant — shared so rasterization, training, and
# inference all agree on what each channel index means.
CHANNEL_ORDER = ["height_above_ground", "verticality", "intensity", "red", "green", "blue"]
