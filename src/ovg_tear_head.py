"""The frozen-feature tear heads for module two."""
import torch
from torch import nn

from .ovg_connected_depth import DIRECTIONS, edge_slices

VARIANTS = {"d_full": 75, "d_nodepth": 71}


class TearHead(nn.Sequential):
    def __init__(self, channels, hidden=16):
        super().__init__(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 4, 1),
        )


def tear_features(feature, rgb, inverse_depth):
    """Pack full-frame OVG inputs. End-of-image edge slots are masked by the caller."""
    batch, channels, height, width = feature.shape
    if channels != 64 or rgb.shape != (batch, 3, height, width):
        raise ValueError("Module two requires native 64-channel OVG features and RGB")
    color_difference = rgb.new_zeros(batch, 4, height, width)
    log_difference = torch.zeros_like(color_difference)
    log_depth = inverse_depth.log()
    for i, (dy, dx) in enumerate(DIRECTIONS):
        a, b = edge_slices(dy, dx)
        aa, bb = (Ellipsis, *a), (Ellipsis, *b)
        color_difference[:, i][aa] = (rgb[aa] - rgb[bb]).abs().mean(1)
        log_difference[:, i][aa] = (log_depth[aa] - log_depth[bb]).abs()
    return torch.cat((feature, rgb, color_difference, log_difference), 1)
