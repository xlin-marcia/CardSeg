import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.modules.anatomy import *
import torch
import pytest


@pytest.fixture
def self_update_block():
    return SelfUpdateBlock(hidden_dim=256)


@pytest.fixture
def cross_update_block():
    return CrossUpdateBlock(hidden_dim=256)


@pytest.fixture
def ap_block():
    return APblock(hidden_dim=256)


@pytest.fixture
def ap_module():
    return AnatomyPriorModule(hidden_dim=256)


@pytest.mark.parametrize("in_channels", [256, 64])
def test_sub_shapes(in_channels, self_update_block):
    x = torch.randn(2, in_channels, 32, 32)  # (B, C, H, W)
    out = self_update_block(x)
    assert out.shape == (2, 256, 32, 32), "SelfUpdateBlock failure"


@pytest.mark.parametrize("in_channels", [256, 64])
def test_cub_shapes(in_channels, cross_update_block):
    prior = torch.randn(2, in_channels, 32, 32)
    features = torch.randn(2, in_channels, 32, 32)
    out = cross_update_block(prior, features)
    assert out.shape == (2, 256, 32, 32), "CrossUpdateBlock failure"


@pytest.mark.parametrize("in_channels", [256, 64])
def test_ap_block_shapes(in_channels, ap_block):
    anatomy_prior = torch.randn(2, in_channels, 32, 32)
    image_features = torch.randn(2, in_channels, 32, 32)
    output = ap_block(anatomy_prior, image_features)
    assert output[0].shape == (2, 256, 32, 32) and output[1].shape == (2, 256, 32, 32), "APblock failure"


@pytest.mark.parametrize("hidden_dim", [256, 64])
def test_ap_module(hidden_dim, ap_module):
    backbone_fpn = [
        torch.randn(2, hidden_dim, 64, 64),
        torch.randn(2, hidden_dim, 32, 32),
        torch.randn(2, hidden_dim, 16, 16),
    ]
    vision_pos_enc = [
        torch.randn(2, hidden_dim, 64, 64),
        torch.randn(2, hidden_dim, 32, 32),
        torch.randn(2, hidden_dim, 16, 16),
    ]
    feat_sizes = [(64, 64), (32, 32), (16, 16)]
    features_dict = {"backbone_fpn": backbone_fpn, "vision_pos_enc": vision_pos_enc, "feat_sizes": feat_sizes}

    enhanced_feats, vision_pos_enc, feat_sizes = ap_module(features_dict)
    # print([enhanced_feats[_].shape for _ in range(len(enhanced_feats))])
    assert [backbone_fpn[_].shape == enhanced_feats[_].shape for _ in range(len(backbone_fpn))], "APmodule failure"


pytest.main([__file__])
