from anatomy import *
import torch
import pytest


def test_sub_forward():
    x = torch.randn(2, 256, 32, 32)  # (B, C, H, W)
    output = self_update_block(x)
    assert output.shape == (2, 256, 32, 32), "SUB output shape mismatch when C == hidden_dims"


def test_sub_diff_c_forward():
    x = torch.randn(2, 32, 32, 32)
    output = self_update_block(x)
    assert output.shape == (2, 256, 32, 32), "SUB output shape mismatch when C != hidden_dims"


def test_cub_forward():
    prior, features = torch.randn(2, 256, 32, 32), torch.randn(2, 256, 32, 32)
    output = cross_update_block(prior, features)
    assert output.shape == (2, 256, 32, 32), "CUB output shape mismatch when C == hidden_dims"


def test_cub_diff_c_forward():
    prior, features = torch.randn(2, 128, 32, 32), torch.randn(2, 128, 32, 32)
    output = cross_update_block(prior, features)
    assert output.shape == (2, 256, 32, 32), "CUB output shape mismatch when C != hidden_dims"


def test_ap_block_forward():
    anatomy_prior, image_features = torch.randn(2, 256, 32, 32), torch.randn(2, 256, 32, 32)
    output = ap_block(anatomy_prior, image_features)
    assert output[0].shape == (2, 256, 32, 32) and output[1].shape == (2, 256, 32, 32), "APblock failure"


def test_ap_block_diff_c_forward():
    anatomy_prior, image_features = torch.randn(2, 64, 32, 32), torch.randn(2, 64, 32, 32)
    output = ap_block(anatomy_prior, image_features)
    assert output[0].shape == (2, 256, 32, 32) and output[1].shape == (2, 256, 32, 32), "APblock failure"


def test_ap_module_forward():
    B = 2
    C = 128
    backbone_fpn = [
        torch.randn(B, C, 64, 64),
        torch.randn(B, C, 32, 32),
        torch.randn(B, C, 16, 16),
    ]
    vision_pos_enc = [
        torch.randn(B, C, 64, 64),
        torch.randn(B, C, 32, 32),
        torch.randn(B, C, 16, 16),
    ]
    feat_sizes = [(64, 64), (32, 32), (16, 16)]
    features_dict = {"backbone_fpn": backbone_fpn, "vision_pos_enc": vision_pos_enc, "feat_sizes": feat_sizes}

    enhanced_feats, vision_pos_enc, feat_sizes = ap_module(features_dict)
    # print([enhanced_feats[_].shape for _ in range(len(enhanced_feats))])
    assert [backbone_fpn[_].shape == enhanced_feats[_].shape for _ in range(len(backbone_fpn))], "APmodule failure"


self_update_block = SelfUpdateBlock(hidden_dim=256)
cross_update_block = CrossUpdateBlock(hidden_dim=256)
ap_block = APblock(hidden_dim=256)
ap_module = AnatomyPriorModule(hidden_dim=256)

pytest.main([__file__])
