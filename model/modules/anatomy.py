# Partially adapted from https://github.com/AlexYouXin/Explicit-Shape-Priors

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfUpdateBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self._dynamic_proj = None

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        if C != self.hidden_dim:
            if self._dynamic_proj is None or self._dynamic_proj.in_channels != C:
                self._dynamic_proj = nn.Conv2d(C, self.hidden_dim, kernel_size=1).to(x.device)
            x = self._dynamic_proj(x)

        x_flat = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        out = self.norm(attn_out + x_flat).transpose(1, 2).view(B, C, H, W)
        return out


class CrossUpdateBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self._proj_prior = None
        self._proj_feat = None
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, prior, features):
        # prior: (B, C, H, W); features: (B, C, H, W)
        B, C_p, H, W = prior.shape
        _, C_f, _, _ = features.shape

        if C_p != self.hidden_dim:
            if self._proj_prior is None or self._proj_prior.in_channels != C_p:
                self._proj_prior = nn.Conv2d(C_p, self.hidden_dim, kernel_size=1).to(prior.device)
            prior = self._proj_prior(prior)

        if C_f != self.hidden_dim:
            if self._proj_feat is None or self._proj_feat.in_channels != C_f:
                self._proj_feat = nn.Conv2d(C_f, self.hidden_dim, kernel_size=1).to(features.device)
            features = self._proj_feat(features)

        prior_flat = prior.flatten(2).transpose(1, 2)  # (B, HW, C)
        feat_flat = features.flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(prior_flat, feat_flat, feat_flat)
        out = self.norm(attn_out + prior_flat).transpose(1, 2).view(B, self.hidden_dim, H, W)
        return out


class APblock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.sub = SelfUpdateBlock(hidden_dim)
        self.cub = CrossUpdateBlock(hidden_dim)

    def forward(self, anatomy_prior, image_features):
        refined_prior = self.sub(anatomy_prior)
        enhanced_features = self.cub(refined_prior, image_features)
        return refined_prior, enhanced_features

class AnatomyPriorModule(nn.Module):
    def __init__(self, hidden_dim, num_structures=3, num_feature_levels=3, prior_shape=(32, 32)):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_structures = num_structures
        self.num_feature_levels = num_feature_levels

        # Learnable priors for anatomical regions: shape (num_structures, C, H, W)
        self.anatomy_priors = nn.Parameter(
            torch.zeros(num_structures, hidden_dim, *prior_shape)
        )
        nn.init.xavier_uniform_(self.anatomy_priors)

        # One AnatomyPrior module per feature level
        self.anatomy_modules = nn.ModuleList([
            APblock(hidden_dim) for _ in range(num_feature_levels)
        ])

    def forward(self, features_dict):
        """
        Args:
            features_dict (dict):
                {
                    "backbone_fpn": List[Tensor],  # multi-scale feature maps, each of shape (B, C, H, W)
                    "vision_pos_enc": Optional[List[Tensor]]
                    "feat_sizes": Optional[List[Tuple[int, int]]]
                }
        Returns:
            Tuple:
                - List[Tensor]: Enhanced multi-scale feature maps
                - vision_pos_enc: unchanged
                - feat_sizes: unchanged
        """
        backbone_feats = features_dict["backbone_fpn"]
        vision_pos_enc = features_dict.get("vision_pos_enc")
        feat_sizes = features_dict.get("feat_sizes")

        B = backbone_feats[0].shape[0]
        enhanced_feats = []


        # protect against OOM by filtering out large feature maps
        max_hw = 128 * 128
        filtered_feats = []
        filtered_modules = []

        for feat, module in zip(backbone_feats, self.anatomy_modules):
            H, W = feat.shape[-2:]
            if H * W <= max_hw:
                filtered_feats.append(feat)
                filtered_modules.append(module)
            else:
                print(f"[Skip] Skipped feature map of size {H}x{W} to avoid OOM.")

        backbone_feats = filtered_feats

        for i, (feat, module) in enumerate(zip(backbone_feats, self.anatomy_modules)):
            # Resize anatomy prior to match current feature size
            resized_prior = F.interpolate(
                self.anatomy_priors, size=feat.shape[-2:], mode='bilinear', align_corners=False
            )  # shape: (num_structures, C, H, W)

            # Average over structures and expand to batch
            mean_prior = resized_prior.mean(dim=0, keepdim=True).repeat(B, 1, 1, 1)

            # Run through anatomy module
            _, enhanced_feat = module(mean_prior, feat)

            # only to check if the module is working correctly, delete in production
            print(f"[AnatomyPriorModule] Applied anatomy prior to feature level {i}, input shape: {feat.shape}, output shape: {enhanced_feat.shape}")
            
            enhanced_feats.append(enhanced_feat)

        return enhanced_feats, vision_pos_enc, feat_sizes