import torch
from sam2_repo.sam2.modeling.sam2_base import SAM2Base
from sam2_repo.sam2.sam2_video_predictor import SAM2VideoPredictor

def eff_memo_bank(
    self,
    vision_feats: torch.Tensor,
    masks: torch.Tensor,
    existing_memory: dict,
    frame_idx: int = 0,
    **kwargs
) -> dict:
    """
    Memory Bank Update Hook

    Replace the default FIFO-based memory update strategy with a custom one.
    This function defines how to insert, update, or replace entries in the memory bank.

    Args:
        vision_feats (torch.Tensor): Shape (HW, B, C)
            Top-level vision features from the encoder.
            C is typically 256 (hidden_dim).
        masks (torch.Tensor): High-resolution binary masks (B, 1, H, W) or (B, H, W)
            Used for mask-guided memory construction.
        existing_memory (dict): Dictionary with current memory content:
            - "maskmem_features": Tensor of shape (B, mem_dim, H', W')
            - "maskmem_pos_enc": Tensor of same shape as features or pos encoding shape
            - "frame_indices": List or Tensor tracking temporal order (optional)

        frame_idx (int): Index of the current frame in sequence (for temporal logic).
        kwargs: Reserved for future compatibility (e.g., attention metadata).

    Returns:
        dict: Updated memory bank containing at least:
            - "maskmem_features": Tensor (B, mem_dim, H', W')
            - "maskmem_pos_enc": Tensor (same shape)
            - (Optional) Additional metadata or buffers for attention

    Notes:
        - Vision features are in 256-dim space; memory features in compressed 64-dim space.
        - You may implement strategies like LRU, temporal decay, or content similarity filtering.
        - Avoid in-place modification unless managed for backpropagation.
    """
    # Debug print to verify that we received the expected keys:
    print(f"[CustomMemory] Called at frame {frame_idx}, keys = {list(existing_memory.keys())}")

    # to make sure the function is being called correctly (for debugging)
    print(f"[CustomMemory] Updating memory at frame {frame_idx}")

    return existing_memory  # No-op placeholder