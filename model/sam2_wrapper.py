import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from sam2_repo.sam2.build_sam import build_sam2
from sam2_repo.sam2.utils.transforms import SAM2Transforms

from model.modules.eff_memo_bank import eff_memo_bank
from model.modules.anatomy import AnatomyPriorModule

# Tested, works as expected
class SAM2Wrapper(nn.Module):
    def __init__(self, config_file, ckpt_path=None, device="cuda"):
        super().__init__()
        self.config_file = config_file
        self.config = self._load_config()

        self.ckpt_path = self.config.ckpt_path
        self.device = self.config.device

        # need to set this when finish lora
        self.lora_hook = None

        # build SAM2 model
        self.model = build_sam2(
            config_file=config_file,
            ckpt_path=self.ckpt_path,
            device=self.device,
            mode="train"
        )
        
        # initialize preprocessing transforms
        self.transforms = SAM2Transforms(
            resolution=self.config.image_size,
            mask_threshold=0.0
        )

    def _load_config(self):
        config_path = os.path.join("configs", f"{self.config_file}.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return OmegaConf.load(config_path)


    def forward(self, x, **kwargs):
        # not to process input during training
        if not self.training:
            x = self.preprocess_input(x)

        x = x.to(self.device)
        # image or video path
        if hasattr(x, 'flat_img_batch'):  # video
            output = self.model.forward(x)
        else:
            output = self.model.forward_image(x)

        # Hook for LORA #
        if self.lora_hook is not None:
            try:
                # Optional: hook can modify model or output
                output = self.lora_hook(self.model, output)
                print("[LoRA] Successfully applied LoRA hook.")
            except Exception as e:
                print(f"[LoRA] Failed to apply LoRA hook: {e}")
        else:
            print("[LoRA] No LoRA hook registered. Skipping.")




        # Placeholder for anatomy prior module
        # recompute feat_sizes on the sky
        feat_sizes = [ (f.shape[-2], f.shape[-1]) for f in output["backbone_fpn"] ]
        output["feat_sizes"] = feat_sizes        

        try:
            # Make sure required keys exist in model output
            required_keys = {"backbone_fpn", "vision_pos_enc", "feat_sizes"}
            print("[Debug] Output keys:", output.keys())
            print("[Debug] Required keys:", required_keys)  
            if all(k in output for k in required_keys):
                anatomy_inputs = {
                    "backbone_fpn": output["backbone_fpn"],
                    "vision_pos_enc": output["vision_pos_enc"],
                    "feat_sizes": output["feat_sizes"]
                }
                enhanced_feats, pos_enc, sizes = self.anatomy_prior(anatomy_inputs)
                # need to replace features in output for memory hook
                output["backbone_fpn"] = enhanced_feats
                output["vision_pos_enc"] = pos_enc
                output["feat_sizes"] = sizes

                # check if the module is working correctly
                print(f"[Success] AnatomyPriorModule applied to {len(enhanced_feats)} feature levels.")
        except Exception as e:
            print(f"[Warning] Anatomy prior hook failed: {e}")



        # Placeholder for memory bank (need to be fixed/adapted)
        if hasattr(self.model, "mask_decoder") and hasattr(self.model.mask_decoder, "memory_bank"):
            #debug print (delete in production)
            print("[Debug] Memory bank hook triggered.")
            try:
                # Access current memory bank
                memory_bank = self.model.mask_decoder.memory_bank
                current_bank = {
                    "maskmem_features":    memory_bank.maskmem_features,
                    "maskmem_pos_enc":     memory_bank.maskmem_pos_enc,
                    "pred_masks":          output.get("pred_masks_high_res"),
                    "obj_ptr":             output.get("obj_ptr"),
                    "object_score_logits": output.get("object_score_logits"),
                    # Keep whatever frame_indices SAM2 has already stored
                    "frame_indices":       getattr(memory_bank, "frame_indices", None),
                }

                # Get vision features and masks from model output
                vision_feats = output.get("vision_feats", None)
                pred_masks = output.get("pred_masks_high_res", None)

                #debug print (delete in production)
                if vision_feats is None:
                    print("[Debug] vision_feats is None.")
                if pred_masks is None:
                    print("[Debug] pred_masks is None.")

                if vision_feats is not None and pred_masks is not None:
                    updated_bank = self.eff_memo_bank(
                        vision_feats=vision_feats,
                        masks=pred_masks,
                        existing_memory=current_bank,
                        frame_idx=kwargs.get("frame_idx", 0)
                    )

                    # Update the memory bank
                    memory_bank.maskmem_features = updated_bank.get("maskmem_features", memory_bank.maskmem_features)
                    memory_bank.maskmem_pos_enc = updated_bank.get("maskmem_pos_enc", memory_bank.maskmem_pos_enc)
                    if "frame_indices" in updated_bank:
                        memory_bank.frame_indices = updated_bank["frame_indices"]
                    
                    # check if successfully updated
                    print(f"[Success] Memory bank updated at frame {kwargs.get('frame_idx', 0)}.")

            except Exception as e:
                print(f"[Warning] Memory bank update skipped: {e}")


        return self.format_output(output)

    def preprocess_input(self, x):
        """Standardize input before feeding into SAM2"""
        # handle batch: (B, C, H, W) → list of (C, H, W)
        if isinstance(x, torch.Tensor) and x.ndim == 4:
            x_list = [img.detach().cpu().numpy().transpose(1, 2, 0) for img in x]
            return self.transforms.forward_batch(x_list)

        # handle single image: (C, H, W)
        if isinstance(x, torch.Tensor) and x.ndim == 3:
            x = x.detach().cpu().numpy().transpose(1, 2, 0)

        return self.transforms(x)

    def format_output(self, model_output, target_format="dict"):
        """Standardize output for downstream processing"""
        if target_format == "dict":
            return {
                'pred_masks': model_output.get('pred_masks'),
                'pred_masks_high_res': model_output.get('pred_masks_high_res'),
                'obj_ptr': model_output.get('obj_ptr')
            }
        return model_output



    ######### Hooks for Custom Modules #########

    def eff_memo_bank(self, vision_feats, masks=None, existing_memory=None, frame_idx=0, **kwargs):
        """
        Hook to override default memory strategy with a custom one.
        
        Parameters:
            vision_feats: Tensor, shape (HW, B, C)
            masks: Tensor, shape varies, typically (B, H, W)
            existing_memory: dict, includes 'maskmem_features', 'maskmem_pos_enc'
            frame_idx: int, optional tracking index for video

        Returns:
            dict with updated memory state
        """
        return eff_memo_bank(
            vision_feats=vision_feats,
            masks=masks,
            existing_memory=existing_memory,
            frame_idx=frame_idx,
            **kwargs
        )

    def anatomy_prior(self, features_dict, **kwargs):
        """
        Hook for anatomy-aware attention module.

        Args:
            features_dict (dict): contains keys "backbone_fpn", "vision_pos_enc", "feat_sizes"

        Returns:
            Tuple: (List[enhanced features], vision_pos_enc, feat_sizes)
        """
        if not hasattr(self, "anatomy_module"):
            self.anatomy_module = AnatomyPriorModule(
                hidden_dim=self.config.hidden_dim,
                num_structures=3,
                num_feature_levels=len(features_dict["backbone_fpn"]),
                prior_shape=(32, 32)
            ).to(self.device)

        return self.anatomy_module(features_dict)

        def register_lora_hook(self, lora_hook):
            """
            Register a LoRA module or patching function.
            It should either modify self.model or return modified outputs.
            """
            self.lora_hook = lora_hook
            print("[LoRA] LoRA hook successfully registered.")


        def get_config(self):
            """Expose config if needed for logging or analysis"""
            return self.config
