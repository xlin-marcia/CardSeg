import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from model.sam2_wrapper import SAM2Wrapper  


def main():
    os.chdir(os.path.abspath(".")) 
    config_file = "sam2_train_config"

    # Reset Hydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Initialize model using Hydra config
    with initialize(config_path="../configs", version_base=None):
        model = SAM2Wrapper(config_file=config_file)

    print("Model initialized with config:", model.config_file)
    model.to(model.device)

    # === Register LoRA hook ===
    ## model.register_lora_hook(lora_function)

    # === Dummy Input ===
    dummy_input = torch.randn(1, 3, model.config.image_size, model.config.image_size).to(model.device)

    # === Train Mode Test ===
    model.train()
    print("\n[Main] Running in training mode...")
    try:
        out = model(dummy_input)
        print("[Main] Training forward pass succeeded.")
    except Exception as e:
        print("[Main] Training forward pass failed:", e)

    # === Eval Mode Test ===
    model.eval()
    print("\n[Main] Running in eval mode...")
    try:
        out = model(dummy_input)
        print("[Main] Eval forward pass succeeded.")
        print("[Main] Output keys:", out.keys())
    except Exception as e:
        print("[Main] Eval forward pass failed:", e)


if __name__ == "__main__":
    main()
