# CardSeg

## Project Structure


```
CardSeg/
├── configs/        # YAML configuration files
│ └── sam2_train_config.yaml
├── lora/           # LoRA for fine tune
│ └── lora.py
├── model/          # SAM2 wrapper and main modules
│ ├── sam2_wrapper.py
│ └── modules/
│ ├── anatomy.py
│ └── eff_memo_bank.py
├── run/             # Script entry points
│ └── main.py
├── training/        # Training logic, loss functions, dataloaders
│ ├── train.py
│ ├── loss.py
│ └── dataloader.py
├── test/ 
├── utils/             # Utility scripts (metrics, logging, etc.)
│ ├── logger.py
│ └── metric.py
└── sam2_repo/          # Cloned Meta SAM2 repository
```

## Initialize

Install the package and download pretrained checkpoints:

```bash
cd sam2_repo
pip install -e .
cd checkpoints && ./download_ckpts.sh && cd ..
```

## Configuration

I currently use sam2.1_hiera_tiny.pt,To ensure compatibility:

- Keep original SAM2 tiny configuration structure unchanged.

- Modify the configuration above the division line of tiny model config.


## Run

Run:
```bash
cd CardSeg
"python run/main.py" 
```

This will initialize the model based on the Hydra configuration and run a forward pass using dummy data.





