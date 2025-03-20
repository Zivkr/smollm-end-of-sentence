import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from dotenv import load_dotenv
from src.dataset import EosDataset
from src.model import SmolLM
import wandb


def train_linear_model(device, epochs, batch_size, learning_rate, checkpoint_path, use_checkpoint):
    load_dotenv()
    # Load dataset
    train_dataset = EosDataset("data/train_split.csv")
    test_dataset = EosDataset("data/test_split.csv")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,
                                 persistent_workers=True)
    print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

    # start_epoch = 0
    # Load checkpoint if available
    if use_checkpoint and os.path.isfile(checkpoint_path):
        model = SmolLM.load_from_checkpoint(checkpoint_path, map_location=device)
    else:
        model = SmolLM(learning_rate).to(device)

    wandb_logger = WandbLogger(project="smollm2-finetuning", log_model=True, tags=["AdamW", "LoRA", "LastRealToken"])
    wandb_logger.experiment.config.update({"batch_size": batch_size, "learning_rate": learning_rate, "epochs": epochs})

    # Training
    trainer = pl.Trainer(accelerator="auto", max_epochs=epochs, log_every_n_steps=50, logger=wandb_logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
    wandb.finish()
    return