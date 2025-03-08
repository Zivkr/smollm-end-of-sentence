import os
import random
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from lightning.pytorch.loggers import WandbLogger
from dotenv import load_dotenv
from torchmetrics import Accuracy
import wandb


load_dotenv()
batch_size = 24
learning_rate = 3e-4
epochs = 1
base_checkpoint = "HuggingFaceTB/SmolLM2-360M"
checkpoint_path = "checkpoints/checkpoint_epoch_2.pth"
device = "mps" if torch.backends.mps.is_available() else "cpu"
chance_to_remove_end = 0.8
use_checkpoint = False
criterion = nn.BCEWithLogitsLoss()


class EosDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.sentence = df["sentence"].tolist()

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        sentence = self.sentence[idx]
        label = 1  # Default label for full sentence
        # Truncate the sentence
        if random.random() < 0.5:
            words = sentence.split()
            if len(words) > 2:  # Ensure at least one word remains
                num_words_to_remove = random.randint(1, len(words) - 2)
                sentence = " ".join(words[:-num_words_to_remove])
                label = 0
        else:
            # Don't truncate the sentence
            random_chance = random.random()
            # delete the symbol at the end of the sentence 80% of the time
            if sentence[-1] in ".!?" and random_chance < chance_to_remove_end:
                sentence = sentence[:-1]

        return {
            'sentence': sentence,
            'eos_label': torch.tensor(label, dtype=torch.float32)
        }


# Pytorch Module
class SmolLM(pl.LightningModule):
    def __init__(self, learning_rate=3e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = AutoModelForCausalLM.from_pretrained(base_checkpoint).to(device)
        self.base_model.lm_head = nn.Identity()
        self.classifier = nn.Sequential(
            # nn.Linear(self.base_model.lm_head.out_features, 1024),
            nn.Linear(960, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Freeze smollm2 parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        # LoRA fine-tuning
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            # Target modules for LoRA
            lora_dropout=0.0,
            bias="none",
            use_dora=True
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        self.base_model.print_trainable_parameters()
        self.save_hyperparameters()
        self.val_accuracy = Accuracy(task="binary")

    def forward(self, x):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]

        # Forward pass through the base model using the attention mask
        out = self.base_model(input_ids, attention_mask=attention_mask)
        logits = out.logits  # shape: (batch_size, seq_len, hidden_dim)

        # Calculate the index of the last non-padding token for each sequence
        last_token_indices = attention_mask.sum(dim=1) - 1  # shape: (batch_size)
        real_batch_size = logits.size(0)
        batch_indices = torch.arange(real_batch_size, device=device)

        # Select logits corresponding to the last non-padding token
        last_logits = logits[batch_indices, last_token_indices, :]  # shape: (batch_size, hidden_dim)

        # Pass the selected logits through the classifier
        output_logits = self.classifier(last_logits)
        return output_logits.squeeze(-1)

    def training_step(self, batch, batch_idx):
        sentences = batch["sentence"]
        labels = batch["eos_label"].to(device)
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
        logits = self(inputs)
        loss = self.criterion(logits, labels)
        self.log('Train Step Loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sentences = batch["sentence"]
        labels = batch["eos_label"].to(device)
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
        logits = self(inputs)
        loss = self.criterion(logits, labels)
        preds = (torch.sigmoid(logits) > 0.5).long()
        self.val_accuracy.update(preds, labels.long())
        self.log('Validation Step Loss', loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Compute and log the overall validation accuracy
        acc = self.val_accuracy.compute()
        self.log('Validation Accuracy', acc, prog_bar=True)
        self.val_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
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
