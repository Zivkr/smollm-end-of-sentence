import pytorch_lightning as pl
import torch
from peft import LoraConfig, get_peft_model
from torch import nn as nn
from torchmetrics import Accuracy
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import base_checkpoint, device, criterion


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
            lora_dropout=0.0,
            bias="none",
            use_dora=True
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        self.save_hyperparameters()
        self.val_accuracy = Accuracy(task="binary")

    def forward(self, x):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]

        out = self.base_model(input_ids, attention_mask=attention_mask)
        logits = out.logits  # shape: (batch_size, seq_len, hidden_dim)

        # Calculate the index of the last non-padding token for each sequence
        last_token_indices = attention_mask.sum(dim=1) - 1
        real_batch_size = logits.size(0)
        batch_indices = torch.arange(real_batch_size, device=device)

        # Select logits corresponding to the last non-padding token
        last_logits = logits[batch_indices, last_token_indices, :]

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
        acc = self.val_accuracy.compute()
        self.log('Validation Accuracy', acc, prog_bar=True)
        self.val_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        return optimizer
