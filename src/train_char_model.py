import os

import random
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

batch_size = 24
learning_rate = 3e-4
epochs = 2
checkpoint = "HuggingFaceTB/SmolLM2-360M"
checkpoint_path = "checkpoints/checkpoint_epoch_2.pth"
device = "mps" if torch.backends.mps.is_available() else "cpu"
chance_to_remove_end = 0.8
use_checkpoint = False


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
class SmolLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.base_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
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
        # self.model.config.output_hidden_states = True

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


if __name__ == "__main__":
    # Load dataset
    train_dataset = EosDataset("../data/train_split.csv")
    test_dataset = EosDataset("../data/test_split.csv")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

    model = SmolLM().to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    start_epoch = 0
    # Load checkpoint if available
    if use_checkpoint and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    wandb.login(key="fcea9af553bd3f3956049026094c5146e1b87b07")
    wandb.init(project="smollm2-finetuning", tags=["AdamW", "LoRA", "LastRealToken"])
    wandb.config.update({"batch_size": batch_size, "learning_rate": learning_rate, "epochs": epochs})


    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training", leave=False)
        for idx, batch in enumerate(train_bar):
            sentences = batch["sentence"]
            label = batch["eos_label"].to(device)
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
            logits = model(inputs)
            loss = criterion(logits, label)
            running_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.set_postfix(train_loss=loss.item())
            if (idx + 1) % 10 == 0:
                # predictions = torch.argmax(logits, dim=1)
                probs = torch.sigmoid(logits)
                predictions = (probs > 0.5).float()
                correct = (predictions == label).sum().item()
                accuracy = 100 * correct / label.size(0)
                wandb.log({"Train Step Loss": loss.item(), "Train Step Accuracy": accuracy})

        # Validation / Testing after each epoch
        model.eval()
        total = 0
        correct = 0
        val_loss = 0
        with torch.no_grad():
            val_bar = tqdm(test_dataloader, desc=f"Epoch {epoch+1} Validation", leave=False)
            for batch in val_bar:
                sentences = batch["sentence"]
                labels = batch["eos_label"].to(device)
                inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                # probabilities = torch.softmax(logits, dim=1)
                # predictions = torch.argmax(probabilities, dim=1)
                probs = torch.sigmoid(logits)
                predictions = (probs > 0.5).float()
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

                # Update tqdm with the current running average validation loss
                avg_val_loss = val_loss / (idx + 1)
                val_bar.set_postfix(val_loss=avg_val_loss)
        accuracy = 100 * correct / total
        wandb.log({"Validation Accuracy": accuracy, "Validation Loss": val_loss / len(test_dataloader)})
        print(f"Validation Accuracy after epoch {epoch + 1}: {accuracy:.2f}%")

        # Save checkpoint after each epoch
        checkpoint_path = f"./checkpoints/hard/checkpoint_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': running_loss,
            'validation_loss': val_loss / len(test_dataloader),
        }, checkpoint_path)

    print("Finetuning complete!")
    wandb.finish()
