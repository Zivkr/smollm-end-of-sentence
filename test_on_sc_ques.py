import json
import random
import lovely_tensors as lt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

lt.monkey_patch()
base_checkpoint = "HuggingFaceTB/SmolLM2-360M"
checkpoint_path = "checkpoints/hard/checkpoint_epoch_2.pth"
device = "mps" if torch.backends.mps.is_available() else "cpu"
chance_to_remove_end = 0.8
batch_size = 24

processed_data = []


def create_data():
    with open("data/SC-Ques/test.jsons", "r") as f:
        for line in f.readlines():
            dict_line = json.loads(line)
            split_sentence = dict_line["stem"].split("___")
            start = split_sentence[0].rstrip()
            end = split_sentence[1].lstrip()
            # Uncomplete sentence
            if start != '':
                processed_data.append((start, 0))
            # Complete sentence
            full_sentence = dict_line["choice_dict"]["choice_dict"][dict_line["answer"]]
            # if full_sentence[-1] in ".!?":
            #     full_sentence = full_sentence[:-1]
            processed_data.append([full_sentence, 1])

    processed_data_df = pd.DataFrame(processed_data, columns=["sentence", "eos_label"])
    processed_data_df.to_csv("data/SC-Ques/processed_test.csv", index=False)


class EosDataset_SC(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.sentence = df["sentence"].tolist()
        self.labels = df["eos_label"].tolist()

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        sentence = self.sentence[idx]
        label = self.labels[idx]

        return {
            'sentence': sentence,
            'eos_label': torch.tensor(label, dtype=torch.float32)
        }


class SmolLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)
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
        # self.model.config.output_hidden_states = True

    def forward(self, x):
        # out = self.base_model(x["input_ids"])
        # logits = out.logits[:, -1, :]  # Select the last token's logits
        # logits = self.classifier(logits)
        # return logits.squeeze(-1)
        # Ensure that the input dictionary contains both "input_ids" and "attention_mask"
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
    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    test_dataset = EosDataset_SC("data/SC-Ques/processed_test_no_punct.csv")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    eos_tokens = ["<|endoftext|>", ".", "?", "!", ";", "\n", "\n\n", "\n\n\n", "\n\n\n\n", ".\"", "\xa0", ".”", ".)",
                  ".,", ".\\\\", ".;", "\n ", "\n  ", "\n  ", "\n\n  ", "\n\n   ", "\n\n    "]
    eos_ids = set()
    for tok in eos_tokens:
        # eos_ids.append(tokenizer.encode(text=tok)[0])
        eos_ids.add(tokenizer.encode(text=tok)[0])
    eos_ids = list(eos_ids)

    model = SmolLM().to(device)
    original_model = AutoModelForCausalLM.from_pretrained(base_checkpoint).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    probabilities_list = []
    labels_list = []

    total = len(test_dataset)
    all_probs = []
    all_probs_original = []
    for batch in tqdm(test_dataloader, desc="Processing batches"):
        sentences = batch["sentence"]
        labels = batch["eos_label"].to(device)
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            logits = model(inputs)
            logits_original = original_model(**inputs).logits

        # Original
        last_token_indices = attention_mask.sum(dim=1) - 1
        batch_size_actual = logits_original.shape[0]
        batch_indices = torch.arange(batch_size_actual, device=device)
        last_token_logits = logits_original[batch_indices, last_token_indices, :]
        # Compute softmax to get probabilities
        probabilities_original = F.softmax(last_token_logits, dim=-1)
        eos_probs = probabilities_original[:, eos_ids].sum(dim=-1)

        # Fine-tuned
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy())
        all_probs_original.extend(eos_probs.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())
    full_probs = torch.tensor(all_probs, device=device)
    full_probs_original = torch.tensor(all_probs_original, device=device)
    labels_tensor = torch.tensor(labels_list, device=device)

    thresholds = [0.5, 0.7, 0.9]
    accuracies = {}
    for threshold in thresholds:
        correct = 0
        predictions = (full_probs > threshold).float()
        correct = (predictions == labels_tensor).sum().item()
        print(f"Accuracy@{int(threshold * 100)}: {((correct / total) * 100.0):.2f}%")
    for threshold in thresholds:
        correct = 0
        predictions = (full_probs_original > threshold).float()
        correct = (predictions == labels_tensor).sum().item()
        print(f"Original Accuracy@{int(threshold * 100)}: {((correct / total) * 100.0):.2f}%")
