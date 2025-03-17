import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import chance_to_remove_end, base_checkpoint
from transformers import AutoTokenizer


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


class EosDatasetToken(Dataset):
    def __init__(self, csv_file, device):
        df = pd.read_csv(csv_file)
        self.device = device
        self.sentence = df["sentence"].tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        sentence = self.sentence[idx]
        label = 1  # Default label for full sentence
        # Truncate the sentence
        if random.random() < 0.5:
            tokens = self.tokenizer.tokenize(sentence)
            tokens_len = len(tokens)
            if tokens_len > 2:  # Ensure at least one token remains
                num_tokens_to_remove = random.randint(1, tokens_len - 2)
                tokens = tokens[:-num_tokens_to_remove]
                sentence = self.tokenizer.convert_tokens_to_string(tokens)
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
