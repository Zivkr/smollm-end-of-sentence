import torch
from torch import nn as nn

batch_size = 24
learning_rate = 3e-4
epochs = 1
base_checkpoint = "HuggingFaceTB/SmolLM2-360M"
checkpoint_path = "checkpoints/checkpoint_epoch_2.pth"
device = "mps" if torch.backends.mps.is_available() else "cpu"
chance_to_remove_end = 0.8
use_checkpoint = False
criterion = nn.BCEWithLogitsLoss()
kaggle_data_path = "/kaggle/input/smollm2-everyday-conversations/"
