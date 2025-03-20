import argparse
from src.train_char_model import train_char_model
from config import *


def main():
    parser = argparse.ArgumentParser(description="Train sentence completeness model.")
    parser.add_argument('--mode', type=str, choices=['word', 'token'], required=True,
                        help='Choose model training mode: "word" for whole-word model or "token" for token-wise model.')
    parser.add_argument('--device', type=str, default=device, help='device to run on')
    parser.add_argument('--epochs', type=int, default=epochs, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=batch_size, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=learning_rate, help='Learning rate for the optimizer')
    parser.add_argument('--checkpoint', type=str, default=checkpoint_path, help='Existing checkpoint path \
                        for model loading')
    parser.add_argument('--use-checkpoint', action='store_true', help='Use existing checkpoint if available')

    args = parser.parse_args()

    train_char_model(args.mode, args.device, args.epochs, args.batch_size, args.lr, args.checkpoint, args.use_checkpoint)


if __name__ == '__main__':
    main()