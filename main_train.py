# https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus
import argparse
import os
import random
from typing import Optional

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import LCRRotHopPlusPlus
from utils import EmbeddingsDataset, train_validation_split
from pytorchtools import EarlyStopping

SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stringify_float(value: float):
    return str(value).replace('.', '-')


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2015, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--hops", default=3, type=int, help="Number of hops in rotatory attention")
    parser.add_argument("--lr", default=0.005, type=float, help="Learning rate")
    parser.add_argument("--dropout", default=0.7, type=float, help="Dropout rate")
    parser.add_argument("--momentum", default=0.95, type=float, help="SGD momentum")
    parser.add_argument("--weight-decay", default=0.00001, type=float, help="L2 weight decay")
    parser.add_argument("--ont-hops", default=None, type=int, required=False,
                        help="Number of ontology hops (leave empty for no ontology injection)")
    args = parser.parse_args()

    year: int = args.year
    lcr_hops: int = args.hops
    dropout_rate: float = args.dropout
    learning_rate: float = args.lr
    momentum: float = args.momentum
    weight_decay: float = args.weight_decay
    ont_hops: Optional[int] = args.ont_hops

    n_epochs = 100
    batch_size = 32
    patience = 30

    set_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = EmbeddingsDataset(year=year, device=device, phase="Train", ont_hops=ont_hops)
    print(f"Using {train_dataset} with {len(train_dataset)} obs for training")
    train_idx, validation_idx = train_validation_split(train_dataset, seed=SEED)

    training_subset = Subset(train_dataset, train_idx)
    validation_subset = Subset(train_dataset, validation_idx)
    print(f"Using {train_dataset} with {len(validation_subset)} obs for validation")

    training_loader = DataLoader(training_subset, batch_size=batch_size, collate_fn=lambda batch: batch)
    validation_loader = DataLoader(validation_subset, collate_fn=lambda batch: batch)

    model = LCRRotHopPlusPlus(hops=lcr_hops, dropout_prob=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    best_accuracy: Optional[float] = None
    best_state_dict: Optional[dict] = None

    models_dir = os.path.join("data", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir,
                              f"{year}_LCR_hops{lcr_hops}_dropout{stringify_float(dropout_rate)}_acc{stringify_float(best_accuracy)}.pt")
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_path)

    train_losses = []
    valid_losses = []

    epochs_progress = tqdm(range(n_epochs), unit='epoch')

    try:
        for epoch in epochs_progress:
            model.train()
            epoch_progress = tqdm(training_loader, unit='batch', leave=False)

            train_loss = 0.0
            train_n_correct = 0
            train_steps = 0
            train_n = 0

            for i, batch in enumerate(epoch_progress):
                torch.set_default_device(device)

                batch_outputs = torch.stack(
                    [model(left, target, right, hops) for (left, target, right), _, hops in batch], dim=0)
                batch_labels = torch.tensor([label.item() for _, label, _ in batch])

                loss: torch.Tensor = criterion(batch_outputs, batch_labels)

                train_loss += loss.item()
                train_steps += 1
                train_n_correct += (batch_outputs.argmax(1) == batch_labels).type(torch.int).sum().item()
                train_n += len(batch)

                epoch_progress.set_description(
                    f"Train Loss: {train_loss / train_steps:.3f}, Train Acc.: {train_n_correct / train_n:.3f}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                torch.set_default_device('cpu')

            model.eval()
            epoch_progress = tqdm(validation_loader, unit='obs', leave=False)
            val_loss = 0.0
            val_steps = 0
            val_n = 0
            val_n_correct = 0

            for i, data in enumerate(epoch_progress):
                torch.set_default_device(device)

                with torch.inference_mode():
                    (left, target, right), label, hops = data[0]
                    output: torch.Tensor = model(left, target, right, hops)
                    val_n_correct += (output.argmax(0) == label).type(torch.int).item()
                    val_n += 1
                    loss = criterion(output, label)
                    val_loss += loss.item()
                    val_steps += 1
                    valid_losses.append(loss.item())

                torch.set_default_device('cpu')

            validation_accuracy = val_n_correct / val_n
            train_loss_avg = np.average(train_losses)
            valid_loss_avg = np.average(valid_losses)

            epoch_len = len(str(n_epochs))
            print(f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] '
                  f'train_loss: {train_loss_avg:.5f} '
                  f'valid_loss: {valid_loss_avg:.5f}')

            train_losses = []
            valid_losses = []

            early_stopping(valid_loss_avg, model)

            if best_accuracy is None or validation_accuracy > best_accuracy:
                epochs_progress.set_description(f"Best Val Acc.: {validation_accuracy:.3f}")
                best_accuracy = validation_accuracy
                best_state_dict = model.state_dict()

            if early_stopping.early_stop:
                print("Early stopping")
                break

    except KeyboardInterrupt:
        print("Interrupted training, saving best model...")

    if best_state_dict is not None:
        models_dir = os.path.join("data", "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir,
                                  f"{year}_LCR_hops{lcr_hops}_lr{stringify_float(learning_rate)}"
                                  f"_dropout{stringify_float(dropout_rate)}_acc{stringify_float(best_accuracy)}.pt")
        with open(model_path, "wb") as f:
            torch.save(best_state_dict, f)
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
