import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import os

from models import HierarchicalFoodAnalysis, NutrientAwareTransformer


class DummyVisionDataset(Dataset):
    def __init__(self, length=100, num_classes=102):
        self.length = length
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            'image': torch.randn(3, 256, 256),
            'mask': torch.randint(0, self.num_classes, (256, 256)),
            'class_labels': torch.rand(self.num_classes),
            'volume': torch.rand(self.num_classes)
        }


class DummyForecastingDataset(Dataset):
    def __init__(self, length=200, look_back=36, predict_horizon=24):
        self.length = length
        self.look_back = look_back
        self.predict_horizon = predict_horizon

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            'src': torch.randn(self.look_back, 5),
            'tgt': torch.randn(self.predict_horizon, 1)
        }



def train_hfa(args):
    print("\n--- Training HFA Model ---")
    device = torch.device("cpu")

    train_dataset = DummyVisionDataset()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    model = HierarchicalFoodAnalysis(num_food_classes=102, pretrained=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"HFA Epoch {epoch + 1}/{args.epochs}")
        for batch in progress_bar:
            images = batch['image'].to(device)
            target_volume = batch['volume'].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs['volume'], target_volume)

            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': loss.item()})

    torch.save(model.state_dict(), '../best_hfa_model.pth')
    print("-> Saved 'best_hfa_model.pth' to project directory.")


def train_transformer(args):

    print("\n--- Training Transformer Model ---")
    device = torch.device("cpu")

    train_dataset = DummyForecastingDataset()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    model = NutrientAwareTransformer(output_seq_len=24).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Transformer Epoch {epoch + 1}/{args.epochs}")
        for batch in progress_bar:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            optimizer.zero_grad()
            prediction = model(src, tgt)
            loss = criterion(prediction, tgt)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': loss.item()})

    torch.save(model.state_dict(), '../best_transformer_model.pth')
    print("-> Saved 'best_transformer_model.pth' to project directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train GlycoSIGHT models.")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (keep this low for a quick test)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (keep this low)')
    args = parser.parse_args()

    print("Starting the training process. This will generate the .pth model files.")
    print("This may take several minutes depending on your computer's speed.")

    train_hfa(args)
    train_transformer(args)

    print("\nModel training complete! You can now run the 'evaluate_system.py' script.")