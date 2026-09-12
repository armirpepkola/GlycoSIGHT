import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import os
import glob
import pandas as pd
import numpy as np

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

# 1. Add the real dataset loader for the Transformer
class RealForecastingDataset(Dataset):
    def __init__(self, data_dir='./processed_data', look_back=36, predict_horizon=24):
        self.samples = []
        csv_files = glob.glob(os.path.join(data_dir, "processed_patient_*.csv"))
        
        if not csv_files:
            print("Warning: No processed CSV files found in", data_dir)
            
        for file in csv_files:
            df = pd.read_csv(file)
            # Extract the 5 expected features
            data = df[['glucose', 'insulin_bolus', 'hfa_c', 'hfa_f', 'hfa_p']].values.astype(np.float32)
            
            # Create sliding windows (36 steps past -> 24 steps future)
            window_size = look_back + predict_horizon
            for i in range(len(data) - window_size + 1):
                src = data[i : i + look_back]
                tgt = data[i + look_back : i + window_size, 0:1] # Predict only glucose
                self.samples.append({'src': src, 'tgt': tgt})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            'src': torch.FloatTensor(self.samples[idx]['src']),
            'tgt': torch.FloatTensor(self.samples[idx]['tgt'])
        }

def train_hfa(args):
    print("\n--- Training HFA Model (Placeholder) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = DummyVisionDataset()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    model = HierarchicalFoodAnalysis(num_food_classes=102, pretrained=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1): # Reduced epochs since it's just dummy data
        model.train()
        progress_bar = tqdm(train_loader, desc=f"HFA Epoch 1/1")
        for batch in progress_bar:
            images = batch['image'].to(device)
            target_volume = batch['volume'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs['volume'], target_volume)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), './best_hfa_model.pth')
    print("-> Saved 'best_hfa_model.pth' to project directory.")

def train_transformer(args):
    print("\n--- Training Transformer Model (Real Data) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Swap the dummy dataset for the real one
    train_dataset = RealForecastingDataset()
    if len(train_dataset) == 0:
        print("Error: Dataset is empty. Run preprocess_ohio.py first.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

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

    torch.save(model.state_dict(), './best_transformer_model.pth')
    print("-> Saved 'best_transformer_model.pth' to project directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train GlycoSIGHT models.")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    train_hfa(args)
    train_transformer(args)