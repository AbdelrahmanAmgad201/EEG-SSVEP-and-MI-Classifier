import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from sklearn.cross_decomposition import CCA
import joblib
from eegnet_model import EEGNet
from tqdm import tqdm

# ----------------------
# Configurations
# ----------------------
DATA_PATH = 'mtc-aic3_dataset'
SSVEP_CHANNELS = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
SAMPLES_PER_TRIAL = 1750
NUM_CHANNELS = 8
TRIAL_DURATION = 7
SAMPLING_RATE = 250
NUM_CLASSES = 4
LABEL_MAP = {'Left':0, 'Right':1, 'Forward':2, 'Backward':3}

# ----------------------
# Data Loader
# ----------------------
def load_trial_data(row, base_path=DATA_PATH):
    dataset = 'train' if row['id'] <= 4800 else ('validation' if row['id'] <= 4900 else 'test')
    eeg_path = f"{base_path}/SSVEP/{dataset}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"
    eeg_data = pd.read_csv(eeg_path)
    trial_num = int(row['trial'])
    start_idx = (trial_num - 1) * SAMPLES_PER_TRIAL
    end_idx = start_idx + SAMPLES_PER_TRIAL
    trial_data = eeg_data[SSVEP_CHANNELS].iloc[start_idx:end_idx].values.T  # shape: (channels, samples)
    return trial_data

class SSVEPDataset(Dataset):
    def __init__(self, index_df, base_path=DATA_PATH, labels_available=True):
        self.index_df = index_df.reset_index(drop=True)
        self.base_path = base_path
        self.labels_available = labels_available
        self.label_map = LABEL_MAP
    def __len__(self):
        return len(self.index_df)
    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        X = load_trial_data(row, self.base_path)
        if self.labels_available:
            y = self.label_map[row['label']]
            return torch.tensor(X, dtype=torch.float32), y
        else:
            return torch.tensor(X, dtype=torch.float32)

# ----------------------
# Preprocessing
# ----------------------
def run_ica_and_remove(X, threshold=0.8):
    # X: (channels, samples)
    ica = FastICA(n_components=NUM_CHANNELS, random_state=42, max_iter=1000)
    S = ica.fit_transform(X.T)  # shape: (samples, components)
    m = np.abs(kurtosis(S, axis=0))
    keep = m < threshold
    S_clean = S[:, keep]
    X_clean = ica.mixing_[:, keep] @ S_clean.T  # shape: (channels_kept, samples)
    return X_clean, ica, keep

def apply_ica(X, ica, keep):
    S = ica.transform(X.T)
    S_clean = S[:, keep]
    X_clean = ica.mixing_[:, keep] @ S_clean.T
    return X_clean

def run_cca(X, y, n_components=4):
    cca = CCA(n_components=n_components)
    X_c, y_c = cca.fit_transform(X.T, y)
    return X_c.T, cca

def apply_cca(X, cca):
    X_c, _ = cca.transform(X.T, np.zeros((X.shape[1], cca.y_weights_.shape[0])))
    return X_c.T

# ----------------------
# Training & Validation
# ----------------------
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        X = X.unsqueeze(1)  # [batch, 1, channels, samples]
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
    return running_loss / len(dataloader.dataset)

def validate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.unsqueeze(1)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

# ----------------------
# Main Pipeline
# ----------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load index files
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_PATH, 'validation.csv'))
    # Filter for SSVEP only
    train_df = train_df[train_df['task'] == 'SSVEP']
    val_df = val_df[val_df['task'] == 'SSVEP']
    # Use only first 100 rows for training
    train_df = train_df.head(100)
    # Preprocess and cache
    preprocess_dir = 'preprocessing_cache'
    os.makedirs(preprocess_dir, exist_ok=True)
    def preprocess_and_cache(df, split):
        X_list, y_list = [], []
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f'Preprocessing {split} data'):
            cache_file = os.path.join(preprocess_dir, f"{split}_{row['id']}.npy")
            if os.path.exists(cache_file):
                X_clean = np.load(cache_file)
            else:
                X = load_trial_data(row)
                X_clean, ica, keep = run_ica_and_remove(X, threshold=0.8)
                if args.use_cca:
                    y_dummy = np.zeros((SAMPLES_PER_TRIAL, NUM_CLASSES))
                    y_dummy[:, LABEL_MAP[row['label']]] = 1
                    X_clean, cca = run_cca(X_clean, y_dummy, n_components=4)
                    joblib.dump(cca, os.path.join(preprocess_dir, f"cca_{split}_{row['id']}.pkl"))
                np.save(cache_file, X_clean)
                joblib.dump((ica, keep), os.path.join(preprocess_dir, f"ica_{split}_{row['id']}.pkl"))
            X_list.append(X_clean)
            if 'label' in row:
                y_list.append(LABEL_MAP[row['label']])
        return X_list, y_list
    if args.mode == 'train':
        print('Preprocessing training data...')
        X_train, y_train = preprocess_and_cache(train_df, 'train')
        print('Preprocessing validation data...')
        X_val, y_val = preprocess_and_cache(val_df, 'val')
        # Convert to tensors
        X_train = torch.tensor(np.stack([x for x in X_train]), dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(np.stack([x for x in X_val]), dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)
        # Dataloaders
        train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=16, shuffle=True)
        val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=16)
        # Model: set num_channels to match preprocessed data
        num_channels = X_train.shape[1]
        model = EEGNet(num_classes=NUM_CLASSES, num_channels=num_channels, sampling_rate=SAMPLING_RATE, trial_duration=TRIAL_DURATION)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # Training loop
        best_acc = 0
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            loss = 0.0
            model.train()
            for X, y in tqdm(train_loader, desc='Training', leave=False):
                X, y = X.to(device), y.to(device)
                X = X.unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(X)
                l = criterion(outputs, y)
                l.backward()
                optimizer.step()
                loss += l.item() * X.size(0)
            loss /= len(train_loader.dataset)
            acc = validate_model(model, val_loader, device)
            print(f"Loss: {loss:.4f} - Val Acc: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'best_ssvep_model.pth')
        print(f"Best validation accuracy: {best_acc:.4f}")
    elif args.mode == 'infer':
        # Inference example: load model and run on validation set
        X_val, y_val = preprocess_and_cache(val_df, 'val')
        X_val = torch.tensor(np.stack([x for x in X_val]), dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)
        num_channels = X_val.shape[1]
        model = EEGNet(num_classes=NUM_CLASSES, num_channels=num_channels, sampling_rate=SAMPLING_RATE, trial_duration=TRIAL_DURATION)
        model.load_state_dict(torch.load('best_ssvep_model.pth', map_location=device))
        model.to(device)
        model.eval()
        val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=16)
        acc = validate_model(model, val_loader, device)
        print(f"Validation accuracy (inference): {acc:.4f}")
    else:
        print('Unknown mode. Use --mode train or --mode infer')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full SSVEP Deep Learning Pipeline with ICA, CCA, and EEGNet')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'], help='train or infer')
    parser.add_argument('--use_cca', action='store_true', help='Use CCA as an additional preprocessing step')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    args = parser.parse_args()
    main(args) 