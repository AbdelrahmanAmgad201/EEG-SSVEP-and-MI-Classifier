import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ----------------------
# CONFIGURATION
# ----------------------
DATA_DIR = 'mtc-aic3_dataset'
EEG_CHANNELS = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
SAMPLE_RATE = 250
TRIAL_DURATION = 7  # seconds
SAMPLES_PER_TRIAL = SAMPLE_RATE * TRIAL_DURATION  # 1750
CLASS_MAP = {'Left': 0, 'Right': 1, 'Forward': 2, 'Backward': 3}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

# ----------------------
# PREPROCESSING
# ----------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=6, highcut=30, fs=250, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data, axis=0)

# ----------------------
# DATASET
# ----------------------
class SSVEPDataset(Dataset):
    def __init__(self, index_df, data_dir, scaler=None, fit_scaler=False):
        self.index_df = index_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.scaler = scaler
        self.fit_scaler = fit_scaler
        self.X = []
        self.y = []
        self._prepare()

    def _prepare(self):
        # Only use SSVEP trials
        ssvep_df = self.index_df[self.index_df['task'] == 'SSVEP']
        for i, row in tqdm(ssvep_df.iterrows(), total=len(ssvep_df), desc='Loading EEG trials'):
            eeg = self._load_trial(row)
            eeg = bandpass_filter(eeg)
            if self.fit_scaler:
                self.X.append(eeg)
                self.y.append(CLASS_MAP[row['label']])
            else:
                self.X.append(eeg)
                self.y.append(CLASS_MAP[row['label']])
        self.X = np.stack(self.X)
        self.y = np.array(self.y)
        if self.fit_scaler:
            # Fit scaler on all training data
            N, T, C = self.X.shape
            self.scaler = StandardScaler()
            self.X = self.X.reshape(-1, C)
            self.X = self.scaler.fit_transform(self.X)
            self.X = self.X.reshape(N, T, C)
        elif self.scaler is not None:
            N, T, C = self.X.shape
            self.X = self.X.reshape(-1, C)
            self.X = self.scaler.transform(self.X)
            self.X = self.X.reshape(N, T, C)

    def _load_trial(self, row):
        # Find dataset type
        id_num = row['id']
        if id_num <= 4800:
            dataset = 'train'
        elif id_num <= 4900:
            dataset = 'validation'
        else:
            dataset = 'test'
        eeg_path = os.path.join(
            self.data_dir,
            'SSVEP',
            dataset,
            str(row['subject_id']),
            str(row['trial_session']),
            'EEGdata.csv'
        )
        # Load only the needed rows
        trial_num = int(row['trial'])
        skiprows = 1 + (trial_num - 1) * SAMPLES_PER_TRIAL
        nrows = SAMPLES_PER_TRIAL
        eeg_df = pd.read_csv(eeg_path, skiprows=range(1, skiprows), nrows=nrows)
        eeg = eeg_df[EEG_CHANNELS].values.astype(np.float32)
        return eeg

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return as (channels, time)
        return torch.tensor(self.X[idx].T, dtype=torch.float32), self.y[idx]

# ----------------------
# MODEL
# ----------------------
class SSVEPNet(nn.Module):
    def __init__(self, n_channels=8, n_classes=4, input_len=1750):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        cnn_out_len = input_len // 8
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * cnn_out_len, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

# ----------------------
# TRAINING
# ----------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    all_preds = []
    all_labels = []
    for X, y in tqdm(loader, desc='Train', leave=False):
        X, y = X.to(device), torch.tensor(y).to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return np.mean(losses), acc

def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in tqdm(loader, desc='Val', leave=False):
            X, y = X.to(device), torch.tensor(y).to(device)
            out = model(X)
            loss = criterion(out, y)
            losses.append(loss.item())
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return np.mean(losses), acc

# ----------------------
# MAIN
# ----------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'validation.csv'))

    # Prepare datasets
    train_set = SSVEPDataset(train_df, DATA_DIR, fit_scaler=True)
    val_set = SSVEPDataset(val_df, DATA_DIR, scaler=train_set.scaler)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    # Model
    model = SSVEPNet(n_channels=8, n_classes=4, input_len=SAMPLES_PER_TRIAL).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    n_epochs = 20
    for epoch in range(1, n_epochs+1):
        print(f'\nEpoch {epoch}/{n_epochs}')
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        print(f'Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc:.4f}')

if __name__ == '__main__':
    main() 