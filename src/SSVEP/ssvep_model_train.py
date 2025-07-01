import pandas as pd
import numpy as np
import os
from scipy.signal import butter, lfilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# 1. Load the train index and filter for SSVEP
train_df = pd.read_csv('mtc-aic3_dataset/train.csv')
ssvep_df = train_df[train_df['task'] == 'SSVEP']

# 2. Helper functions for preprocessing and feature extraction
def bandpass(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)

def extract_ssvep_features(trial, fs=250):
    n = trial.shape[0]
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(trial, axis=0))
    target_freqs = [7, 8, 10, 13]
    features = []
    for f in target_freqs:
        idx = np.argmin(np.abs(freqs - f))
        features.extend(fft_vals[idx, :])  # 8 channels
    return np.array(features)

def load_ssvep_trial(row):
    eeg_path = f"mtc-aic3_dataset/SSVEP/train/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"
    eeg_data = pd.read_csv(eeg_path)
    trial_num = int(row['trial'])
    samples_per_trial = 1750
    start_idx = (trial_num - 1) * samples_per_trial
    end_idx = start_idx + samples_per_trial
    eeg_channels = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    trial_data = eeg_data.loc[start_idx:end_idx-1, eeg_channels].values
    return trial_data

# 3. Build feature matrix and label vector
X = []
y = []
fs = 250
for idx, row in ssvep_df.iterrows():
    try:
        trial = load_ssvep_trial(row)
        filtered = bandpass(trial, 6, 15, fs)
        standardized = (filtered - filtered.mean(axis=0)) / filtered.std(axis=0)
        features = extract_ssvep_features(standardized)
        X.append(features)
        y.append(row['label'])
    except Exception as e:
        print(f"Error processing row {idx}: {e}")

X = np.array(X)
y = np.array(y)

# 4. Train/test split for quick validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# 5. Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Evaluate
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))

# 7. Save model
joblib.dump(clf, 'mtc-aic3_dataset/ssvep_rf_model.joblib')
print('Model saved to mtc-aic3_dataset/ssvep_rf_model.joblib') 