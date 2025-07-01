import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from collections import defaultdict

# -----------------------------
# Helper Functions
# -----------------------------
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    return lfilter(b, a, data, axis=-1)

def compute_accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

def pattern_match(test_epoch, templates):
    # Correlate test_epoch with each template, return the index of the best match
    corrs = [np.corrcoef(test_epoch.flatten(), t.flatten())[0,1] for t in templates]
    return np.argmax(corrs)

# -----------------------------
# Data Loader and Slicer
# -----------------------------
class EEGDataLoader:
    def __init__(self, data_path, index_csv, channels, samples_per_trial, fs):
        self.data_path = data_path
        self.index_df = pd.read_csv(index_csv)
        self.channels = channels
        self.samples_per_trial = samples_per_trial
        self.fs = fs
    def get_trials(self, max_trials=None):
        # Returns: epochs (n_trials, n_channels, n_samples), labels (n_trials,)
        epochs = []
        labels = []
        for i, row in self.index_df.iterrows():
            if max_trials and len(epochs) >= max_trials:
                break
            if row['task'] != 'SSVEP':
                continue
            eeg_path = os.path.join(self.data_path, 'SSVEP', 'train', row['subject_id'], str(row['trial_session']), 'EEGdata.csv')
            eeg_data = pd.read_csv(eeg_path)
            trial_num = int(row['trial'])
            start_idx = (trial_num - 1) * self.samples_per_trial
            end_idx = start_idx + self.samples_per_trial
            trial_data = eeg_data[self.channels].iloc[start_idx:end_idx].values.T  # (channels, samples)
            epochs.append(trial_data)
            labels.append(row['label'])
        return np.array(epochs), np.array(labels)

# -----------------------------
# Filtering
# -----------------------------
class EEGFilter:
    def __init__(self, lowcut=6, highcut=80, fs=250, order=4):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
    def apply(self, epochs):
        # epochs: (n_trials, n_channels, n_samples)
        return np.array([bandpass_filter(epoch, self.lowcut, self.highcut, self.fs, self.order) for epoch in epochs])

# -----------------------------
# Feature Extraction
# -----------------------------
class CCAFeatureExtractor:
    def __init__(self, n_harmonics=2, fs=250, stim_freqs=None):
        self.n_harmonics = n_harmonics
        self.fs = fs
        self.stim_freqs = stim_freqs if stim_freqs is not None else [8, 10, 12, 15]  # Example SSVEP freqs
    def _reference_signals(self, n_samples, freq):
        t = np.arange(n_samples) / self.fs
        ref = []
        for h in range(1, self.n_harmonics+1):
            ref.append(np.sin(2 * np.pi * freq * h * t))
            ref.append(np.cos(2 * np.pi * freq * h * t))
        return np.stack(ref, axis=0)
    def extract(self, epoch):
        # epoch: (n_channels, n_samples)
        n_samples = epoch.shape[1]
        corrs = []
        for freq in self.stim_freqs:
            ref = self._reference_signals(n_samples, freq)
            cca = CCA(n_components=1)
            try:
                cca.fit(epoch.T, ref.T)
                U, V = cca.transform(epoch.T, ref.T)
                corr = np.corrcoef(U[:,0], V[:,0])[0,1]
            except Exception:
                corr = 0
            corrs.append(corr)
        return np.array(corrs)
    def batch_extract(self, epochs):
        return np.array([self.extract(epoch) for epoch in epochs])

class TRCAFeatureExtractor:
    def extract(self, epochs, labels):
        # epochs: (n_trials, n_channels, n_samples), labels: (n_trials,)
        # For each class, average all epochs to create a template
        templates = []
        for label in np.unique(labels):
            class_epochs = epochs[labels == label]
            templates.append(np.mean(class_epochs, axis=0))
        return templates

# -----------------------------
# Classifier
# -----------------------------
class LDAClassifier:
    def __init__(self):
        self.clf = LinearDiscriminantAnalysis()
    def fit(self, X, y):
        self.clf.fit(X, y)
    def predict(self, X):
        return self.clf.predict(X)
    def score(self, X, y):
        return self.clf.score(X, y)

# -----------------------------
# SSVEP Framework
# -----------------------------
class SSVEPFramework:
    def __init__(self, data_path, index_csv, channels, samples_per_trial, fs=250):
        self.loader = EEGDataLoader(data_path, index_csv, channels, samples_per_trial, fs)
        self.filter = EEGFilter(fs=fs)
        self.cca_extractor = CCAFeatureExtractor(fs=fs)
        self.trca_extractor = TRCAFeatureExtractor()
        self.classifier = LDAClassifier()
    def run_offline(self, max_trials=None, use_cca=True, n_splits=5):
        print('Loading and slicing data...')
        epochs, labels = self.loader.get_trials(max_trials=max_trials)
        print('Filtering...')
        epochs = self.filter.apply(epochs)
        if use_cca:
            print('Extracting CCA features...')
            X = self.cca_extractor.batch_extract(epochs)
        else:
            print('Extracting TRCA templates...')
            X = self.trca_extractor.extract(epochs, labels)
        print('Running cross-validation...')
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs = []
        for train_idx, test_idx in kf.split(epochs):
            if use_cca:
                self.classifier.fit(X[train_idx], labels[train_idx])
                y_pred = self.classifier.predict(X[test_idx])
            else:
                # Pattern matching with TRCA templates
                templates = [np.mean(epochs[train_idx][labels[train_idx]==l], axis=0) for l in np.unique(labels)]
                y_pred = [np.unique(labels)[pattern_match(epochs[i], templates)] for i in test_idx]
            acc = compute_accuracy(labels[test_idx], y_pred)
            accs.append(acc)
            print(f'Fold accuracy: {acc:.4f}')
        print(f'Average cross-validation accuracy: {np.mean(accs):.4f}')
    # You can add online/simulated mode here as needed

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    # Config
    DATA_PATH = 'mtc-aic3_dataset'
    INDEX_CSV = os.path.join(DATA_PATH, 'train.csv')
    CHANNELS = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    SAMPLES_PER_TRIAL = 1750
    FS = 250
    # Run framework
    framework = SSVEPFramework(DATA_PATH, INDEX_CSV, CHANNELS, SAMPLES_PER_TRIAL, fs=FS)
    framework.run_offline(max_trials=200, use_cca=True, n_splits=5) 