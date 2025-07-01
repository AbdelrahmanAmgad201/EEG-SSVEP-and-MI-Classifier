import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, iirnotch
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from collections import defaultdict
import joblib
import argparse
from tqdm import tqdm

# -----------------------------
# Helper Functions
# -----------------------------
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    return lfilter(b, a, data, axis=-1)

def notch_filter(data, notch_freq, fs, quality=30):
    w0 = notch_freq / (fs / 2)  # Normalize frequency for iirnotch
    b, a = iirnotch(w0, quality)
    return lfilter(b, a, data, axis=-1)

def compute_accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

# -----------------------------
# Data Loader for MTC-AIC3 SSVEP
# -----------------------------
class EEGDataLoader:
    def __init__(self, data_path, index_csv, channels, samples_per_trial, fs):
        self.data_path = data_path
        self.index_df = pd.read_csv(index_csv)
        self.channels = channels
        self.samples_per_trial = samples_per_trial
        self.fs = fs
        # Only keep SSVEP trials
        self.index_df = self.index_df[self.index_df['task'] == 'SSVEP'].reset_index(drop=True)
    def get_trials(self, max_trials=None, test_mode=False):
        epochs = []
        labels = []
        ids = []
        for i, row in tqdm(self.index_df.iterrows(), total=len(self.index_df), desc='Loading trials'):
            if max_trials and len(epochs) >= max_trials:
                break
            dataset = 'test' if test_mode else 'train'
            eeg_path = os.path.join(self.data_path, 'SSVEP', dataset, row['subject_id'], str(row['trial_session']), 'EEGdata.csv')
            eeg_data = pd.read_csv(eeg_path)
            trial_num = int(row['trial'])
            start_idx = (trial_num - 1) * self.samples_per_trial
            end_idx = start_idx + self.samples_per_trial
            trial_data = eeg_data[self.channels].iloc[start_idx:end_idx].values.T  # (channels, samples)
            # Simple artifact rejection: skip epochs with large amplitude (e.g., >500 uV)
            if np.max(np.abs(trial_data)) > 500:
                continue
            epochs.append(trial_data)
            if not test_mode:
                labels.append(row['label'])
            ids.append(row['id'])
        if test_mode:
            return np.array(epochs), np.array(ids)
        else:
            return np.array(epochs), np.array(labels), np.array(ids)

# -----------------------------
# Filtering
# -----------------------------
class EEGFilter:
    def __init__(self, lowcut=6, highcut=20, fs=250, order=4, notch_freq=50):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.notch_freq = notch_freq
    def apply(self, epochs):
        filtered = []
        for epoch in tqdm(epochs, desc='Filtering epochs'):
            x = bandpass_filter(epoch, self.lowcut, self.highcut, self.fs, self.order)
            x = notch_filter(x, self.notch_freq, self.fs)
            filtered.append(x)
        return np.array(filtered)

# -----------------------------
# FBCCA Feature Extraction
# -----------------------------
class FBCCAFeatureExtractor:
    def __init__(self, fs=250, stim_freqs=None, n_harmonics=3):
        self.fs = fs
        self.n_harmonics = n_harmonics
        # SSVEP frequencies: Left=10Hz, Right=13Hz, Forward=7Hz, Backward=8Hz
        self.stim_freqs = stim_freqs if stim_freqs is not None else [10, 13, 7, 8]
    def _reference_signals(self, n_samples, freq):
        t = np.arange(n_samples) / self.fs
        ref = []
        for h in range(1, self.n_harmonics+1):
            ref.append(np.sin(2 * np.pi * freq * h * t))
            ref.append(np.cos(2 * np.pi * freq * h * t))
        return np.stack(ref, axis=0)
    def extract(self, epoch):
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
        return np.array([self.extract(epoch) for epoch in tqdm(epochs, desc='Extracting FBCCA features')])

# -----------------------------
# Classifiers
# -----------------------------
class KNNClassifier:
    def __init__(self, n_neighbors=5):
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    def fit(self, X, y):
        self.clf.fit(X, y)
    def predict(self, X):
        return self.clf.predict(X)
    def score(self, X, y):
        return self.clf.score(X, y)

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
# SSVEP Pipeline
# -----------------------------
class SSVEPFBCCAPipeline:
    def __init__(self, data_path, index_csv, channels, samples_per_trial, fs=250, classifier_type='lda'):
        self.data_path = data_path
        self.index_csv = index_csv
        self.channels = channels
        self.samples_per_trial = samples_per_trial
        self.fs = fs
        self.loader = EEGDataLoader(data_path, index_csv, channels, samples_per_trial, fs)
        self.filter = EEGFilter(fs=fs)
        self.fbcca_extractor = FBCCAFeatureExtractor(fs=fs)
        if classifier_type == 'knn':
            self.classifier = KNNClassifier()
        else:
            self.classifier = LDAClassifier()
    def run_cross_validation(self, max_trials=None, n_splits=5):
        print('Loading and slicing data...')
        epochs, labels, _ = self.loader.get_trials(max_trials=max_trials)
        print('Filtering...')
        epochs = self.filter.apply(epochs)
        print('Extracting FBCCA features...')
        X = self.fbcca_extractor.batch_extract(epochs)
        print('Running cross-validation...')
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs = []
        for train_idx, test_idx in kf.split(X):
            self.classifier.fit(X[train_idx], labels[train_idx])
            y_pred = self.classifier.predict(X[test_idx])
            acc = compute_accuracy(labels[test_idx], y_pred)
            accs.append(acc)
            print(f'Fold accuracy: {acc:.4f}')
        print(f'Average cross-validation accuracy: {np.mean(accs):.4f}')
    def train_and_save(self, model_path='fbcca_model.pkl'):
        print('Loading and slicing all training data...')
        epochs, labels, _ = self.loader.get_trials()
        print('Filtering...')
        epochs = self.filter.apply(epochs)
        print('Extracting FBCCA features...')
        X = self.fbcca_extractor.batch_extract(epochs)
        print('Training classifier...')
        self.classifier.fit(X, labels)
        print(f'Saving model to {model_path}')
        joblib.dump({'classifier': self.classifier, 'fbcca': self.fbcca_extractor, 'filter': self.filter, 'channels': self.channels}, model_path)
    def infer_on_test(self, test_index_csv, model_path='fbcca_model.pkl', output_csv='test_predictions.csv'):
        print('Loading model...')
        model = joblib.load(model_path)
        classifier = model['classifier']
        fbcca = model['fbcca']
        filter_obj = model['filter']
        channels = model['channels']
        print('Loading and slicing test data...')
        test_loader = EEGDataLoader(self.data_path, test_index_csv, channels, self.samples_per_trial, self.fs)
        epochs, ids = test_loader.get_trials(test_mode=True)
        print('Filtering...')
        epochs = filter_obj.apply(epochs)
        print('Extracting FBCCA features...')
        X = fbcca.batch_extract(epochs)
        print('Predicting...')
        y_pred = classifier.predict(X)
        # Save predictions
        df = pd.DataFrame({'id': ids, 'label': y_pred})
        df.to_csv(output_csv, index=False)
        print(f'Saved predictions to {output_csv}')

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSVEP FBCCA Pipeline')
    parser.add_argument('--mode', type=str, default='cv', choices=['cv', 'train', 'infer'], help='cv: cross-validation, train: train and save, infer: load and predict on test')
    parser.add_argument('--classifier', type=str, default='lda', choices=['knn', 'lda'], help='Classifier type')
    parser.add_argument('--model_path', type=str, default='fbcca_model.pkl', help='Path to save/load model')
    parser.add_argument('--output_csv', type=str, default='test_predictions.csv', help='Output CSV for test predictions')
    parser.add_argument('--max_trials', type=int, default=None, help='Max trials for cross-validation')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds for cross-validation')
    args = parser.parse_args()

    DATA_PATH = 'mtc-aic3_dataset'
    TRAIN_INDEX_CSV = os.path.join(DATA_PATH, 'train.csv')
    TEST_INDEX_CSV = os.path.join(DATA_PATH, 'test.csv')
    # Use only occipital channels for SSVEP
    CHANNELS = ['PO7', 'OZ', 'PO8', 'PZ']
    SAMPLES_PER_TRIAL = 1750
    FS = 250

    pipeline = SSVEPFBCCAPipeline(DATA_PATH, TRAIN_INDEX_CSV, CHANNELS, SAMPLES_PER_TRIAL, fs=FS, classifier_type=args.classifier)

    if args.mode == 'cv':
        pipeline.run_cross_validation(max_trials=args.max_trials, n_splits=args.n_splits)
    elif args.mode == 'train':
        pipeline.train_and_save(model_path=args.model_path)
    elif args.mode == 'infer':
        pipeline.infer_on_test(TEST_INDEX_CSV, model_path=args.model_path, output_csv=args.output_csv)

# For further improvement:
# - Try more advanced artifact rejection (e.g., ICA, EOG regression)
# - Try data augmentation (noise, time shift)
# - Try deep learning (EEGNet)
# - Use ensemble of classifiers

# For further improvement:
# - Try more advanced artifact rejection (e.g., ICA, EOG regression)
# - Try data augmentation (noise, time shift)
# - Try deep learning (EEGNet)
# - Use ensemble of classifiers 