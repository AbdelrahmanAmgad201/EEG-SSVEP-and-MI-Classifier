"""
Official SSVEP ROCKET pipeline with ICA+MARA artifact rejection.
Dependencies: tsai, torch, mne, mne-mara, scikit-learn, tqdm, numpy, pandas
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import mne
from mne_mara import compute_mara
from sklearn.utils import shuffle
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import confusion_matrix
import tsai.all as ts

# Parameters
DATA_ROOT = 'mtc-aic3_dataset/SSVEP/train'
CHANNELS = ['PO7', 'OZ', 'PO8']
NEW_SRATE = 200
TRIAL_SEC = 7
TRIAL_SAMPLES = NEW_SRATE * TRIAL_SEC
N_CHANNELS = len(CHANNELS)
CV_NUMBER = 20
TEST_SIZE = 0.1
ICA_COMPONENTS = 3  # number of ICA components to use
MARA_THRESHOLD = 0.5  # threshold for artifact classification

# Helper: Load a single trial, apply ICA+MARA, and return occipital channels
def load_and_preprocess_trial(eeg_path, trial_num):
    eeg_data = pd.read_csv(eeg_path)
    step = int(250 / NEW_SRATE)
    start_idx = (trial_num - 1) * 1750
    end_idx = start_idx + 1750
    trial = eeg_data.iloc[start_idx:end_idx][CHANNELS].values
    trial = trial[::step]  # downsample to 200Hz
    trial_T = trial.T
    info = mne.create_info(ch_names=CHANNELS, sfreq=NEW_SRATE, ch_types='eeg')
    raw = mne.io.RawArray(trial_T, info, verbose=False)
    ica = mne.preprocessing.ICA(n_components=ICA_COMPONENTS, method='picard', max_iter='auto', random_state=97, verbose=False)
    ica.fit(raw)
    mara_scores = compute_mara(raw, ica)
    artifact_idx = np.where(mara_scores > MARA_THRESHOLD)[0]
    raw_clean = ica.apply(raw, exclude=artifact_idx)
    cleaned = raw_clean.get_data().T
    cleaned = (cleaned - cleaned.mean(axis=0)) / cleaned.std(axis=0)
    return cleaned.T

# 1. Load index and build dataset
train_df = pd.read_csv('mtc-aic3_dataset/train.csv')
ssvep_df = train_df[train_df['task'] == 'SSVEP']

X = []
y = []
print('Loading and preprocessing trials with ICA+MARA...')
for idx, row in tqdm(ssvep_df.iterrows(), total=len(ssvep_df)):
    eeg_path = os.path.join(DATA_ROOT, row['subject_id'], str(row['trial_session']), 'EEGdata.csv')
    try:
        trial = load_and_preprocess_trial(eeg_path, int(row['trial']))
        X.append(trial)
        y.append(row['label'])
    except Exception as e:
        print(f'Error at {eeg_path}, trial {row["trial"]}: {e}')

X = np.stack(X)
y = np.array(y)

# 2. Cross-validation with ROCKET + RidgeClassifierCV
def cross_validation_rocket(X, y, cv_number=CV_NUMBER, test_size=TEST_SIZE):
    acc = []
    cfm = np.zeros((4, 4))
    batch_tfms = [ts.TSStandardize(by_sample=True)]
    tfms = [None, [ts.Categorize()]]
    for _ in tqdm(range(cv_number), desc='Cross-validation'):
        X_, y_ = shuffle(X, y)
        splits = ts.get_splits(y_, valid_size=test_size, stratify=True, shuffle=True, random_state=42)
        dls = ts.get_ts_dls(X_, y_, splits=splits, tfms=tfms, drop_last=False, shuffle_train=False, batch_tfms=batch_tfms, bs=10000)
        model = ts.build_ts_model(ts.ROCKET, dls=dls)
        X_train, y_train = ts.create_rocket_features(dls.train, model)
        X_valid, y_valid = ts.create_rocket_features(dls.valid, model)
        ridge = RidgeClassifierCV(alphas=np.logspace(-8, 8, 17), normalize=True)
        ridge.fit(X_train, y_train)
        acc.append(ridge.score(X_valid, y_valid))
        cfm += confusion_matrix(y_valid, ridge.predict(X_valid), labels=[0,1,2,3])
        print(f'alpha: {ridge.alpha_:.2E}  train: {ridge.score(X_train, y_train):.5f}  valid: {ridge.score(X_valid, y_valid):.5f}')
    return acc, cfm

acc, cfm = cross_validation_rocket(X, y)

# 3. Save results
pd.DataFrame({'accuracy': acc}).to_csv('mtc-aic3_dataset/rocket_cv_accuracy.csv', index=False)
pd.DataFrame(cfm, columns=['Left','Right','Forward','Backward'], index=['Left','Right','Forward','Backward']).to_csv('mtc-aic3_dataset/rocket_confusion_matrix.csv')
print('Cross-validation accuracy and confusion matrix saved.') 