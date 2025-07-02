import os
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt, butter , coherence, csd, welch, hilbert
import scipy.signal as signal
from scipy.stats import kurtosis , skew
import pandas as pd
import joblib
from xgboost import XGBClassifier
################################################
# Replace with your path of data to be infered #
################################################
Path_of_the_data_set = r"C:\Users\Badr\Desktop\mtcaic3\MI\test"
# ---------------------------------------------------------------------------
# Independent Component Analysis (ICA) helper
# ---------------------------------------------------------------------------
def apply_ica(data):
    data = np.asarray(data)                           # ensure ndarray
    if data.ndim == 1:                                # reshape 1‑D → 2‑D
        data = data[:, None]

    ica = FastICA(                                    # ICA model
        n_components=data.shape[1],
        random_state=42,
        max_iter=2000,
        tol=1e-5
    )

    S = ica.fit_transform(data)                       # sources
    k = np.abs(kurtosis(S, axis=0, fisher=False))     # kurtosis per source
    S[:, k > 5] = 0                                   # suppress artifactual comps
    cleaned = ica.inverse_transform(S)                # back‑project
    return cleaned.squeeze()                          # drop singleton dims


# ---------------------------------------------------------------------------
# End‑to‑end preprocessing + feature extraction
# ---------------------------------------------------------------------------
def preprocessing_and_loading(
    subjects_of_train,
    path_of_MI_diret_train,
    output_csv="MI_features.csv"
):
    """Pipeline: load raw MI data → preprocess → extract features → CSV.

    Each row in the resulting CSV corresponds to one trial.
    """
    # ----------------------------- constants --------------------------------
    fs = 250                          # sampling frequency (Hz)
    baseline_end = fs                 # 1‑s baseline (first 250 samples)
    fbcsp_bands = [                   # FBCSP sub‑bands (Hz)
        (8, 12), (12, 16), (16, 20),
        (20, 24), (24, 28), (28, 32)
    ]

    # -------------------- helper: band‑power (FFT) --------------------------
    def bandpower(sig, f_lo, f_hi):
        """Relative power within [f_lo, f_hi] using periodogram."""
        freqs = np.fft.rfftfreq(len(sig), 1 / fs)
        pxx = np.abs(np.fft.rfft(sig)) ** 2
        band_mask = (freqs >= f_lo) & (freqs <= f_hi)
        return np.sum(pxx[band_mask]) / len(sig)

    # ----------------------------- main loop --------------------------------
    rows, X_all = [], []

    for sbj in subjects_of_train:
        # Notch + band‑pass filters
        f0, Q = 50, 30                               # notch (50 Hz line noise)
        b_notch, a_notch = iirnotch(f0, Q, fs)
        b_bp, a_bp = butter(4, [8 / (fs / 2), 30 / (fs / 2)], btype="band")

        sbj_path = os.path.join(path_of_MI_diret_train, sbj)

        for sess in os.listdir(sbj_path):
            # Assumes a single CSV per session directory
            csv_path = os.path.join(sbj_path, sess,
                                    os.listdir(os.path.join(sbj_path, sess))[0])
            data_df = pd.read_csv(csv_path)

            for tr in range(1, 11):                  # 10 trials per session
                # ------------------------------------------------------------------
                # 1) Segment & ICA cleaning
                # ------------------------------------------------------------------
                seg = data_df.iloc[2250 * (tr - 1):2250 * tr, 1:9].to_numpy()
                seg = apply_ica(seg)

                # ------------------------------------------------------------------
                # 2) Filtering (notch → band‑pass)
                # ------------------------------------------------------------------
                c3 = filtfilt(b_notch, a_notch, seg[:, 1] - seg[:, 1].mean())
                cz = filtfilt(b_notch, a_notch, seg[:, 2] - seg[:, 2].mean())
                c4 = filtfilt(b_notch, a_notch, seg[:, 3] - seg[:, 3].mean())

                for ch in range(seg.shape[1]):                       # band‑pass all
                    seg[:, ch] = filtfilt(b_bp, a_bp, seg[:, ch])

                c3 = filtfilt(b_bp, a_bp, c3)
                cz = filtfilt(b_bp, a_bp, cz)
                c4 = filtfilt(b_bp, a_bp, c4)

                # ------------------------------------------------------------------
                # 3) Baseline correction (ERD/ERS)
                # ------------------------------------------------------------------
                c3 -= c3[:baseline_end].mean()
                cz -= cz[:baseline_end].mean()
                c4 -= c4[:baseline_end].mean()

                # ------------------------------------------------------------------
                # 4) Time‑domain features
                # ------------------------------------------------------------------
                row = {
                    "subject_id": sbj,
                    "session_id": sess,
                    "trial_id": tr,
                    # basic stats (C3)
                    "min_val_C3":   float(c3.min()),
                    "max_val_C3":   float(c3.max()),
                    "mean_val_C3":  float(c3.mean()),
                    "energy_C3":    float(np.sum(c3 ** 2)),
                    # basic stats (C4)
                    "min_val_C4":   float(c4.min()),
                    "max_val_C4":   float(c4.max()),
                    "mean_val_C4":  float(c4.mean()),
                    "energy_C4":    float(np.sum(c4 ** 2))
                }

                # ------------------------------------------------------------------
                # 5) ERD/ERS (8–30 Hz band power)
                # ------------------------------------------------------------------
                base_c3 = bandpower(c3[:baseline_end], 8, 30)
                task_c3 = bandpower(c3[baseline_end:], 8, 30)
                base_c4 = bandpower(c4[:baseline_end], 8, 30)
                task_c4 = bandpower(c4[baseline_end:], 8, 30)

                row.update({
                    "alpha_C3":  float(bandpower(c3, 8, 12)),
                    "alpha_C4":  float(bandpower(c4, 8, 12)),
                    "erd_C3_pct": 10 * np.log10(task_c3 / base_c3) if base_c3 else 0.0,
                    "erd_C4_pct": 10 * np.log10(task_c4 / base_c4) if base_c4 else 0.0
                })

                # ------------------------------------------------------------------
                # 6) Connectivity: coherence / imaginary coherence / PLV
                # ------------------------------------------------------------------
                f_coh, coh = signal.coherence(c3, c4, fs=fs, nperseg=fs * 2)
                mu_band = (f_coh >= 8) & (f_coh <= 30)
                coh_mu = float(coh[mu_band].mean()) if mu_band.any() else 0.0

                f_csd, Pxy = signal.csd(c3, c4, fs=fs, nperseg=fs * 2)
                f_pxx, Pxx = signal.welch(c3, fs=fs, nperseg=fs * 2)
                f_pyy, Pyy = signal.welch(c4, fs=fs, nperseg=fs * 2)
                imcoh = np.imag(Pxy) / np.sqrt(Pxx * Pyy)
                imcoh_mu = float(imcoh[(f_csd >= 8) & (f_csd <= 30)].mean())

                phi = np.angle(signal.hilbert(c3)) - np.angle(signal.hilbert(c4))
                plv_val = float(np.abs(np.exp(1j * phi).mean()))

                row.update({
                    "coh_C3C4_mu":   coh_mu,
                    "imcoh_C3C4_mu": imcoh_mu,
                    "plv_C3C4":      plv_val
                })

                # ------------------------------------------------------------------
                # 7) Higher‑order stats
                # ------------------------------------------------------------------
                row.update({
                    "range_C3": float(np.ptp(c3)),
                    "std_C3":   float(c3.std()),
                    "skew_C3":  float(skew(c3)),
                    "kurt_C3":  float(kurtosis(c3)),
                    "range_C4": float(np.ptp(c4)),
                    "std_C4":   float(c4.std()),
                    "skew_C4":  float(skew(c4)),
                    "kurt_C4":  float(kurtosis(c4))
                })

                # ------------------------------------------------------------------
                # 8) Frequency‑domain features (whole 0–40 Hz spectrum)
                # ------------------------------------------------------------------
                freqs = np.fft.rfftfreq(len(c3), 1 / fs)
                mask40 = freqs <= 40
                power_c3 = np.abs(np.fft.rfft(c3)) ** 2
                power_c4 = np.abs(np.fft.rfft(c4)) ** 2

                mean_f_c3 = np.sum(freqs * power_c3) / np.sum(power_c3)
                mean_f_c4 = np.sum(freqs * power_c4) / np.sum(power_c4)
                bw_c3 = np.sqrt(np.sum(((freqs - mean_f_c3) ** 2) * power_c3) / np.sum(power_c3))
                bw_c4 = np.sqrt(np.sum(((freqs - mean_f_c4) ** 2) * power_c4) / np.sum(power_c4))
                beta_c3 = np.sum(power_c3[(freqs >= 13) & (freqs <= 30)]) / len(c3)
                beta_c4 = np.sum(power_c4[(freqs >= 13) & (freqs <= 30)]) / len(c4)

                row.update({
                    "peak_freq_C3":   float(freqs[np.argmax(power_c3)]),
                    "peak_freq_C4":   float(freqs[np.argmax(power_c4)]),
                    "mean_freq_C3":   float(mean_f_c3),
                    "mean_freq_C4":   float(mean_f_c4),
                    "bw_C3":          float(bw_c3),
                    "bw_C4":          float(bw_c4),
                    "beta_power_C3":  float(beta_c3),
                    "beta_power_C4":  float(beta_c4),
                    "psd_mean_C3":    float(Pxx.mean()),
                    "psd_mean_C4":    float(Pyy.mean())
                })

                # ------------------------------------------------------------------
                # 9) Filter bank CSP‑like band powers
                # ------------------------------------------------------------------
                for lo, hi in fbcsp_bands:
                    mask = (freqs >= lo) & (freqs < hi)
                    row[f"fbcsp_{lo}_{hi}_C3"] = float(np.sum(power_c3[mask]) / len(c3))
                    row[f"fbcsp_{lo}_{hi}_C4"] = float(np.sum(power_c4[mask]) / len(c4))

                # Accumulate
                rows.append(row)
                X_all.append(np.stack([c3, cz, c4], axis=1))

    # ----------------------------------------------------------------------
    # 10) Common Spatial Patterns (CSP) on concatenated trials
    # ----------------------------------------------------------------------
    if X_all:
        X_all_arr = np.array(X_all)
        cov = np.mean([np.cov(trial.T) for trial in X_all_arr], axis=0)
        eigvals, eigvecs = np.linalg.eigh(cov)
        n_filt = min(4, eigvecs.shape[1])
        W = eigvecs[:, eigvals.argsort()[::-1][:n_filt]]

        for row, trial in zip(rows, X_all_arr):
            Z = trial @ W                         # spatially‑filtered signals
            var = np.var(Z, axis=0)
            logvar = np.log(var / var.sum())      # normalized log‑variance
            for k, v in enumerate(logvar, start=1):
                row[f"csp{k}"] = float(v)

    # ----------------------------------------------------------------------
    # 11) Save to CSV
    # ----------------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df
# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

the_data_set=[]
subjects_of_data = os.listdir(Path_of_the_data_set)
the_data_set = preprocessing_and_loading(subjects_of_data,Path_of_the_data_set,"MI_features_train.csv")



#full feature list
feature_names = ['min_val_C3', 'max_val_C3', 'mean_val_C3', 'energy_C3', 'min_val_C4', 'max_val_C4', 'mean_val_C4',
                'energy_C4', 'alpha_C3', 'alpha_C4', 'erd_C3_pct', 'erd_C4_pct', 'coh_C3C4_mu', 'imcoh_C3C4_mu',
                'plv_C3C4', 'range_C3', 'std_C3', 'skew_C3', 'kurt_C3', 'range_C4', 'std_C4', 'skew_C4', 'kurt_C4',
                'peak_freq_C3', 'peak_freq_C4', 'mean_freq_C3', 'mean_freq_C4', 'bw_C3', 'bw_C4', 'beta_power_C3',
                'beta_power_C4', 'psd_mean_C3', 'psd_mean_C4', 'fbcsp_8_12_C3', 'fbcsp_8_12_C4', 'fbcsp_12_16_C3',
                'fbcsp_12_16_C4', 'fbcsp_16_20_C3', 'fbcsp_16_20_C4', 'fbcsp_20_24_C3', 'fbcsp_20_24_C4',
                'fbcsp_24_28_C3', 'fbcsp_24_28_C4', 'fbcsp_28_32_C3', 'fbcsp_28_32_C4', 'csp1', 'csp2', 'csp3']

# Inverse label mapping
inverse_label_map = {0: 'Left', 1: 'Right'}

# Load components
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler_for_best_features2.pkl")
best_features = joblib.load("selected_feature_list2.pkl")

# Step 1: Scale full Data (must have all 48 columns, in same order)
X_scaled_full = scaler.transform(the_data_set[feature_names])

# Step 2: Wrap scaled data back into a DataFrame for column access
X_scaled_df = pd.DataFrame(X_scaled_full, columns=feature_names)

# Step 3: Select only best features (now scaled)
X_new_scaled = X_scaled_df[best_features]
# Step 4: Predict
y_pred = model.predict(X_new_scaled)
y_proba = model.predict_proba(X_new_scaled)



# Step 5: Decode prediction
decoded_preds = [inverse_label_map[p] for p in y_pred]
# Step 6: Show results
for i in range(len(y_pred)):
    print(f"Sample {i+1}:")
    print(f"  Predicted class: {y_pred[i]} ({decoded_preds[i]})")
    print(f"  Class probabilities: {y_proba[i]}")
    print()
results_df = pd.DataFrame({

    "label": decoded_preds,

})

