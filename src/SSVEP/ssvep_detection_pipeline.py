import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, welch
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import os
import warnings
from tqdm import tqdm
import seaborn as sns
from collections import Counter
import mne
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.spatialfilters import CSP
warnings.filterwarnings('ignore')

class SSVEPDetector:
    """Advanced SSVEP detection using established libraries and methods"""
    
    def __init__(self, base_path='mtc-aic3_dataset'):
        self.base_path = base_path
        self.sampling_rate = 250  # Hz
        self.eeg_channels = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
        self.ssvep_channels = ['PO7', 'OZ', 'PO8']  # Occipital channels for SSVEP
        
        # Load the training data index
        self.train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
        self.validation_df = pd.read_csv(os.path.join(base_path, 'validation.csv'))
        self.test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))
        
        # SSVEP frequency mapping (typical SSVEP frequencies)
        self.ssvep_frequencies = {
            'Left': 8.0,      # 8 Hz
            'Right': 10.0,    # 10 Hz  
            'Forward': 12.0,  # 12 Hz
            'Backward': 15.0  # 15 Hz
        }
        
        # Initialize MNE info object
        self.info = mne.create_info(
            ch_names=self.eeg_channels,
            sfreq=self.sampling_rate,
            ch_types=['eeg'] * len(self.eeg_channels)
        )
        
        # Initialize classifiers
        self.cca_classifier = None
        self.psd_classifier = None
        self.riemann_classifier = None
        self.ensemble_classifier = None
        
    def load_trial_data(self, row):
        """Load EEG data for a specific trial"""
        id_num = row['id']
        if id_num <= 4800:
            dataset = 'train'
        elif id_num <= 4900:
            dataset = 'validation'
        else:
            dataset = 'test'
        
        eeg_path = f"{self.base_path}/{row['task']}/{dataset}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"
        eeg_data = pd.read_csv(eeg_path)
        
        trial_num = int(row['trial'])
        if row['task'] == 'MI':
            samples_per_trial = 2250  # 9 seconds * 250 Hz
        else:  # SSVEP
            samples_per_trial = 1750  # 7 seconds * 250 Hz
        
        start_idx = (trial_num - 1) * samples_per_trial
        end_idx = start_idx + samples_per_trial - 1
        
        trial_data = eeg_data.iloc[start_idx:end_idx+1]
        return trial_data
    
    def preprocess_signal(self, raw_data):
        """Advanced preprocessing using MNE"""
        if isinstance(raw_data, pd.DataFrame):
            data = raw_data[self.eeg_channels].values
        else:
            data = raw_data
        
        # Create MNE Raw object
        raw = mne.io.RawArray(data.T, self.info)
        
        # Apply preprocessing pipeline
        # 1. Filtering
        raw.filter(5, 40, picks='eeg', method='iir', iir_params=dict(order=4, ftype='butter'))
        
        # 2. Notch filter for power line
        raw.notch_filter(50, picks='eeg')
        
        # 3. Remove DC offset
        raw.apply_hilbert(picks='eeg', envelope=False)
        
        # 4. ICA for artifact removal
        try:
            ica = mne.preprocessing.ICA(n_components=8, random_state=42, max_iter=800)
            ica.fit(raw, picks='eeg')
            
            # Automatically detect and remove eye blinks and heart artifacts
            eog_indices, eog_scores = ica.find_bads_eog(raw)
            ica.exclude = eog_indices[:2]  # Remove top 2 EOG components
            
            # Apply ICA
            raw = ica.apply(raw)
        except Exception as e:
            print(f"ICA failed, continuing without it: {e}")
        
        return raw.get_data().T
    
    def generate_reference_signals(self, frequencies, duration=7.0, harmonics=3):
        """Generate reference signals for CCA"""
        t = np.arange(0, duration, 1/self.sampling_rate)
        reference_signals = {}
        
        for label, freq in frequencies.items():
            ref_signal = []
            for h in range(1, harmonics + 1):
                # Sine and cosine components for each harmonic
                ref_signal.extend([
                    np.sin(2 * np.pi * h * freq * t),
                    np.cos(2 * np.pi * h * freq * t)
                ])
            reference_signals[label] = np.array(ref_signal).T
        
        return reference_signals
    
    def extract_psd_features(self, data, freqs_range=(5, 40)):
        """Extract PSD-based features"""
        features = []
        
        for ch_idx in range(data.shape[1]):
            channel_data = data[:, ch_idx]
            
            # Calculate PSD
            freqs, psd = welch(channel_data, fs=self.sampling_rate, 
                             nperseg=256, scaling='density')
            
            # Filter frequencies of interest
            freq_mask = (freqs >= freqs_range[0]) & (freqs <= freqs_range[1])
            freqs_filtered = freqs[freq_mask]
            psd_filtered = psd[freq_mask]
            
            # Extract features
            # 1. Peak frequencies and amplitudes
            peaks, properties = signal.find_peaks(psd_filtered, height=np.max(psd_filtered)*0.1, distance=5)
            if len(peaks) > 0:
                peak_freqs = freqs_filtered[peaks]
                peak_amps = psd_filtered[peaks]
                # Get top 3 peaks
                top_indices = np.argsort(peak_amps)[-3:]
                top_freqs = peak_freqs[top_indices]
                top_amps = peak_amps[top_indices]
            else:
                top_freqs = np.zeros(3)
                top_amps = np.zeros(3)
            
            # 2. Statistical features
            mean_psd = np.mean(psd_filtered)
            std_psd = np.std(psd_filtered)
            max_psd = np.max(psd_filtered)
            
            # 3. Frequency band powers
            alpha_power = np.mean(psd_filtered[(freqs_filtered >= 8) & (freqs_filtered <= 13)])
            beta_power = np.mean(psd_filtered[(freqs_filtered >= 13) & (freqs_filtered <= 30)])
            
            # Combine features
            channel_features = np.concatenate([
                top_freqs, top_amps, [mean_psd, std_psd, max_psd, alpha_power, beta_power]
            ])
            features.extend(channel_features)
        
        return np.array(features)
    
    def extract_riemann_features(self, data):
        """Extract Riemannian geometry features using pyriemann"""
        # Calculate covariance matrices
        covs = Covariances(estimator='lwf').fit_transform(data.reshape(1, *data.shape))
        return covs[0]
    
    def train_cca_classifier(self, train_data, train_labels):
        """Train CCA-based classifier"""
        print("Training CCA classifier...")
        
        # Generate reference signals
        reference_signals = self.generate_reference_signals(self.ssvep_frequencies)
        
        # Train CCA for each class
        cca_models = {}
        for label in self.ssvep_frequencies.keys():
            if label in train_labels:
                # Get trials for this class
                class_indices = [i for i, l in enumerate(train_labels) if l == label]
                if len(class_indices) > 0:
                    class_data = [train_data[i] for i in class_indices]
                    
                    # Use first trial as template (or average)
                    template = np.mean(class_data, axis=0)
                    
                    # Create CCA model
                    cca = CCA(n_components=min(3, len(self.ssvep_channels)))
                    cca_models[label] = cca
        
        self.cca_classifier = cca_models
        print("✓ CCA classifier trained")
    
    def train_psd_classifier(self, train_data, train_labels):
        """Train PSD-based classifier"""
        print("Training PSD classifier...")
        
        # Extract PSD features
        train_features = []
        for trial_data in tqdm(train_data, desc="Extracting PSD features"):
            features = self.extract_psd_features(trial_data)
            train_features.append(features)
        
        X_train = np.array(train_features)
        y_train = np.array(train_labels)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train classifier (LDA works well for SSVEP)
        classifier = LinearDiscriminantAnalysis()
        classifier.fit(X_train_scaled, y_train)
        
        self.psd_classifier = {
            'classifier': classifier,
            'scaler': scaler
        }
        print("✓ PSD classifier trained")
    
    def train_riemann_classifier(self, train_data, train_labels):
        """Train Riemannian geometry classifier"""
        print("Training Riemannian classifier...")
        
        # Extract covariance features
        train_features = []
        for trial_data in tqdm(train_data, desc="Extracting Riemannian features"):
            cov = self.extract_riemann_features(trial_data)
            train_features.append(cov)
        
        # Train MDM classifier
        mdm = MDM()
        mdm.fit(train_features, train_labels)
        
        self.riemann_classifier = mdm
        print("✓ Riemannian classifier trained")
    
    def train_ensemble_classifier(self, train_data, train_labels):
        """Train ensemble classifier combining multiple methods"""
        print("Training ensemble classifier...")
        
        # Get predictions from all classifiers
        ensemble_features = []
        
        for trial_data in tqdm(train_data, desc="Extracting ensemble features"):
            features = []
            
            # PSD features
            psd_features = self.extract_psd_features(trial_data)
            features.extend(psd_features)
            
            # Riemannian features (flatten covariance matrix)
            riemann_features = self.extract_riemann_features(trial_data)
            features.extend(riemann_features.flatten())
            
            ensemble_features.append(features)
        
        X_train = np.array(ensemble_features)
        y_train = np.array(train_labels)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train ensemble classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        classifier.fit(X_train_scaled, y_train)
        
        self.ensemble_classifier = {
            'classifier': classifier,
            'scaler': scaler
        }
        print("✓ Ensemble classifier trained")
    
    def predict_cca(self, test_data):
        """Predict using CCA"""
        if not self.cca_classifier:
            return None
        
        predictions = []
        for trial_data in test_data:
            best_correlation = -1
            best_label = None
            
            for label, cca in self.cca_classifier.items():
                # Generate reference signal for this class
                ref_signal = self.generate_reference_signals({label: self.ssvep_frequencies[label]})[label]
                
                # Apply CCA
                try:
                    # Use occipital channels only
                    occipital_indices = [self.eeg_channels.index(ch) for ch in self.ssvep_channels]
                    occipital_data = trial_data[:, occipital_indices]
                    
                    # Calculate correlation
                    correlation = np.corrcoef(occipital_data.flatten(), ref_signal.flatten())[0, 1]
                    if not np.isnan(correlation) and correlation > best_correlation:
                        best_correlation = correlation
                        best_label = label
                except:
                    continue
            
            predictions.append(best_label if best_label else 'Left')
        
        return predictions
    
    def predict_psd(self, test_data):
        """Predict using PSD features"""
        if not self.psd_classifier:
            return None
        
        # Extract features
        test_features = []
        for trial_data in test_data:
            features = self.extract_psd_features(trial_data)
            test_features.append(features)
        
        X_test = np.array(test_features)
        X_test_scaled = self.psd_classifier['scaler'].transform(X_test)
        
        # Predict
        predictions = self.psd_classifier['classifier'].predict(X_test_scaled)
        return predictions
    
    def predict_riemann(self, test_data):
        """Predict using Riemannian geometry"""
        if not self.riemann_classifier:
            return None
        
        # Extract features
        test_features = []
        for trial_data in test_data:
            cov = self.extract_riemann_features(trial_data)
            test_features.append(cov)
        
        # Predict
        predictions = self.riemann_classifier.predict(test_features)
        return predictions
    
    def predict_ensemble(self, test_data):
        """Predict using ensemble classifier"""
        if not self.ensemble_classifier:
            return None
        
        # Extract ensemble features
        test_features = []
        for trial_data in test_data:
            features = []
            
            # PSD features
            psd_features = self.extract_psd_features(trial_data)
            features.extend(psd_features)
            
            # Riemannian features
            riemann_features = self.extract_riemann_features(trial_data)
            features.extend(riemann_features.flatten())
            
            test_features.append(features)
        
        X_test = np.array(test_features)
        X_test_scaled = self.ensemble_classifier['scaler'].transform(X_test)
        
        # Predict
        predictions = self.ensemble_classifier['classifier'].predict(X_test_scaled)
        return predictions
    
    def load_and_process_dataset(self, dataset_df, max_trials_per_class=None):
        """Load and process SSVEP dataset"""
        print(f"Loading and processing {len(dataset_df)} trials...")
        
        processed_data = []
        labels = []
        trial_info = []
        
        # Filter only SSVEP trials
        ssvep_trials = dataset_df[dataset_df['task'] == 'SSVEP'].copy()
        
        if len(ssvep_trials) == 0:
            print("No SSVEP trials found in this dataset")
            return processed_data, labels, trial_info
        
        # Check if we have labels
        if 'label' in ssvep_trials.columns:
            ssvep_trials = ssvep_trials.dropna(subset=['label'])
            
            if len(ssvep_trials) == 0:
                print("No SSVEP trials with valid labels found")
                return processed_data, labels, trial_info
            
            print(f"Found {len(ssvep_trials)} SSVEP trials with labels")
        else:
            print("No label column found in SSVEP trials")
            return processed_data, labels, trial_info
        
        # Limit trials per class if specified
        if max_trials_per_class is not None:
            try:
                ssvep_trials = ssvep_trials.groupby('label').head(max_trials_per_class)
            except KeyError:
                print("Warning: Could not group by label, processing all SSVEP trials")
        
        print(f"Processing {len(ssvep_trials)} SSVEP trials...")
        
        for idx, row in tqdm(ssvep_trials.iterrows(), total=len(ssvep_trials), desc="Processing trials"):
            try:
                # Load trial data
                raw_data = self.load_trial_data(row)
                
                if not raw_data.empty:
                    # Preprocess the signal
                    processed_trial = self.preprocess_signal(raw_data)
                    
                    processed_data.append(processed_trial)
                    labels.append(str(row['label']))
                    
                    trial_info.append({
                        'id': row['id'],
                        'subject_id': row['subject_id'],
                        'trial_session': row['trial_session'],
                        'trial': row['trial']
                    })
                    
            except Exception as e:
                print(f"Error processing trial {row['id']}: {e}")
                continue
        
        print(f"Successfully processed {len(processed_data)} trials")
        return processed_data, labels, trial_info
    
    def run_full_pipeline(self, max_trials_per_class=None):
        """Run complete SSVEP detection pipeline"""
        print("Starting SSVEP Detection Pipeline...")
        
        # Load and process training data
        print("\n1. Loading training data...")
        train_data, train_labels, train_info = self.load_and_process_dataset(
            self.train_df, max_trials_per_class
        )
        
        # Load and process validation data
        print("\n2. Loading validation data...")
        val_data, val_labels, val_info = self.load_and_process_dataset(
            self.validation_df, max_trials_per_class
        )
        
        # Load and process test data
        print("\n3. Loading test data...")
        test_data, test_labels, test_info = self.load_and_process_dataset(
            self.test_df, max_trials_per_class
        )
        
        if len(train_data) == 0:
            print("❌ No training data available!")
            return None
        
        # Train all classifiers
        print("\n4. Training classifiers...")
        self.train_cca_classifier(train_data, train_labels)
        self.train_psd_classifier(train_data, train_labels)
        self.train_riemann_classifier(train_data, train_labels)
        self.train_ensemble_classifier(train_data, train_labels)
        
        # Evaluate all classifiers
        print("\n5. Evaluating classifiers...")
        results = {}
        
        # CCA evaluation
        if self.cca_classifier:
            print("Evaluating CCA classifier...")
            train_pred_cca = self.predict_cca(train_data)
            val_pred_cca = self.predict_cca(val_data)
            test_pred_cca = self.predict_cca(test_data)
            
            results['cca'] = {
                'train_acc': accuracy_score(train_labels, train_pred_cca) if train_pred_cca else 0,
                'val_acc': accuracy_score(val_labels, val_pred_cca) if val_pred_cca else 0,
                'test_acc': accuracy_score(test_labels, test_pred_cca) if test_pred_cca else 0,
                'train_pred': train_pred_cca,
                'val_pred': val_pred_cca,
                'test_pred': test_pred_cca
            }
        
        # PSD evaluation
        if self.psd_classifier:
            print("Evaluating PSD classifier...")
            train_pred_psd = self.predict_psd(train_data)
            val_pred_psd = self.predict_psd(val_data)
            test_pred_psd = self.predict_psd(test_data)
            
            results['psd'] = {
                'train_acc': accuracy_score(train_labels, train_pred_psd) if train_pred_psd else 0,
                'val_acc': accuracy_score(val_labels, val_pred_psd) if val_pred_psd else 0,
                'test_acc': accuracy_score(test_labels, test_pred_psd) if test_pred_psd else 0,
                'train_pred': train_pred_psd,
                'val_pred': val_pred_psd,
                'test_pred': test_pred_psd
            }
        
        # Riemann evaluation
        if self.riemann_classifier:
            print("Evaluating Riemannian classifier...")
            train_pred_riemann = self.predict_riemann(train_data)
            val_pred_riemann = self.predict_riemann(val_data)
            test_pred_riemann = self.predict_riemann(test_data)
            
            results['riemann'] = {
                'train_acc': accuracy_score(train_labels, train_pred_riemann) if train_pred_riemann else 0,
                'val_acc': accuracy_score(val_labels, val_pred_riemann) if val_pred_riemann else 0,
                'test_acc': accuracy_score(test_labels, test_pred_riemann) if test_pred_riemann else 0,
                'train_pred': train_pred_riemann,
                'val_pred': val_pred_riemann,
                'test_pred': test_pred_riemann
            }
        
        # Ensemble evaluation
        if self.ensemble_classifier:
            print("Evaluating Ensemble classifier...")
            train_pred_ensemble = self.predict_ensemble(train_data)
            val_pred_ensemble = self.predict_ensemble(val_data)
            test_pred_ensemble = self.predict_ensemble(test_data)
            
            results['ensemble'] = {
                'train_acc': accuracy_score(train_labels, train_pred_ensemble) if train_pred_ensemble else 0,
                'val_acc': accuracy_score(val_labels, val_pred_ensemble) if val_pred_ensemble else 0,
                'test_acc': accuracy_score(test_labels, test_pred_ensemble) if test_pred_ensemble else 0,
                'train_pred': train_pred_ensemble,
                'val_pred': val_pred_ensemble,
                'test_pred': test_pred_ensemble
            }
        
        # Print results
        print("\n" + "="*60)
        print("SSVEP DETECTION RESULTS")
        print("="*60)
        
        for method, result in results.items():
            print(f"\n{method.upper()} Classifier:")
            print(f"  Training Accuracy: {result['train_acc']:.4f}")
            print(f"  Validation Accuracy: {result['val_acc']:.4f}")
            print(f"  Test Accuracy: {result['test_acc']:.4f}")
        
        # Find best method
        best_method = max(results.keys(), key=lambda x: results[x]['val_acc'])
        print(f"\nBest Method: {best_method.upper()} (Val Acc: {results[best_method]['val_acc']:.4f})")
        
        return {
            'results': results,
            'best_method': best_method,
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'train_labels': train_labels,
            'val_labels': val_labels,
            'test_labels': test_labels
        }

def main():
    # Initialize SSVEP detector
    print("Initializing SSVEP Detector...")
    detector = SSVEPDetector()
    
    # Run the complete pipeline
    print("\n" + "="*60)
    print("STARTING SSVEP DETECTION PIPELINE")
    print("="*60)
    
    try:
        results = detector.run_full_pipeline(
            max_trials_per_class=None  # Use all available trials
        )
        
        if results:
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            # Print detailed results
            best_method = results['best_method']
            best_results = results['results'][best_method]
            
            print(f"\nBEST PERFORMANCE ({best_method.upper()}):")
            print(f"- Training Accuracy: {best_results['train_acc']:.4f}")
            print(f"- Validation Accuracy: {best_results['val_acc']:.4f}")
            print(f"- Test Accuracy: {best_results['test_acc']:.4f}")
            
            # Print classification report for best method
            if best_results['val_pred']:
                print(f"\nValidation Classification Report ({best_method.upper()}):")
                print(classification_report(results['val_labels'], best_results['val_pred']))
            
            return results
        else:
            print("\nPipeline failed to complete!")
            return None
            
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main() 