import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.cross_decomposition import CCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import pickle
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

class SSVEPCCADetector:
    def __init__(self, base_path='mtc-aic3_dataset'):
        self.base_path = base_path
        self.sampling_rate = 250  # Hz
        self.trial_duration = 7  # seconds for SSVEP
        self.samples_per_trial = self.trial_duration * self.sampling_rate  # 1750 samples
        
        # SSVEP frequencies for each class
        self.frequency_map = {
            'Left': 10,      # 10 Hz
            'Right': 13,     # 13 Hz
            'Forward': 7,    # 7 Hz
            'Backward': 8    # 8 Hz
        }
        
        # Only use occipital channels for SSVEP detection
        self.occipital_channels = ['PO7', 'OZ', 'PO8']
        
        # CCA model
        self.cca_models = {}
        self.reference_signals = {}
        
        # Load the data index files
        self.train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
        self.validation_df = pd.read_csv(os.path.join(base_path, 'validation.csv'))
        
        # Filter for SSVEP data only
        self.ssvep_train = self.train_df[self.train_df['task'] == 'SSVEP'].copy()
        self.ssvep_validation = self.validation_df[self.validation_df['task'] == 'SSVEP'].copy()
        
        print(f"SSVEP Training samples: {len(self.ssvep_train)}")
        print(f"SSVEP Validation samples: {len(self.ssvep_validation)}")

    def load_trial_data(self, row):
        """Load EEG data for a specific trial"""
        # Determine dataset type based on ID range
        id_num = row['id']
        if id_num <= 4800:
            dataset = 'train'
        else:
            dataset = 'validation'
        
        # Construct the path to EEGdata.csv
        eeg_path = f"{self.base_path}/{row['task']}/{dataset}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"
        
        # Check if file exists
        if not os.path.exists(eeg_path):
            raise FileNotFoundError(f"EEG file not found: {eeg_path}")
        
        # Load the entire EEG file
        eeg_data = pd.read_csv(eeg_path)
        
        # Check if we have enough data
        if len(eeg_data) < self.samples_per_trial:
            raise ValueError(f"Not enough samples in EEG file: {len(eeg_data)} < {self.samples_per_trial}")
        
        # Calculate indices for the specific trial
        trial_num = int(row['trial'])
        start_idx = (trial_num - 1) * self.samples_per_trial
        end_idx = start_idx + self.samples_per_trial - 1
        
        # Check if trial indices are valid
        if end_idx >= len(eeg_data):
            raise ValueError(f"Trial {trial_num} exceeds available data: end_idx {end_idx} >= {len(eeg_data)}")
        
        # Extract the trial data
        trial_data = eeg_data.iloc[start_idx:end_idx+1]
        
        # Check if we have the required channels
        missing_channels = [ch for ch in self.occipital_channels if ch not in trial_data.columns]
        if missing_channels:
            raise ValueError(f"Missing channels in trial data: {missing_channels}")
        
        return trial_data

    def preprocess_eeg(self, eeg_data):
        """Apply preprocessing to EEG data"""
        # Extract only occipital channels
        occipital_data = eeg_data[self.occipital_channels].values
        
        # Remove DC mean
        occipital_data = occipital_data - np.mean(occipital_data, axis=0)
        
        # Apply notch filter at 50 Hz
        b_notch, a_notch = signal.iirnotch(50, 30, self.sampling_rate)
        for i in range(occipital_data.shape[1]):
            occipital_data[:, i] = signal.filtfilt(b_notch, a_notch, occipital_data[:, i])
        
        # Apply bandpass filter (1-60 Hz for SSVEP - include harmonics)
        nyquist = self.sampling_rate / 2
        low = 1 / nyquist
        high = 60 / nyquist
        b_bandpass, a_bandpass = butter(4, [low, high], btype='band')
        for i in range(occipital_data.shape[1]):
            occipital_data[:, i] = signal.filtfilt(b_bandpass, a_bandpass, occipital_data[:, i])
        
        return occipital_data

    def generate_filter_bank_reference_signals(self, frequency, duration=7):
        """Generate reference signals for Filter Bank CCA"""
        t = np.arange(0, duration, 1/self.sampling_rate)
        
        # Define frequency bands around the target frequency
        bands = [
            (frequency - 0.5, frequency + 0.5),  # Narrow band around fundamental
            (frequency * 2 - 1, frequency * 2 + 1),  # Around 2nd harmonic
            (frequency * 3 - 1.5, frequency * 3 + 1.5),  # Around 3rd harmonic
        ]
        
        all_references = []
        for band_low, band_high in bands:
            # Generate reference for this band
            ref_signals = []
            for h in range(1, 4):  # 3 harmonics
                sin_component = np.sin(2 * np.pi * frequency * h * t)
                cos_component = np.cos(2 * np.pi * frequency * h * t)
                ref_signals.extend([sin_component, cos_component])
            all_references.append(np.column_stack(ref_signals))
        
        return all_references

    def train_cca_models(self):
        """Train Filter Bank CCA models for each SSVEP class."""
        print("Training Filter Bank CCA models...")
        self.cca_models = {}
        self.reference_signals = {}
        
        for class_name, frequency in self.frequency_map.items():
            print(f"Training FBCCA for {class_name} ({frequency} Hz)...")
            
            # Generate filter bank reference signals
            filter_bank_refs = self.generate_filter_bank_reference_signals(frequency)
            self.reference_signals[class_name] = filter_bank_refs
            
            # Get training data for this class
            class_data = self.ssvep_train[self.ssvep_train['label'] == class_name]
            
            if len(class_data) == 0:
                print(f"Warning: No training data for class {class_name}")
                continue
            
            # Prepare EEG data for this class
            eeg_data_list = []
            for _, row in tqdm(class_data.iterrows(), total=len(class_data), desc=f"Loading {class_name}"):
                try:
                    trial_data = self.load_trial_data(row)
                    processed_eeg = self.preprocess_eeg(trial_data)
                    eeg_data_list.append(processed_eeg)
                except Exception as e:
                    print(f"Error loading trial {row['id']}: {e}")
                    continue
            
            if len(eeg_data_list) == 0:
                print(f"Warning: No valid EEG data for class {class_name}")
                continue
            
            # Train CCA models for each frequency band
            band_models = []
            for band_idx, ref_signals in enumerate(filter_bank_refs):
                # Stack all trials for this class
                all_eeg_data = np.vstack(eeg_data_list)
                all_ref_data = np.tile(ref_signals, (len(eeg_data_list), 1))
                
                # Create and fit CCA model for this band
                cca = CCA(n_components=min(3, len(ref_signals[0])))
                try:
                    cca.fit(all_eeg_data, all_ref_data)
                    band_models.append(cca)
                except Exception as e:
                    print(f"Error training CCA for {class_name} band {band_idx}: {e}")
                    band_models.append(None)
            
            self.cca_models[class_name] = band_models
            print(f"Successfully trained FBCCA for {class_name} using {len(eeg_data_list)} trials")
        
        print("Filter Bank CCA training completed!")

    def predict_single_trial(self, eeg_data):
        """Predict SSVEP class for a single trial using Filter Bank CCA."""
        processed_eeg = self.preprocess_eeg(eeg_data)
        correlations = {}
        
        for class_name, band_models in self.cca_models.items():
            filter_bank_refs = self.reference_signals[class_name]
            class_correlations = []
            
            # Compute correlation for each frequency band
            for band_idx, (cca_model, ref_signals) in enumerate(zip(band_models, filter_bank_refs)):
                if cca_model is None:
                    class_correlations.append(0.0)
                    continue
                
                try:
                    # Transform using the trained CCA model
                    eeg_transformed, ref_transformed = cca_model.transform(processed_eeg, ref_signals)
                    
                    # Calculate correlation between transformed signals
                    correlation = np.corrcoef(eeg_transformed[:, 0], ref_transformed[:, 0])[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                    class_correlations.append(abs(correlation))
                except Exception as e:
                    class_correlations.append(0.0)
            
            # Combine correlations from all bands (weighted average)
            if class_correlations:
                # Weight fundamental frequency more heavily
                weights = [0.5, 0.3, 0.2]  # Fundamental, 2nd harmonic, 3rd harmonic
                correlations[class_name] = np.average(class_correlations, weights=weights)
            else:
                correlations[class_name] = 0.0
        
        predicted_class = max(correlations, key=correlations.get)
        return predicted_class, correlations

    def evaluate_model(self, data_df, dataset_name="Validation"):
        """Evaluate the model on a dataset"""
        print(f"Evaluating model on {dataset_name} set...")
        
        predictions = []
        true_labels = []
        all_correlations = []
        
        for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc=f"Evaluating {dataset_name}"):
            try:
                # Load and predict
                trial_data = self.load_trial_data(row)
                predicted_class, correlations = self.predict_single_trial(trial_data)
                
                if predicted_class is not None:
                    predictions.append(predicted_class)
                    true_labels.append(row['label'])
                    all_correlations.append(correlations)
                else:
                    print(f"Failed to predict trial {row['id']}")
                    
            except Exception as e:
                print(f"Error processing trial {row['id']}: {e}")
                continue
        
        # Calculate accuracy
        if len(predictions) > 0:
            accuracy = accuracy_score(true_labels, predictions)
            
            # Generate detailed report
            report = classification_report(true_labels, predictions, output_dict=True)
            conf_matrix = confusion_matrix(true_labels, predictions, labels=list(self.frequency_map.keys()))
            
            print(f"{dataset_name} Accuracy: {accuracy:.4f}")
            
            return {
                'accuracy': accuracy,
                'predictions': predictions,
                'true_labels': true_labels,
                'classification_report': report,
                'confusion_matrix': conf_matrix,
                'correlations': all_correlations
            }
        else:
            print(f"No valid predictions for {dataset_name} set")
            return None

    def save_model(self, filename='ssvep_cca_model.pkl'):
        """Save the trained model"""
        model_data = {
            'cca_models': self.cca_models,
            'reference_signals': self.reference_signals,
            'frequency_map': self.frequency_map,
            'sampling_rate': self.sampling_rate,
            'occipital_channels': self.occipital_channels
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")

    def load_model(self, filename='ssvep_cca_model.pkl'):
        """Load a trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.cca_models = model_data['cca_models']
        self.reference_signals = model_data['reference_signals']
        self.frequency_map = model_data['frequency_map']
        self.sampling_rate = model_data['sampling_rate']
        self.occipital_channels = model_data['occipital_channels']
        
        print(f"Model loaded from {filename}")

    def generate_reports(self, validation_results, report_dir='reports'):
        """Generate detailed reports and visualizations"""
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save classification report
        report_file = os.path.join(report_dir, f'classification_report_{timestamp}.txt')
        with open(report_file, 'w') as f:
            f.write("SSVEP CCA Detection Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Accuracy: {validation_results['accuracy']:.4f}\n\n")
            f.write("Detailed Classification Report:\n")
            f.write(classification_report(validation_results['true_labels'], 
                                        validation_results['predictions']))
        
        # 2. Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        cm = validation_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.frequency_map.keys()),
                   yticklabels=list(self.frequency_map.keys()))
        plt.title('Confusion Matrix - SSVEP CCA Detection')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, f'confusion_matrix_{timestamp}.png'), dpi=300)
        plt.close()
        
        # 3. Create correlation analysis plot
        plt.figure(figsize=(12, 8))
        correlations_df = pd.DataFrame(validation_results['correlations'])
        
        # Box plot of correlations for each class
        correlations_df.boxplot(figsize=(10, 6))
        plt.title('Correlation Coefficients by SSVEP Class')
        plt.ylabel('Correlation Coefficient')
        plt.xlabel('SSVEP Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, f'correlation_analysis_{timestamp}.png'), dpi=300)
        plt.close()
        
        # 4. Create detailed results summary
        summary_file = os.path.join(report_dir, f'results_summary_{timestamp}.txt')
        with open(summary_file, 'w') as f:
            f.write("SSVEP CCA Detection - Results Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Overall Accuracy: {validation_results['accuracy']:.4f}\n\n")
            
            f.write("Frequency Mapping:\n")
            for class_name, freq in self.frequency_map.items():
                f.write(f"  {class_name}: {freq} Hz\n")
            f.write("\n")
            
            f.write("Per-Class Performance:\n")
            report = validation_results['classification_report']
            for class_name in self.frequency_map.keys():
                if class_name in report:
                    f.write(f"  {class_name}:\n")
                    f.write(f"    Precision: {report[class_name]['precision']:.4f}\n")
                    f.write(f"    Recall: {report[class_name]['recall']:.4f}\n")
                    f.write(f"    F1-Score: {report[class_name]['f1-score']:.4f}\n")
                    f.write(f"    Support: {report[class_name]['support']}\n\n")
            
            f.write("Model Configuration:\n")
            f.write(f"  Sampling Rate: {self.sampling_rate} Hz\n")
            f.write(f"  Trial Duration: {self.trial_duration} seconds\n")
            f.write(f"  Channels Used: {', '.join(self.occipital_channels)}\n")
            f.write(f"  Preprocessing: DC removal, 50Hz notch, 1-60Hz bandpass\n")
            f.write(f"  CCA Components: 3 (minimum of channels and reference harmonics)\n")
        
        print(f"Reports generated in {report_dir}/")
        print(f"Files created:")
        print(f"  - {report_file}")
        print(f"  - {summary_file}")
        print(f"  - confusion_matrix_{timestamp}.png")
        print(f"  - correlation_analysis_{timestamp}.png")

def main():
    # Create SSVEP detector
    detector = SSVEPCCADetector()
    
    # Train CCA models
    detector.train_cca_models()
    
    # Evaluate on validation set
    validation_results = detector.evaluate_model(detector.ssvep_validation, "Validation")
    
    if validation_results:
        # Generate reports
        detector.generate_reports(validation_results)
        
        # Save the trained model
        detector.save_model('ssvep_cca_model.pkl')
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Validation Accuracy: {validation_results['accuracy']:.4f}")
        print("Model saved as 'ssvep_cca_model.pkl'")
        print("Reports generated in 'reports/' folder")
        print("="*50)
    else:
        print("Training failed - no valid results obtained")

if __name__ == "__main__":
    main() 