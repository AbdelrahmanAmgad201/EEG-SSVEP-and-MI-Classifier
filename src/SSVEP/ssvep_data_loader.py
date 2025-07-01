import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class SSVEPDataset(Dataset):
    """
    Custom Dataset for SSVEP data from MTC-AIC3 dataset
    """
    def __init__(self, csv_file, base_path='mtc-aic3_dataset', transform=None, 
                 eeg_channels=None, trial_duration=7, sampling_rate=250):
        """
        Args:
            csv_file (str): Path to the CSV file (train.csv, validation.csv, test.csv)
            base_path (str): Base path to the dataset directory
            transform: Optional transform to be applied on EEG data
            eeg_channels (list): List of EEG channels to use (default: all 8 channels)
            trial_duration (int): Duration of each trial in seconds
            sampling_rate (int): Sampling rate in Hz
        """
        self.base_path = base_path
        self.transform = transform
        self.trial_duration = trial_duration
        self.sampling_rate = sampling_rate
        self.samples_per_trial = trial_duration * sampling_rate
        
        # Load the CSV file
        self.data_df = pd.read_csv(csv_file)
        
        # Filter for SSVEP data only
        self.data_df = self.data_df[self.data_df['task'] == 'SSVEP'].reset_index(drop=True)
        
        # Define EEG channels (all 8 channels from the dataset)
        if eeg_channels is None:
            self.eeg_channels = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
        else:
            self.eeg_channels = eeg_channels
            
        # Create label mapping
        self.label_mapping = {
            'Left': 0,
            'Right': 1, 
            'Forward': 2,
            'Backward': 3
        }
        
        # Cache for loaded EEG files to avoid repeated disk I/O
        self.eeg_cache = {}
        
        print(f"Loaded {len(self.data_df)} SSVEP trials")
        print(f"Using {len(self.eeg_channels)} EEG channels: {self.eeg_channels}")
        
    def __len__(self):
        return len(self.data_df)
    
    def _load_eeg_file(self, subject_id, task, dataset, trial_session):
        """Load EEG file with caching"""
        cache_key = f"{subject_id}_{task}_{dataset}_{trial_session}"
        
        if cache_key not in self.eeg_cache:
            eeg_path = f"{self.base_path}/{task}/{dataset}/{subject_id}/{trial_session}/EEGdata.csv"
            
            if not os.path.exists(eeg_path):
                raise FileNotFoundError(f"EEG file not found: {eeg_path}")
                
            eeg_data = pd.read_csv(eeg_path)
            self.eeg_cache[cache_key] = eeg_data
            
        return self.eeg_cache[cache_key]
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
        # Determine dataset type based on ID
        id_num = row['id']
        if id_num <= 4800:
            dataset = 'train'
        elif id_num <= 4900:
            dataset = 'validation'
        else:
            dataset = 'test'
        
        # Load EEG data
        eeg_data = self._load_eeg_file(
            row['subject_id'], 
            row['task'], 
            dataset, 
            row['trial_session']
        )
        
        # Extract trial data
        trial_num = int(row['trial'])
        start_idx = (trial_num - 1) * self.samples_per_trial
        end_idx = start_idx + self.samples_per_trial - 1
        
        trial_data = eeg_data.iloc[start_idx:end_idx+1]
        
        # Extract EEG channels
        eeg_trial = trial_data[self.eeg_channels].values.astype(np.float32)
        
        # Apply transform if specified
        if self.transform:
            eeg_trial = self.transform(eeg_trial)
        
        # Convert to tensor - ensure shape is [channels, time]
        eeg_tensor = torch.FloatTensor(eeg_trial)
        
        # Ensure correct shape: [channels, time]
        if len(eeg_tensor.shape) == 2:
            # If shape is [time, channels], transpose to [channels, time]
            if eeg_tensor.shape[0] > eeg_tensor.shape[1]:
                eeg_tensor = eeg_tensor.T
        else:
            # If somehow we have a different shape, reshape appropriately
            eeg_tensor = eeg_tensor.view(len(self.eeg_channels), -1)
        
        # Handle label
        if 'label' in row:
            label = self.label_mapping[row['label']]
            label_tensor = torch.LongTensor([label])
            return eeg_tensor, label_tensor
        else:
            # For test data without labels
            return eeg_tensor, torch.LongTensor([-1])  # Placeholder label

class EEGPreprocessor:
    """
    Preprocessing pipeline for EEG data
    """
    def __init__(self, sampling_rate=250, notch_freq=50, bandpass_freq=(0.5, 100)):
        self.sampling_rate = sampling_rate
        self.notch_freq = notch_freq
        self.bandpass_freq = bandpass_freq
        self.scaler = StandardScaler()
        
    def __call__(self, eeg_data):
        """
        Apply preprocessing pipeline
        Args:
            eeg_data: numpy array of shape [channels, time_samples]
        Returns:
            Preprocessed EEG data
        """
        # Ensure correct shape
        if len(eeg_data.shape) == 2 and eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T  # [time, channels] -> [channels, time]
        
        # Apply bandpass filter (simple implementation)
        eeg_filtered = self._apply_bandpass(eeg_data)
        
        # Apply notch filter for power line noise
        eeg_filtered = self._apply_notch(eeg_filtered)
        
        # Normalize each channel
        eeg_normalized = self._normalize_channels(eeg_filtered)
        
        return eeg_normalized
    
    def _apply_bandpass(self, eeg_data):
        """Simple bandpass filter using FFT"""
        from scipy import signal
        
        # Design bandpass filter
        nyquist = self.sampling_rate / 2
        low = self.bandpass_freq[0] / nyquist
        high = self.bandpass_freq[1] / nyquist
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            filtered_data[ch] = signal.filtfilt(
                *signal.butter(4, [low, high], btype='band'),
                eeg_data[ch]
            )
        
        return filtered_data
    
    def _apply_notch(self, eeg_data):
        """Notch filter for power line noise"""
        from scipy import signal
        
        # Design notch filter
        nyquist = self.sampling_rate / 2
        notch_freq = self.notch_freq / nyquist
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            filtered_data[ch] = signal.filtfilt(
                *signal.iirnotch(notch_freq, 30),
                eeg_data[ch]
            )
        
        return filtered_data
    
    def _normalize_channels(self, eeg_data):
        """Normalize each channel independently"""
        normalized_data = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            channel_data = eeg_data[ch]
            normalized_data[ch] = (channel_data - np.mean(channel_data)) / (np.std(channel_data) + 1e-8)
        
        return normalized_data

def create_ssvep_dataloaders(train_csv, val_csv, test_csv, base_path='mtc-aic3_dataset',
                           batch_size=32, num_workers=0, preprocess=True):
    """
    Create DataLoaders for SSVEP training, validation, and testing
    
    Args:
        train_csv (str): Path to training CSV
        val_csv (str): Path to validation CSV  
        test_csv (str): Path to test CSV
        base_path (str): Base path to dataset
        batch_size (int): Batch size for DataLoaders
        num_workers (int): Number of workers for DataLoader
        preprocess (bool): Whether to apply preprocessing
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    # Create preprocessor if needed
    transform = EEGPreprocessor() if preprocess else None
    
    # Create datasets
    train_dataset = SSVEPDataset(train_csv, base_path, transform=transform)
    val_dataset = SSVEPDataset(val_csv, base_path, transform=transform)
    test_dataset = SSVEPDataset(test_csv, base_path, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created DataLoaders:")
    print(f"  Training: {len(train_loader)} batches")
    print(f"  Validation: {len(val_loader)} batches") 
    print(f"  Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the data loader
    base_path = 'mtc-aic3_dataset'
    train_csv = os.path.join(base_path, 'train.csv')
    val_csv = os.path.join(base_path, 'validation.csv')
    test_csv = os.path.join(base_path, 'test.csv')
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_ssvep_dataloaders(
        train_csv, val_csv, test_csv, base_path, batch_size=8
    )
    
    # Test a batch
    for batch_idx, (eeg_data, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  EEG data shape: {eeg_data.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Label values: {labels.squeeze().numpy()}")
        break 