import pandas as pd
import os

def test_data_loading():
    base_path = 'mtc-aic3_dataset'
    
    # Load the data index files
    train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
    validation_df = pd.read_csv(os.path.join(base_path, 'validation.csv'))
    
    # Filter for SSVEP data only
    ssvep_train = train_df[train_df['task'] == 'SSVEP'].copy()
    ssvep_validation = validation_df[validation_df['task'] == 'SSVEP'].copy()
    
    print(f"SSVEP Training samples: {len(ssvep_train)}")
    print(f"SSVEP Validation samples: {len(ssvep_validation)}")
    
    # Check class distribution
    print("\nTraining class distribution:")
    print(ssvep_train['label'].value_counts())
    
    print("\nValidation class distribution:")
    print(ssvep_validation['label'].value_counts())
    
    # Test loading a few trials
    print("\nTesting data loading...")
    
    for i, (_, row) in enumerate(ssvep_train.head(3).iterrows()):
        print(f"\nTrial {i+1}: ID={row['id']}, Class={row['label']}")
        
        # Determine dataset type
        id_num = row['id']
        if id_num <= 4800:
            dataset = 'train'
        else:
            dataset = 'validation'
        
        # Construct path
        eeg_path = f"{base_path}/{row['task']}/{dataset}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"
        print(f"Path: {eeg_path}")
        
        # Check if file exists
        if os.path.exists(eeg_path):
            print(f"✓ File exists")
            
            # Load data
            eeg_data = pd.read_csv(eeg_path)
            print(f"✓ Loaded EEG data: {eeg_data.shape}")
            print(f"  Columns: {list(eeg_data.columns)}")
            
            # Check for required channels
            required_channels = ['PO7', 'OZ', 'PO8']
            missing_channels = [ch for ch in required_channels if ch not in eeg_data.columns]
            if missing_channels:
                print(f"✗ Missing channels: {missing_channels}")
            else:
                print(f"✓ All required channels present")
            
            # Check trial extraction
            trial_duration = 7  # seconds
            sampling_rate = 250  # Hz
            samples_per_trial = trial_duration * sampling_rate  # 1750
            
            trial_num = int(row['trial'])
            start_idx = (trial_num - 1) * samples_per_trial
            end_idx = start_idx + samples_per_trial - 1
            
            print(f"  Trial {trial_num}: samples {start_idx} to {end_idx}")
            
            if end_idx < len(eeg_data):
                trial_data = eeg_data.iloc[start_idx:end_idx+1]
                print(f"✓ Trial data extracted: {trial_data.shape}")
                
                # Check data values
                occipital_data = trial_data[required_channels].values
                print(f"  Occipital data shape: {occipital_data.shape}")
                print(f"  Data range: {occipital_data.min():.2f} to {occipital_data.max():.2f}")
                print(f"  Data mean: {occipital_data.mean():.2f}")
            else:
                print(f"✗ Trial indices exceed data length")
        else:
            print(f"✗ File not found")

if __name__ == "__main__":
    test_data_loading() 