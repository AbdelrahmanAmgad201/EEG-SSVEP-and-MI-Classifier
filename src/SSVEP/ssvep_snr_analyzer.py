import pandas as pd
import numpy as np
from scipy import signal
from amir_gui import EEGSignalProcessor # Reuse the processor
import os
from tqdm import tqdm

# Class to Frequency Mapping
SSVEP_FREQUENCIES = {
    'Left': 10,
    'Right': 13,
    'Forward': 7,
    'Backward': 8
}

# Initialize the EEGSignalProcessor outside the function for caching benefits
processor = EEGSignalProcessor()

def calculate_snr(eeg_signal, target_frequency, sampling_rate, bandwidth=1.0, noise_band_offset=1.0):
    """
    Calculate Signal-to-Noise Ratio (SNR) for a given EEG signal at a target frequency.
    SNR is defined as power at target frequency / mean power in neighboring noise bands.
    
    Args:
        eeg_signal (np.array): 1D array of EEG data for a single channel.
        target_frequency (float): The specific frequency to measure the signal power.
        sampling_rate (int): Sampling rate of the EEG signal.
        bandwidth (float): Bandwidth around the target frequency for the signal power (e.g., +/- 0.5 Hz).
        noise_band_offset (float): Offset from target frequency to define noise bands (e.g., 1.0 Hz).
                                   Noise bands will be [target - offset - bw/2, target - offset + bw/2]
                                   and [target + offset - bw/2, target + offset + bw/2].
    
    Returns:
        float: The calculated SNR in dB.
    """
    if len(eeg_signal) < 256: # Welch needs a reasonable segment length
        return np.nan # Not enough data for reliable PSD

    freqs, psd = signal.welch(eeg_signal, fs=sampling_rate, nperseg=256, noverlap=128)

    # Find index of target frequency
    target_idx = np.argmin(np.abs(freqs - target_frequency))
    
    # Define signal band (e.g., target_frequency +/- 0.5 Hz)
    signal_band_indices = np.where(
        (freqs >= target_frequency - bandwidth/2) & 
        (freqs <= target_frequency + bandwidth/2)
    )[0]
    
    # Ensure signal band indices are valid
    if len(signal_band_indices) == 0:
        return np.nan
        
    signal_power = np.mean(psd[signal_band_indices])

    # Define noise bands (e.g., [target-offset-bw/2, target-offset+bw/2] and [target+offset-bw/2, target+offset+bw/2])
    # Exclude the target frequency range itself
    noise_band1_indices = np.where(
        (freqs >= target_frequency - noise_band_offset - bandwidth/2) &
        (freqs <= target_frequency - noise_band_offset + bandwidth/2)
    )[0]
    noise_band2_indices = np.where(
        (freqs >= target_frequency + noise_band_offset - bandwidth/2) &
        (freqs <= target_frequency + noise_band_offset + bandwidth/2)
    )[0]
    
    all_noise_indices = np.concatenate((noise_band1_indices, noise_band2_indices))
    all_noise_indices = np.unique(all_noise_indices) # Remove duplicates
    
    # Remove any overlap with signal band (though offsets should prevent this if chosen carefully)
    all_noise_indices = [idx for idx in all_noise_indices if idx not in signal_band_indices]

    if len(all_noise_indices) == 0 or np.sum(psd[all_noise_indices]) == 0:
        noise_power = 1e-9 # Avoid division by zero
    else:
        noise_power = np.mean(psd[all_noise_indices])
    
    snr_val = signal_power / noise_power
    return 10 * np.log10(snr_val) # Convert to dB

def evaluate_preprocessing_strategy(trial_row, strategy_name, auto_ica_threshold=None):
    """
    Evaluates a single preprocessing strategy for a given trial and returns SNR.
    """
    raw_data_df = processor.load_trial_data(trial_row)
    eeg_data_np = raw_data_df[processor.eeg_channels].values
    target_frequency = SSVEP_FREQUENCIES[trial_row['label']]
    
    processed_signal = None
    channels_to_evaluate = []

    if strategy_name == "No Preprocessing":
        processed_signal = eeg_data_np
        channels_to_evaluate = processor.ssvep_channels # Evaluate on SSVEP channels
    elif strategy_name == "Auto ICA":
        # Process with auto ICA
        processed_signal = processor.process_signal(
            eeg_data_np, 
            auto_remove_artifacts=True,
            # If auto_ica_threshold is None, default from processor will be used (85%)
            # Otherwise, use the provided one.
            # Note: The process_signal does not directly take threshold. 
            # It is handled internally by auto_detect_bad_components. 
            # We'll rely on the default set in the GUI, or use a specific test run for different thresholds.
        )
        channels_to_evaluate = processor.ssvep_channels # Evaluate on SSVEP channels after ICA
    elif strategy_name == "Average All Channels":
        processed_signal = np.mean(eeg_data_np, axis=1, keepdims=True) # Average across channels
        channels_to_evaluate = [0] # Single averaged channel
    elif strategy_name == "Average SSVEP Channels":
        ssvep_channel_indices = [processor.eeg_channels.index(ch) for ch in processor.ssvep_channels]
        processed_signal = np.mean(eeg_data_np[:, ssvep_channel_indices], axis=1, keepdims=True)
        channels_to_evaluate = [0] # Single averaged channel
    elif strategy_name == "Single Channel (OZ)":
        oz_idx = processor.eeg_channels.index('OZ')
        processed_signal = eeg_data_np[:, oz_idx:oz_idx+1]
        channels_to_evaluate = [0] # Single channel
    
    snrs = []
    for ch_name_or_idx in channels_to_evaluate:
        if isinstance(ch_name_or_idx, str):
            ch_idx = processor.eeg_channels.index(ch_name_or_idx)
        else:
            ch_idx = ch_name_or_idx # It's already an index for averaged or single channel
            
        channel_data = processed_signal[:, ch_idx]
        snr = calculate_snr(channel_data, target_frequency, processor.sampling_rate)
        if not np.isnan(snr):
            snrs.append(snr)
            
    if snrs:        
        return np.mean(snrs)
    return np.nan # No valid SNR

def run_snr_comparison_simulation(num_trials=20):
    """
    Runs a simulation to compare SNR for different preprocessing strategies.
    """
    print(f"\n🔬 Running SNR Comparison Simulation ({num_trials} SSVEP validation trials)")
    print("=" * 70)

    # Filter for SSVEP validation trials
    ssvep_val_df = processor.validation_df[processor.validation_df['task'] == 'SSVEP'].reset_index(drop=True)
    
    if len(ssvep_val_df) == 0:
        print("No SSVEP trials found in validation set. Please check dataset.")
        return

    # Select a random subset of trials
    selected_trials = ssvep_val_df.sample(min(num_trials, len(ssvep_val_df)), random_state=42)

    strategies = [
        "No Preprocessing", 
        "Auto ICA", 
        "Average All Channels", 
        "Average SSVEP Channels",
        "Single Channel (OZ)"
    ]

    results = {strategy: [] for strategy in strategies}

    for idx, trial_row in tqdm(selected_trials.iterrows(), total=len(selected_trials), desc="Processing Trials"):
        for strategy in strategies:
            snr = evaluate_preprocessing_strategy(trial_row, strategy)
            if not np.isnan(snr):
                results[strategy].append(snr)
    
    print("\n--- SNR Results (Mean ± Std Dev dB) ---")
    best_strategy = ""
    best_snr = -np.inf

    for strategy, snr_list in results.items():
        if snr_list:
            mean_snr = np.mean(snr_list)
            std_snr = np.std(snr_list)
            print(f"{strategy:<25}: {mean_snr:.2f} ± {std_snr:.2f} dB")
            if mean_snr > best_snr:
                best_snr = mean_snr
                best_strategy = strategy
        else:
            print(f"{strategy:<25}: No valid SNR calculated")
            
    print(f"\n🏆 Best Preprocessing Strategy: {best_strategy} (Avg SNR: {best_snr:.2f} dB)")
    print("=" * 70)
    
    # Optional: Analyze specific artifact components if Auto ICA was run
    if "Auto ICA" in strategies and processor.ica_scores is not None:
        print("\n--- Auto ICA Components (Example from last processed trial) ---")
        for i in range(processor.ica_components.shape[1]):
            print(f"IC {i+1}: E_blink={processor.ica_scores['eye_blink'][i]:.2f}, "
                  f"Muscle={processor.ica_scores['muscle'][i]:.2f}, "
                  f"Line Noise={processor.ica_scores['line_noise'][i]:.2f}, "
                  f"Kurtosis={processor.ica_scores['general_artifact'][i]:.2f}")

if __name__ == "__main__":
    run_snr_comparison_simulation(num_trials=50) # You can adjust the number of trials 