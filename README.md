# EEG-SSVEP-and-MI-Classifier
![Cover Image](images/cover.jpg)

This repository contains the official submission of the **MindCloud** team for the **MTC-AIC 3** competition. It includes all the source code, documentation, and model checkpoints for our EEG signal classifier designed for both SSVEP and MI tasks.

## Installation

### Clone the repository
```bash
git clone git@github.com:AbdelrahmanAmgad201/EEG-SSVEP-and-MI-Classifier.git
cd EEG-SSVEP-and-MI-Classifier
```

## Preprocessing
## MI (Motor Imagery)
Extensive research and analysis led us to select the C3, C4, and Cz channels for the Motor Imagery (MI) task, as these electrodes are closely associated with sensorimotor activity in the brain and are known to capture MI-related signals effectively.

The preprocessing pipeline involved several critical steps to ensure clean and informative EEG data:

### Frequency Domain Transformation
The raw EEG signals were transformed into the frequency domain using the Fast Fourier Transform (FFT). This allowed us to better isolate and process relevant frequency components.

### DC Offset Removal
To eliminate the baseline drift, we subtracted the mean (DC component) from each channel, effectively centering the signals.

### Notch Filtering (48–52 Hz)
A notch filter was applied in the 48–52 Hz range to suppress power-line interference and eliminate electrical noise commonly present in EEG recordings.

### Band-Pass Filtering (8–30 Hz)
We applied a band-pass filter to isolate the 8–30 Hz frequency band, which includes the mu (8–12 Hz) and beta (13–30 Hz) rhythms—key frequency components associated with motor imagery activity.

### Artifact Removal using ICA
Independent Component Analysis (ICA) was used to identify and remove artifacts such as eye blinks, muscle activity, and other noise sources. This significantly improved the signal quality and helped retain only the neural components relevant to the MI task.

## Chosen Architecture
## Results
## Challenges and Insights

## Other Contributions
### Signal visualiser
We developed a signal visualizer GUI to support data exploration and preprocessing. You can find it [here](https://github.com/AmirKaseb/EEG-GUI)
### SSVEP Simulator
We developed and hosted a real-time SSVEP simulator that can be configured and used for further real-time testing. It is available [here](https://gilded-kitsune-dfc3ec.netlify.app/)
## Future Plans
We have been exploring the possibility of acquiring a EEG signal headset to facilitate real-time testing and improving model accuracy and robustness to reach industry level performance.

## References

