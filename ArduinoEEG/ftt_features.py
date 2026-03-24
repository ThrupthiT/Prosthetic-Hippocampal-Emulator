import numpy as np
import pandas as pd

# Load windowed data
X = np.load("eeg_windows.npy")
y = np.load("labels.npy")
subjects = np.load("subjects.npy")

# Parameters
SAMPLE_RATE = 250  # Hz
WINDOW_SIZE = X.shape[1]

def band_power(signal, low, high, fs):
    freqs = np.fft.rfftfreq(len(signal), d=1/fs)
    fft_vals = np.abs(np.fft.rfft(signal)) ** 2
    mask = (freqs >= low) & (freqs <= high)
    return np.sum(fft_vals[mask])

rows = []

for i, window in enumerate(X):
    delta = band_power(window, 0.5, 4, SAMPLE_RATE)
    theta = band_power(window, 4, 8, SAMPLE_RATE)
    alpha = band_power(window, 8, 13, SAMPLE_RATE)
    beta  = band_power(window, 13, 30, SAMPLE_RATE)

    powers = {
        "delta": delta,
        "theta": theta,
        "alpha": alpha,
        "beta": beta
    }

    dominant = max(powers, key=powers.get)

    rows.append([
        subjects[i],
        delta,
        theta,
        alpha,
        beta,
        dominant,
        y[i]
    ])

# Save to CSV
df = pd.DataFrame(rows, columns=[
    "subject_id", "delta", "theta", "alpha", "beta", "dominant_wave", "label"
])

df.to_csv("eeg_frequency_features.csv", index=False)

print("Saved eeg_frequency_features.csv")
