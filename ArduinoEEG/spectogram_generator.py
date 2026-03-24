import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Load windowed data
X = np.load("eeg_windows.npy")
y = np.load("labels.npy")

SAMPLE_RATE = 250  # Hz

# Output directories
base_dir = "spectrograms"
encoding_dir = os.path.join(base_dir, "encoding")
decoding_dir = os.path.join(base_dir, "decoding")

os.makedirs(encoding_dir, exist_ok=True)
os.makedirs(decoding_dir, exist_ok=True)

for i, window in enumerate(X):
    label = y[i]

    # Generate spectrogram
    freqs, times, Sxx = spectrogram(
        window,
        fs=SAMPLE_RATE,
        nperseg=min(8, len(window)),
        noverlap=4
    )

    # Plot and save as image
    plt.figure(figsize=(2, 2))
    plt.pcolormesh(times, freqs, Sxx, shading='gouraud')
    plt.axis('off')

    if label == "encoding":
        save_path = os.path.join(encoding_dir, f"enc_{i}.png")
    else:
        save_path = os.path.join(decoding_dir, f"dec_{i}.png")

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

print("Spectrogram images generated successfully.")
