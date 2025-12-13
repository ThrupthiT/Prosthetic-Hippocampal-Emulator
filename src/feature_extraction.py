import os
import pandas as pd
import numpy as np
import mne

DATASET_ROOT = r"C:\projects\prosthetic_hippocampal_using_cnn\eegmmidb"
OUTPUT_CSV = r"C:\projects\prosthetic_hippocampal_using_cnn\features\eeg_band_features.csv"

# EEG frequency bands
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}


def compute_band_power(raw, fmin, fmax):
    """Compute average band power using PSD from raw EEG."""
    psd, freqs = mne.time_frequency.psd_array_welch(
        raw.get_data(),
        sfreq=raw.info['sfreq'],
        fmin=fmin,
        fmax=fmax,
        n_fft=1024,
        verbose=False
    )
    return psd.mean()


def extract_features_from_edf(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    raw.filter(1, 45, verbose=False)

    features = {}
    for band, (fmin, fmax) in BANDS.items():
        features[band] = compute_band_power(raw, fmin, fmax)

    return features


def main():
    rows = []

    for root, dirs, files in os.walk(DATASET_ROOT):
        for file in files:
            if file.endswith(".edf"):
                edf_path = os.path.join(root, file)

                subject = os.path.basename(root)
                recording = file.replace(".edf", "")

                print(f"Processing {edf_path} ...")

                feats = extract_features_from_edf(edf_path)
                feats["subject"] = subject
                feats["recording"] = recording

                # determine dominant wave
                feats["dominant_wave"] = max(BANDS.keys(), key=lambda b: feats[b])

                rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFeature CSV saved → {OUTPUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()
