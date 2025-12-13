import os
import mne
import matplotlib.pyplot as plt

# ------------ CONFIG -------------
DATASET_ROOT = r"C:\projects\prosthetic_hippocampal_using_cnn\eegmmidb"
OUTPUT_DIR = r"C:\projects\prosthetic_hippocampal_using_cnn\outputs"
# ---------------------------------


def get_all_edf_files(root_dir):
    edf_files = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".edf"):
                edf_files.append(os.path.join(root, f))
    return edf_files


def load_eeg(path):
    print(f"\nProcessing: {path}")
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.pick_types(eeg=True)  # pick EEG channels only
    return raw


def preprocess_raw(raw, l_freq=1, h_freq=45):
    print("Applying band-pass filter 1–45 Hz ...")
    return raw.copy().filter(l_freq, h_freq, verbose=False)


def plot_eeg_image(raw, save_path):
    """
    Save a clean EEG plot (one channel) suitable for ML training.
    """
    channel = raw.ch_names[0]  # auto-select first EEG channel
    print(f"Using channel: {channel}")

    data, times = raw[channel][0], raw.times

    plt.figure(figsize=(10, 3))
    plt.plot(times[:2000], data[0][:2000])  # plot first 2 seconds
    plt.title(f"EEG Signal - {channel}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved image: {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    edf_files = get_all_edf_files(DATASET_ROOT)
    print(f"\nFound {len(edf_files)} EDF files.\n")

    for edf in edf_files:
        try:
            raw = load_eeg(edf)
            raw = preprocess_raw(raw)

            filename = os.path.basename(edf).replace(".edf", ".png")
            save_path = os.path.join(OUTPUT_DIR, filename)

            plot_eeg_image(raw, save_path)

        except Exception as e:
            print(f"Error processing {edf}: {e}")


if __name__ == "__main__":
    main()
