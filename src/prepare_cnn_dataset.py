import os
import shutil
import pandas as pd

# Paths
CSV_PATH = r"C:\projects\prosthetic_hippocampal_using_cnn\features\eeg_band_features.csv"
OUTPUT_DIR = r"C:\projects\prosthetic_hippocampal_using_cnn\outputs"
CNN_DATASET_DIR = r"C:\projects\prosthetic_hippocampal_using_cnn\cnn_dataset"

# Create CNN dataset folders
encode_dir = os.path.join(CNN_DATASET_DIR, "encode")
recall_dir = os.path.join(CNN_DATASET_DIR, "recall")

os.makedirs(encode_dir, exist_ok=True)
os.makedirs(recall_dir, exist_ok=True)

def map_wave_to_label(wave):
    """Map dominant EEG wave → class label."""
    wave = wave.lower()

    if wave in ["theta", "gamma"]:
        return "encode"
    elif wave in ["alpha", "beta"]:
        return "recall"
    else:
        return None


def main():
    df = pd.read_csv(CSV_PATH)

    print("Loaded CSV with", len(df), "rows.")

    for idx, row in df.iterrows():
        recording = row["recording"]           # e.g., Subject01_1
        dominant_wave = row["dominant_wave"]   # e.g., theta

        label = map_wave_to_label(dominant_wave)

        if label is None:
            print(f"⚠ Skipping {recording}: Unknown wave '{dominant_wave}'")
            continue

        png_name = f"{recording}.png"
        src_path = os.path.join(OUTPUT_DIR, png_name)

        if not os.path.exists(src_path):
            print(f"⚠ Missing image: {src_path}")
            continue

        # Destination path
        dst_folder = encode_dir if label == "encode" else recall_dir
        dst_path = os.path.join(dst_folder, png_name)

        shutil.copy(src_path, dst_path)
        print(f"✔ {png_name} → {label}")

    print("\n🎉 CNN Dataset prepared successfully!")
    print(f"  Encode images → {encode_dir}")
    print(f"  Recall images → {recall_dir}")


if __name__ == "__main__":
    main()
