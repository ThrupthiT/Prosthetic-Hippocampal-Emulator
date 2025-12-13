import pandas as pd

FEATURE_CSV = r"C:\projects\prosthetic_hippocampal_using_cnn\features\eeg_band_features.csv"

def compute_dominant_wave(row):
    waves = {
        "theta": row["theta"],
        "alpha": row["alpha"],
        "beta": row["beta"],
        "gamma": row["gamma"],
    }
    return max(waves, key=waves.get)

def main():
    df = pd.read_csv(FEATURE_CSV)

    # Create new label column
    df["dominant_wave"] = df.apply(compute_dominant_wave, axis=1)

    # Save updated CSV
    df.to_csv(FEATURE_CSV, index=False)

    print("dominant_wave column added successfully!")
    print(df.head())

if __name__ == "__main__":
    main()
