import pandas as pd
import numpy as np

# Load raw EEG CSV
df = pd.read_csv("sample_eeg_arduino_data.csv")

# Parameters
SAMPLE_RATE = 250       # Hz
WINDOW_SIZE = 10       # samples (1 second)
OVERLAP = 5           # 50% overlap
STEP = WINDOW_SIZE - OVERLAP

windows = []
labels = []
subjects = []

# Group by subject and task to avoid mixing
for (subject, task), group in df.groupby(["subject_id", "task"]):
    eeg_signal = group["eeg_value"].values
    
    for start in range(0, len(eeg_signal) - WINDOW_SIZE + 1, STEP):
        window = eeg_signal[start:start + WINDOW_SIZE]
        windows.append(window)
        labels.append(task)
        subjects.append(subject)

# Convert to numpy arrays
X = np.array(windows)          # shape: (num_windows, 250)
y = np.array(labels)
subjects = np.array(subjects)

print("Total windows created:", X.shape[0])
print("Each window shape:", X.shape[1])
np.save("eeg_windows.npy", X)
np.save("labels.npy", y)
np.save("subjects.npy", subjects)
