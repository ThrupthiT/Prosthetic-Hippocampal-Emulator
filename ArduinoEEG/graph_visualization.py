import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Input and output folders
input_folder = "spectrograms/encoding"   # change if needed
output_folder = "graph_images"
os.makedirs(output_folder, exist_ok=True)

def spectrogram_to_graph(img_path):
    # Open image using PIL
    img = Image.open(img_path).convert("L")  # convert to grayscale
    img = np.array(img)

    # Normalize
    img = img / 255.0

    # Convert to 1D signal
    signal = np.sum(img, axis=0)   # better than mean

    # Normalize signal
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

    return signal

# Process all images
for file in os.listdir(input_folder):
    if file.endswith(".png"):
        img_path = os.path.join(input_folder, file)

        signal = spectrogram_to_graph(img_path)

        # Time axis
        time = np.linspace(0, 4, len(signal))

        # Plot graph
        plt.figure(figsize=(10, 3))
        plt.plot(time, signal)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (µV)")
        plt.title("EEG Signal")

        # Save image
        save_path = os.path.join(output_folder, file)
        plt.savefig(save_path)
        plt.close()

print("Graph images generated successfully ✅")