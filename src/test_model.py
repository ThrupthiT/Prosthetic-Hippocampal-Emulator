import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from datetime import datetime
import os

MODEL_PATH = r"C:\projects\prosthetic_hippocampal_using_cnn\outputs\memory_cnn.pth"
LOG_PATH = r"C:\projects\prosthetic_hippocampal_using_cnn\outputs\prediction_log.txt"

# ---------------------------------------------------
# FIXED ARCHITECTURE – must match TRAINING EXACTLY
# ---------------------------------------------------
def load_trained_model():
    model = models.resnet18(weights=None)

    # SAME Conv1 modification as training
    model.conv1 = nn.Conv2d(
        1,      # 1 channel input from grayscale EEG image
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # SAME FC layer as training
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)   # encode vs recall

    # Load weights
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model


# ---------------------------------------------------
# Image transform — must match training transforms
# ---------------------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # SAME AS TRAINING
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ---------------------------------------------------
# Logging Function
# ---------------------------------------------------
def log_prediction(image_path, predicted_label):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    with open(LOG_PATH, "a") as f:
        f.write(f"{timestamp}, {os.path.basename(image_path)}, {predicted_label}\n")

    print(f"Logged → {timestamp}, {predicted_label}")


# ---------------------------------------------------
# Prediction function
# ---------------------------------------------------
def predict_image(img_path):
    model = load_trained_model()

    img = Image.open(img_path).convert("L")  # load as grayscale
    img = transform(img).unsqueeze(0)        # add batch dim

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    classes = ["encode", "recall"]
    predicted_label = classes[pred]

    # Log the prediction
    log_prediction(img_path, predicted_label)

    return predicted_label


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    test_img = r"C:\projects\prosthetic_hippocampal_using_cnn\test_samples\test2.png"
    print("\nTesting:", test_img)

    result = predict_image(test_img)
    print("Predicted Class →", result)


if __name__ == "__main__":
    main()
