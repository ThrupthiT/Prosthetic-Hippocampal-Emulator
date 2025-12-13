import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os
from datetime import datetime

MODEL_PATH = r"C:\projects\prosthetic_hippocampal_using_cnn\outputs\memory_cnn.pth"
DB_PATH = r"C:\projects\prosthetic_hippocampal_using_cnn\memory_db.json"


# ------------------------------------------------
# 1. Load CNN Model (same architecture as training)
# ------------------------------------------------
def load_model():
    model = models.resnet18(weights="DEFAULT")

    # Modify first conv layer for grayscale input
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # Modify FC layer for 2 classes (encode / recall)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model


# ------------------------------------------------
# 2. Image Transform (same as training)
# ------------------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ------------------------------------------------
# 3. Load/Create memory DB
# ------------------------------------------------
def load_memory_db():
    if not os.path.exists(DB_PATH):
        return {"memories": []}

    try:
        with open(DB_PATH, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and "memories" in data:
            return data
        else:
            return {"memories": []}

    except:
        return {"memories": []}


def save_memory_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=4)


# ------------------------------------------------
# 4. CNN Prediction
# ------------------------------------------------
def classify_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_t)
        pred = torch.argmax(logits, dim=1).item()

    return "encode" if pred == 0 else "recall"


# ------------------------------------------------
# 5. Helper: find the closest memory to system time
# ------------------------------------------------
def find_closest_memory(memories):
    if not memories:
        return None

    now = datetime.now()

    closest = None
    smallest_diff = None

    for mem in memories:
        try:
            event_time = datetime.strptime(mem["event_time"], "%I:%M %p")
            event_time = event_time.replace(
                year=now.year, month=now.month, day=now.day
            )
        except:
            continue  # skip invalid time

        diff = abs((event_time - now).total_seconds())

        if smallest_diff is None or diff < smallest_diff:
            smallest_diff = diff
            closest = mem

    return closest


# ------------------------------------------------
# 6. MAIN LOGIC
# ------------------------------------------------
def main():
    print("Enter EEG image path:")
    img_path = input("> ").strip()

    if not os.path.exists(img_path):
        print("❌ File not found!")
        return

    model = load_model()
    prediction = classify_image(model, img_path)

    print(f"\n🧠 Classified as: **{prediction.upper()}**\n")

    db = load_memory_db()

    # ---------------------------
    # ENCODE
    # ---------------------------
    if prediction == "encode":
        print("What should I remember?")
        memory_text = input("> ")

        print("When should I remind you? (Ex: 5:00 PM)")
        memory_time = input("> ").strip()

        memory_entry = {
            "text": memory_text,
            "event_time": memory_time,     # <-- user input event time
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        db["memories"].append(memory_entry)
        save_memory_db(db)

        print("\n✅ Memory saved successfully!\n")

    # ---------------------------
    # RECALL
    # ---------------------------
    else:
        if len(db["memories"]) == 0:
            print("⚠ No stored memories.")
            return

        closest = find_closest_memory(db["memories"])

        if closest:
            print("\n🧠 Closest Memory:")
            print(f"- {closest['text']}")
            print(f"- Scheduled at {closest['event_time']}\n")
        else:
            print("⚠ No valid memories found.")


if __name__ == "__main__":
    main()
