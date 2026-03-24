import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

MODEL_PATH = r"C:\projects\prosthetic_hippocampal_using_cnn\outputs\memory_cnn.pth"
DB_PATH = r"C:\projects\prosthetic_hippocampal_using_cnn\memory_db.json"

# ---------------------------------------------------------
# 1. Load Semantic Embedding Model
# ---------------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# 2. Load CNN Memory Classifier
# ---------------------------------------------------------
from torchvision import models

def load_model():
    model = models.resnet18(weights="DEFAULT")
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # encode vs recall

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------
# 3. Image Transform
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ---------------------------------------------------------
# 4. DB Handling
# ---------------------------------------------------------
def load_memory_db():
    if not os.path.exists(DB_PATH):
        return {"memories": []}

    try:
        with open(DB_PATH, "r") as f:
            data = json.load(f)
            return data if "memories" in data else {"memories": []}
    except:
        return {"memories": []}

def save_memory_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=4)


# ---------------------------------------------------------
# 5. CNN Classification
# ---------------------------------------------------------
def classify_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_t)
        pred = torch.argmax(logits, dim=1).item()

    return "encode" if pred == 0 else "recall"


# ---------------------------------------------------------
# 6. Hippocampal Recall Scoring
# ---------------------------------------------------------
def compute_recall_score(memory, current_time, recall_query_embedding=None):
    # --- Time bias (WEAK, not dominant) ---
    mem_time = datetime.strptime(memory["time"], "%Y-%m-%d %H:%M")
    time_diff = abs((current_time - mem_time).total_seconds() / 60)
    time_score = max(0, 1 - (time_diff / 180))  # 3-hour decay

    # --- Semantic similarity (PRIMARY driver) ---
    if recall_query_embedding is not None:
        mem_emb = torch.tensor(memory["embedding"])
        semantic_score = float(util.cos_sim(mem_emb, recall_query_embedding))
    else:
        semantic_score = 0.3  # weak default to avoid ties

    # --- Memory strength (decay-based competition) ---
    strength = memory.get("strength", 1.0)

    # --- Final hippocampal-style score ---
    final_score = (
        0.75 * semantic_score +
        0.25 * time_score
    ) * strength

    return final_score


# ---------------------------------------------------------
# 7. Main App Logic
# ---------------------------------------------------------
def main():
    print("Enter EEG image path:")
    img_path = input("> ").strip()

    if not os.path.exists(img_path):
        print(" File not found!")
        return

    model = load_model()
    prediction = classify_image(model, img_path)

    print(f"\n Classified as: **{prediction.upper()}**\n")

    db = load_memory_db()

    # -------------------------------------------------
    # ENCODE MODE
    # -------------------------------------------------
    if prediction == "encode":
        print("What do you want me to remember?")
        text = input("> ")

        print("At what time should this be recalled? (HH:MM, 24hr)")
        time_input = input("> ")

        today = datetime.now().strftime("%Y-%m-%d")
        full_time = f"{today} {time_input}"

        emb = embedder.encode(text).tolist()

        entry = {
            "text": text,
            "time": full_time,
            "embedding": emb,
            "strength": 1.0   # initial encoding strength
        }

        db["memories"].append(entry)
        save_memory_db(db)

        print("\n Memory encoded successfully!\n")

    # -------------------------------------------------
    # RECALL MODE
    # -------------------------------------------------
    else:
        if len(db["memories"]) == 0:
            print("⚠ No stored memories found.")
            return

        print("Describe what you're trying to recall (optional):")
        query = input("> ").strip()

        query_emb = embedder.encode(query) if query else None
        now = datetime.now()

        best_memory = None
        best_score = -1

        for mem in db["memories"]:
            score = compute_recall_score(mem, now, query_emb)
            if score > best_score:
                best_score = score
                best_memory = mem

        # Strength decay after recall (competition effect)
        best_memory["strength"] *= 0.9
        save_memory_db(db)

        print("\n Retrieved memory:\n")
        print(f" {best_memory['text']}")
        print(f" Time context: {best_memory['time']}")
        print(f" Recall confidence: {best_score:.2f}\n")


if __name__ == "__main__":
    main()
