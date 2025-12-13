import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DATASET_DIR = r"C:\projects\prosthetic_hippocampal_using_cnn\cnn_dataset"
BATCH_SIZE = 8
EPOCHS = 10
LR = 0.0001
MODEL_PATH = r"C:\projects\prosthetic_hippocampal_using_cnn\outputs\memory_cnn.pth"

# ----------------------------
# 1. Transforms (Grayscale!)
# ----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # FORCE 1 CHANNEL
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# 2. Load Dataset
# ----------------------------
train_data = datasets.ImageFolder(root=DATASET_DIR, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

print("Classes:", train_data.classes)

# ----------------------------
# 3. Load ResNet and Modify It
# ----------------------------
model = models.resnet18(weights="DEFAULT")

# MODIFY FIRST LAYER TO ACCEPT 1 CHANNEL INSTEAD OF 3
model.conv1 = nn.Conv2d(
    1,      # <-- input channels changed from 3 to 1
    64, 
    kernel_size=7, 
    stride=2, 
    padding=3,
    bias=False
)

# MODIFY OUTPUT LAYER FOR 2 CLASSES
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # encode vs recall

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ----------------------------
# 4. Loss & Optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----------------------------
# 5. Training Loop
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# ----------------------------
# 6. Save Model
# ----------------------------
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")
