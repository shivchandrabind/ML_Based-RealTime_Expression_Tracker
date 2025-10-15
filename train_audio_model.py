import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from audio_dataset_loader import AudioFeatureDataset
from tqdm import tqdm
import os

# Paths
feature_dir = "data/audio/features"
model_save_path = "saved_models/audio_lie_model.pth"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Dataset
dataset = AudioFeatureDataset(feature_dir)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8)

# Model
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 2 classes: lie, truth
        )

    def forward(self, x):
        return self.model(x)

# Detect input feature size from a batch
for features, _ in train_loader:
    input_size = features.shape[1]
    break

model = MLP(input_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 20

print("🚀 Training started...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"🧠 Epoch {epoch+1}: Training Loss = {total_loss:.4f}")

# Validation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in val_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"✅ Validation Accuracy: {accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), model_save_path)
print(f"💾 Model saved to: {model_save_path}")
