import os
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from dataset_loader import CustomDataset
from tqdm import tqdm  # ✅ Progress bar

# Paths
dataset_path = "data/processed/all_combined"

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
dataset = CustomDataset(root_dir=dataset_path, transform=transform)
print(f"✅ Total samples found: {len(dataset)}")

# Optionally use a subset for testing
# dataset = torch.utils.data.Subset(dataset, list(range(2000)))

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model setup
model = models.resnet18(weights="DEFAULT")
model.fc = torch.nn.Linear(model.fc.in_features, 4)

# Use CPU
device = torch.device("cpu")
model = model.to(device)

# Loss & Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
print("🚀 Starting training...")

num_epochs = 1
for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
# Save model
os.makedirs('saved_models', exist_ok=True)
torch.save(model.state_dict(), 'saved_models/lie_detection_model.pth')
print("✅ Model saved at 'saved_models/lie_detection_model.pth'")
