import torch
from torchvision import transforms, models
from PIL import Image
import sys
import os

# Load and preprocess image
image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = transform(image).unsqueeze(0)  # Add batch dimension

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 4 classes (casme2, lpw, samm, ubfc)
model.load_state_dict(torch.load('saved_models/lie_detection_model.pth'))
model.eval()

# Predict
with torch.no_grad():
    output = model(image)
    predicted_class = torch.argmax(output, dim=1).item()

# Class mapping (update if needed)
class_map = {0: 'casme2', 1: 'lpw', 2: 'samm', 3: 'ubfc'}
print(f"🧠 Predicted class: {class_map[predicted_class]}")
import torch
from torchvision import models, transforms
from PIL import Image
import sys
import os

# Check for input image
if len(sys.argv) < 2:
    print("Please provide an image path.")
    sys.exit(1)

image_path = sys.argv[1]

# Load the image
image = Image.open(image_path).convert("RGB")

# Transform the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_tensor = transform(image).unsqueeze(0)

# Load the trained model
model = models.resnet18(weights=None)
num_classes = 4  # adjust if necessary
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("saved_models/lie_detection_model.pth", map_location=torch.device('cpu')))
model.eval()

# Predict
with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)

# Map predicted index to dataset/class name
class_names = ["casme2", "lpw", "samm", "ubfc"]
pred_class = class_names[predicted.item()]
print("🧠 Predicted class:", pred_class)

# Add custom mapping from dataset to Lie/Truth
label_mapping = {
    "casme2": "Lie",
    "samm": "Lie",
    "lpw": "Truth",
    "ubfc": "Truth"
}
predicted_label = label_mapping.get(pred_class, "Unknown")
print("🧠 Predicted label:", predicted_label)
