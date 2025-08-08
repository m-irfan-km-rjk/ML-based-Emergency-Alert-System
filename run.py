import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

# ========== 1. Set Device ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 2. Rebuild and Load the Model ==========
model_path = "best_emergency_model.pth"

if not os.path.exists(model_path):
    print(f"❌ Model file '{model_path}' not found.")
    sys.exit(1)

# Define class names based on training
class_names = ["accident", "collapse", "fire", "flood", "normal"]

# Rebuild ResNet18 model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ========== 3. Define Transform ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========== 4. Get Image Path ==========
img_path = input("📁 Enter the full path to the image: ").strip()

if not os.path.isfile(img_path):
    print("❌ Image file not found.")
    sys.exit(1)

# ========== 5. Load and Predict ==========
try:
    image = Image.open(img_path).convert("RGB")
except Exception as e:
    print(f"❌ Could not open image: {e}")
    sys.exit(1)

input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted = torch.max(probs, 1)

predicted_class = class_names[predicted.item()]
confidence_score = confidence.item() * 100

# ========== 6. Display Result ==========
print(f"✅ Predicted class: {predicted_class} ({confidence_score:.2f}% confidence)")

plt.imshow(image)
plt.title(f"Prediction: {predicted_class} ({confidence_score:.1f}%)")
plt.axis('off')
plt.show()