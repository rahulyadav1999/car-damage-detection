
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# Select device: Apple M2 GPU (MPS) or CPU
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

trained_model = None

class_names = [
    'Front Breakage',
    'Front Crushed',
    'Front Normal',
    'Rear Breakage',
    'Rear Crushed',
    'Rear Normal'
]


# Load the ResNet50 architecture
class CarClassifierResNet(nn.Module):

    def __init__(self, num_classes=6):
        super().__init__()

        self.model = models.resnet50(weights=None)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Prediction function
def predict(image_path):

    global trained_model

    # Image preprocessing
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Load trained model only once
    if trained_model is None:

        trained_model = CarClassifierResNet()

        # Locate saved model
        BASE_DIR = Path(__file__).resolve().parent.parent
        MODEL_PATH = BASE_DIR / "saved_model.pth"

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        # Load trained weights
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )

        trained_model.load_state_dict(state_dict)

        trained_model = trained_model.to(device)
        trained_model.eval()

    # Make prediction
    with torch.no_grad():

        output = trained_model(image_tensor)

        _, predicted_class = torch.max(output, 1)

        return class_names[predicted_class.item()]
