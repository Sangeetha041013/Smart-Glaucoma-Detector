# src/predict_single.py
import torch
from torchvision import transforms
from PIL import Image

def load_model(model_path, model_type="resnet50"):
    from torchvision import models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    if model_type == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif model_type == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Unsupported model_type")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, class_names, device


def preprocess_image(image: Image.Image, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # add batch dimension


def predict_image(model, device, image: Image.Image, class_names, img_size=224):
    input_tensor = preprocess_image(image, img_size=img_size).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    pred_idx = probs.argmax()
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_class, confidence, probs
