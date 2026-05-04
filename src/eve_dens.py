# src/evaluate_densenet121.py
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np
from torchvision import models

from dataset import get_data_loaders

DATA_DIR = r"data"
MODEL_PATH = r"models/densenet121_glaucoma.pth"
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("Using device:", DEVICE)
    
    # Load test data
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE
    )
    
    num_classes = len(class_names)
    
    # Load model
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Get predictions
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Calculate metrics
    test_acc = np.mean(y_true == y_pred) * 100
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    # ROC-AUC for binary classification
    roc_auc = None
    if num_classes == 2:
        y_prob_positive = y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_prob_positive)
        roc_auc = auc(fpr, tpr)
    
    # Print results
    print("=" * 60)
    print("DenseNet121 Evaluation Results")
    print("=" * 60)
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{report}")
    
    if roc_auc is not None:
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Save to text file
    with open("densenet121_evaluation.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DenseNet121 Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n")
        f.write(f"Classification Report:\n{report}\n")
        
        if roc_auc is not None:
            f.write(f"ROC-AUC Score: {roc_auc:.4f}\n")
    
    print("\n✅ Results saved to 'densenet121_evaluation.txt'")

if __name__ == "__main__":
    main()