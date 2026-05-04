# src/evaluate_densenet121.py
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models

from dataset import get_data_loaders
from utils import evaluate

DATA_DIR = r"data"
MODEL_PATH = r"models/densenet121_glaucoma.pth"
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_confusion_matrix(cm, classes):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC curve"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return fig

def main():
    print("Using device:", DEVICE)
    
    # Load test data only
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE
    )
    
    num_classes = len(class_names)
    
    # Load model
    model = models.densenet121(weights=None)  # We'll load our own weights
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Get class names from checkpoint
    loaded_class_names = checkpoint.get('class_names', class_names)
    print(f"Classes: {loaded_class_names}")
    
    # For storing predictions and probabilities
    all_labels = []
    all_preds = []
    all_probs = []
    
    # Evaluation loop
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
    
    # Calculate overall test accuracy
    test_acc = np.mean(y_true == y_pred) * 100
    
    print("\n" + "="*50)
    print("DenseNet121 Evaluation Results")
    print("="*50)
    print(f"\n🔥 Test Accuracy: {test_acc:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📊 Confusion Matrix:\n{cm}")
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=loaded_class_names, digits=4))
    
    # Per-class accuracy from confusion matrix
    print(f"\n🎯 Per-class Accuracy:")
    for i, class_name in enumerate(loaded_class_names):
        if cm[i, i] + cm[i, :].sum() > 0:
            class_acc = cm[i, i] / cm[i, :].sum() * 100
            print(f"  {class_name}: {class_acc:.2f}%")
    
    # ROC Curve and AUC (for binary classification)
    if num_classes == 2:
        # Get probabilities for positive class (class 1)
        y_prob_positive = y_prob[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true, y_prob_positive)
        roc_auc = auc(fpr, tpr)
        
        print(f"\n📈 ROC-AUC Score: {roc_auc:.4f}")
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        print(f"🎯 Optimal threshold: {optimal_threshold:.4f}")
        print(f"   - FPR at optimal: {fpr[optimal_idx]:.4f}")
        print(f"   - TPR at optimal: {tpr[optimal_idx]:.4f}")
        
        # Plot ROC curve
        roc_fig = plot_roc_curve(fpr, tpr, roc_auc)
        plt.savefig('roc_curve_densenet121.png', dpi=300, bbox_inches='tight')
        print("✅ ROC curve saved as 'roc_curve_densenet121.png'")
    
    # Plot confusion matrix
    cm_fig = plot_confusion_matrix(cm, loaded_class_names)
    plt.savefig('confusion_matrix_densenet121.png', dpi=300, bbox_inches='tight')
    print("✅ Confusion matrix saved as 'confusion_matrix_densenet121.png'")
    
    # Display sample predictions
    print(f"\n👁️ Sample predictions:")
    sample_indices = np.random.choice(len(y_true), min(5, len(y_true)), replace=False)
    for idx in sample_indices:
        true_label = loaded_class_names[y_true[idx]]
        pred_label = loaded_class_names[y_pred[idx]]
        confidence = np.max(y_prob[idx]) * 100
        print(f"  True: {true_label:<15} | Pred: {pred_label:<15} | Confidence: {confidence:.1f}%")
    
    print("\n" + "="*50)
    print("Evaluation Complete!")
    print("="*50)

if __name__ == "__main__":
    main()