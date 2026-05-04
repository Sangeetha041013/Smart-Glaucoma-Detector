# src/evaluate_resnet50.py
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models

from dataset import get_data_loaders

DATA_DIR = r"data"
MODEL_PATH = r"models/resnet50_glaucoma.pth"  # Change this to your ResNet50 model path
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
           title='Confusion Matrix - ResNet50',
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

def plot_roc_curve(fpr, tpr, roc_auc, model_name="ResNet50"):
    """Plot ROC curve"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return fig

def load_resnet50_model(model_path, num_classes, device):
    """Load ResNet50 model with custom classifier"""
    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Modify the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

def get_predictions(model, dataloader, device):
    """Get predictions, true labels, and probabilities"""
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def print_detailed_report(y_true, y_pred, y_prob, class_names, num_classes):
    """Print detailed evaluation report"""
    # Overall accuracy
    test_acc = np.mean(y_true == y_pred) * 100
    print(f"\n🔥 Test Accuracy: {test_acc:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📊 Confusion Matrix:\n{cm}")
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Per-class accuracy
    print(f"\n🎯 Per-class Accuracy:")
    for i, class_name in enumerate(class_names):
        if cm[i, i] + cm[i, :].sum() > 0:
            class_acc = cm[i, i] / cm[i, :].sum() * 100
            print(f"  {class_name}: {class_acc:.2f}%")
    
    # ROC-AUC for binary classification
    if num_classes == 2:
        y_prob_positive = y_prob[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true, y_prob_positive)
        roc_auc = auc(fpr, tpr)
        
        print(f"\n📈 ROC-AUC Score: {roc_auc:.4f}")
        
        # Optimal threshold
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        print(f"🎯 Optimal threshold: {optimal_threshold:.4f}")
        print(f"   - FPR at optimal: {fpr[optimal_idx]:.4f}")
        print(f"   - TPR at optimal: {tpr[optimal_idx]:.4f}")
        
        return cm, fpr, tpr, roc_auc
    
    return cm, None, None, None

def main():
    print("="*60)
    print("ResNet50 Model Evaluation")
    print("="*60)
    print(f"Using device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    
    # Load test data
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE
    )
    
    num_classes = len(class_names)
    print(f"\n📁 Dataset Info:")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Classes: {class_names}")
    print(f"  - Test samples: {len(test_loader.dataset)}")
    
    # Load model
    print(f"\n🔄 Loading ResNet50 model...")
    model, checkpoint = load_resnet50_model(MODEL_PATH, num_classes, DEVICE)
    
    # Get class names from checkpoint or default
    loaded_class_names = checkpoint.get('class_names', class_names)
    print(f"✅ Model loaded successfully!")
    
    # Get predictions
    print(f"\n🔍 Running evaluation on test set...")
    y_true, y_pred, y_prob = get_predictions(model, test_loader, DEVICE)
    
    # Generate detailed report
    print("\n" + "="*60)
    print("Evaluation Results - ResNet50")
    print("="*60)
    
    cm, fpr, tpr, roc_auc = print_detailed_report(
        y_true, y_pred, y_prob, loaded_class_names, num_classes
    )
    
    # Visualizations
    print(f"\n📊 Generating visualizations...")
    
    # Confusion matrix plot
    cm_fig = plot_confusion_matrix(cm, loaded_class_names)
    cm_filename = 'confusion_matrix_resnet50.png'
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved as '{cm_filename}'")
    
    # ROC curve plot (for binary classification)
    if num_classes == 2 and fpr is not None and tpr is not None:
        roc_fig = plot_roc_curve(fpr, tpr, roc_auc, "ResNet50")
        roc_filename = 'roc_curve_resnet50.png'
        plt.savefig(roc_filename, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curve saved as '{roc_filename}'")
    
    # Display sample predictions
    print(f"\n👁️ Sample predictions (random 5 samples):")
    if len(y_true) > 0:
        sample_indices = np.random.choice(len(y_true), min(5, len(y_true)), replace=False)
        for idx in sample_indices:
            true_label = loaded_class_names[y_true[idx]]
            pred_label = loaded_class_names[y_pred[idx]]
            confidence = np.max(y_prob[idx]) * 100
            correct = "✓" if y_true[idx] == y_pred[idx] else "✗"
            print(f"  {correct} True: {true_label:<15} | Pred: {pred_label:<15} | Confidence: {confidence:.1f}%")
    
    # Additional statistics
    print(f"\n📈 Additional Statistics:")
    print(f"  - Total test samples: {len(y_true)}")
    print(f"  - Correct predictions: {np.sum(y_true == y_pred)}")
    print(f"  - Wrong predictions: {np.sum(y_true != y_pred)}")
    
    # Class distribution in predictions
    print(f"\n📊 Prediction distribution:")
    unique, counts = np.unique(y_pred, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        percentage = (count / len(y_pred)) * 100
        print(f"  - {loaded_class_names[cls_idx]}: {count} samples ({percentage:.1f}%)")
    
    # Save detailed results to file
    results_filename = 'resnet50_evaluation_results.txt'
    with open(results_filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ResNet50 Evaluation Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Accuracy: {np.mean(y_true == y_pred) * 100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=loaded_class_names, digits=4))
        
        if num_classes == 2 and roc_auc is not None:
            f.write(f"\nROC-AUC Score: {roc_auc:.4f}\n")
    
    print(f"\n💾 Detailed results saved to '{results_filename}'")
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()