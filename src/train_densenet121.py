# src/train_densenet121.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from dataset import get_data_loaders
from utils import train_one_epoch, evaluate

DATA_DIR = r"data"
SAVE_PATH = r"models/densenet121_glaucoma.pth"
NUM_EPOCHS = 15
BATCH_SIZE = 32
IMG_SIZE = 224
LEARNING_RATE = 1e-4

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE
    )

    num_classes = len(class_names)

    # Load pretrained DenseNet121
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_report = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.4f}")
        print("Validation classification report:\n", val_report)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                },
                SAVE_PATH,
            )
            print(f"✅ New best model saved with val acc = {best_val_acc:.4f}")

    checkpoint = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, test_acc, test_report = evaluate(model, test_loader, criterion, device)
    print("\n=== Test Performance (DenseNet121) ===")
    print("Test Accuracy:", test_acc)
    print(test_report)


if __name__ == "__main__":
    main()
