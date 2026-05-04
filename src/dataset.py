# src/dataset.py
import os
import torch
from torchvision import datasets, transforms

def get_data_loaders(data_dir, batch_size=32, img_size=224):
    """
    Expects directory structure:
    data_dir/
      train/
        glaucoma/
        non_glaucoma/
      val/
        glaucoma/
        non_glaucoma/
      test/
        glaucoma/
        non_glaucoma/
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")
    test_dir  = os.path.join(data_dir, "test")

    # Data augmentation for train, simple resize+normalize for val/test
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=eval_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    class_names = train_dataset.classes  # ['glaucoma', 'non_glaucoma'] or similar

    return train_loader, val_loader, test_loader, class_names
