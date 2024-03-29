import warnings

from omegaconf import OmegaConf
from timm import create_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MNIHDataset, get_transforms
from train import train
from utils import MetricProcessor


warnings.filterwarnings("ignore")

def main(cfg):
    # Set dataset
    train_transforms, val_transforms = get_transforms(cfg.dataset)
    train_dataset = MNIHDataset(cfg.dataset, train=True, transforms=train_transforms)
    val_dataset = MNIHDataset(cfg.dataset, train=False, transforms=val_transforms)
    class_weights = train_dataset.cls_weights
    cls2label = train_dataset.cls2label

    # Set dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.hypers['batch_size'],
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.hypers['batch_size'],
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    # Create the model
    model = create_model(cfg.hypers['model_name'], pretrained=True, num_classes=cfg.hypers['num_classes'])

    # Set the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define the loss function and optimizer for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=cfg.hypers['learning_rate'])
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)  # Binary cross-entropy loss

    # Metric Processor
    mproc = MetricProcessor(cfg.hypers)

    # Training loop
    train(cfg, device, model, optimizer, criterion, train_loader, val_loader, mproc)


if __name__ == '__main__':
    # Read config
    cfg = OmegaConf.load('./config.yml')

    main(cfg.vgg16)
