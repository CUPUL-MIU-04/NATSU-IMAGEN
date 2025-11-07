import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_data_loader(dataset_name='cifar10', batch_size=128, image_size=64):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
    elif dataset_name == 'mnist':
        dataset = datasets.MNIST(root='./data', download=True, transform=transform)
    else:
        raise ValueError("Dataset no soportado")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader