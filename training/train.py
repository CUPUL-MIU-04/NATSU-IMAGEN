import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.gan import GAN
import os

class GANTrainer:
    def __init__(self, z_dim=100, channels=3, lr=0.0002, device='cuda'):
        self.device = device
        self.gan = GAN(z_dim, channels).to(device)
        self.optimizer_G = optim.Adam(self.gan.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.gan.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
        self.z_dim = z_dim
        
    def train_epoch(self, dataloader, epoch):
        self.gan.generator.train()
        self.gan.discriminator.train()
        
        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(self.device)
            
            # Etiquetas reales y falsas
            real_labels = torch.ones(batch_size, device=self.device)
            fake_labels = torch.zeros(batch_size, device=self.device)
            
            # Entrenar Discriminador
            self.optimizer_D.zero_grad()
            
            # Discriminar imágenes reales
            outputs_real = self.gan.discriminator(real_imgs)
            loss_real = self.criterion(outputs_real, real_labels)
            
            # Discriminar imágenes falsas
            noise = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
            fake_imgs = self.gan.generator(noise)
            outputs_fake = self.gan.discriminator(fake_imgs.detach())
            loss_fake = self.criterion(outputs_fake, fake_labels)
            
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            self.optimizer_D.step()
            
            # Entrenar Generador
            self.optimizer_G.zero_grad()
            
            outputs = self.gan.discriminator(fake_imgs)
            loss_G = self.criterion(outputs, real_labels)
            loss_G.backward()
            self.optimizer_G.step()
            
            if i % 100 == 0:
                print(f'Epoch [{epoch}], Step [{i}/{len(dataloader)}], '
                      f'Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}')
    
    def train(self, dataloader, epochs, save_interval=10):
        for epoch in range(epochs):
            self.train_epoch(dataloader, epoch)
            
            if epoch % save_interval == 0:
                self.save_model(f'checkpoint_epoch_{epoch}.pth')
                
    def save_model(self, path):
        torch.save({
            'generator_state_dict': self.gan.generator.state_dict(),
            'discriminator_state_dict': self.gan.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
        }, path)
        
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.gan.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.gan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])