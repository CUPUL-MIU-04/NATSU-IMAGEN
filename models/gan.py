import torch
import torch.nn as nn
from .generator import Generator
from .discriminator import Discriminator

class GAN(nn.Module):
    def __init__(self, z_dim=100, channels=3, feature_map_size=64):
        super(GAN, self).__init__()
        self.generator = Generator(z_dim, channels, feature_map_size)
        self.discriminator = Discriminator(channels, feature_map_size)
        self.z_dim = z_dim
        
    def generate(self, num_images, device='cpu'):
        noise = torch.randn(num_images, self.z_dim, 1, 1, device=device)
        with torch.no_grad():
            generated_images = self.generator(noise)
        return generated_images
    
    def discriminate(self, images):
        with torch.no_grad():
            predictions = self.discriminator(images)
        return predictions