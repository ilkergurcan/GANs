import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class generator(nn.Module):
    def __init__(self,latent_size, lr):
        super(generator, self).__init__()

        self.fc = nn.Linear(latent_size, 256*7*7)

        self.seq = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.Sigmoid()
        )

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.BCELoss()

    def wasserstein_loss(self, y_label, y_pred):
        return -T.mean(y_label * y_pred)

    def forward(self, noise):
        hidden = self.fc(noise)
        flatten = hidden.view(-1, 256, 7, 7)

        generated = self.seq(flatten)

        return generated
