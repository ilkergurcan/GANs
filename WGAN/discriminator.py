import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class discriminator(nn.Module):
    def __init__(self, lr):
        super(discriminator,self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1),
            nn.LeakyReLU(0.2),

        )

        self.fc = nn.Linear(12544, 1)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.BCELoss()

    def wasserstein_loss(self, y_label, y_pred):
        return -T.mean(y_label * y_pred)

    def forward(self, image):
        image = image.view(-1, 1, 28, 28)
        seq = self.seq(image)

        flat = seq.view(seq.size(0), -1)

        #hidden = T.sigmoid(self.fc(flat))
        hidden = self.fc(flat)

        return hidden
