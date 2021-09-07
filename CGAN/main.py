import torchvision
import cv2
from tensorflow.keras.datasets import mnist
import numpy as np
from generator import generator
from discriminator import discriminator
import torch.nn as nn
import torch as T
from torch.utils.data import DataLoader
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tensorflow.keras.utils import to_categorical


latent_size = 100
batch_size = 64
train_steps = 25
lr = 0.0002
decay = 6e-8

train_list = []
index = np.arange(batch_size)
D_losses = []


if __name__ == "__main__":

    G = generator(latent_size, lr)
    D = discriminator(lr)

    G = G.to(device)
    D = D.to(device)

    G = G.float()
    D = D.float()



    data = torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ]))
    loaded = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    for epoch in range(train_steps):
        D_loss_total = 0
        A_loss_total = 0
        for batch_idx, (example_data, example_targets) in enumerate(loaded):
            y_train = to_categorical(example_targets)
            y_train = T.Tensor(y_train).to(device)
            x_train = T.Tensor(example_data).to(device)
            if len(y_train[1]) == 9:
                continue
            #print(len(y_train[1]), batch_idx)

            noise = np.random.uniform(-1.0,
                                      1.0,
                                      size=[batch_size, latent_size])

            noise = T.Tensor(noise).to(device)

            fake_labels = np.eye(len(y_train[1]))[np.random.choice(len(y_train[1]),batch_size)]
            fake_labels = T.Tensor(fake_labels).to(device)


            # Discriminator Training
            D.optimizer.zero_grad()
            fake_images = G(noise.float(), fake_labels.float())

            one = np.ones(batch_size)
            zero = np.zeros(batch_size)
            one = T.Tensor(one).to(device)
            zero = T.Tensor(zero).to(device)
            y = T.cat((one, zero))
            y = T.reshape(y, (128, 1)).to(device)

            D_input_images = T.cat((x_train, fake_images))
            D_input_labels = T.cat((y_train, fake_labels))

            D_output = D(D_input_images.float(), D_input_labels.float())
            D_loss = D.loss(D_output, y).to(device)
            D_loss_total += D_loss.item()
            D_loss.backward()

            D_optim = D.optimizer
            D_optim.step()

            # Adversarial Training
            G.optimizer.zero_grad()
            fake_images = G(noise.float(), fake_labels.float())

            D_output = D(fake_images.float(), fake_labels.float())
            y = np.ones(batch_size)
            y = T.Tensor(y)
            y = T.reshape(y, (64, 1)).to(device)

            A_loss = G.loss(D_output, y).to(device)
            A_loss_total += A_loss.item()
            A_loss.backward()

            G_optim = G.optimizer
            G_optim.step()

        print(f"For epoch {epoch} Discriminator Loss is : {D_loss_total / len(loaded)}")
        print(f"For epoch {epoch} Adversarial Loss is : {A_loss_total / len(loaded)}")
        print("///////////////////////////////////////////////////////////////////////////////")

    G.eval()
    images = []
    labels = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8,9,9,9,9,9]
    labels = to_categorical(labels)
    labels = T.Tensor(labels).to(device)
    noise = np.random.uniform(-1.0,
                              1.0,
                              size=[50, latent_size])
    noise = T.Tensor(noise).to(device)
    fake_images = G(noise.float(), labels.float())
    counter = 0

    for image in fake_images:
        img = image.cpu().detach()
        img = img.reshape((28, 28))
        img = img * 255.0
        img = np.array(img)
        cv2.imwrite(f"generated_images/image{counter}.jpg", img)
        counter += 1