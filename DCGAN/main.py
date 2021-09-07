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

latent_size = 100
batch_size = 64
train_steps = 25
lr = 0.0002
decay = 6e-8

flat = np.zeros((32,28*28))
d = np.reshape(flat, (32,1,28,28))


# #(28,28,1) değil (1,28,28) yap
# cv2.imshow("", x_train[1])
# cv2.waitKey(0)

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



    (x_train, _), (_, _) = mnist.load_data()

    for x in x_train:
        t = np.reshape(x, (1,28,28))
        train_list.append(t)

    train_list = np.array(train_list)
    train_list = train_list / 255.0
    x_train = T.Tensor(train_list).to(device)

    train = DataLoader(x_train, batch_size, shuffle=True, drop_last=True)
    print(len(train))
    for epoch in range(train_steps):
        D_loss_total = 0
        A_loss_total = 0
        for batch in train:
            noise = np.random.uniform(-1.0,
                                      1.0,
                                      size=[batch_size, latent_size])
            noise = T.Tensor(noise).to(device)

            # Discriminator Training
            D.optimizer.zero_grad()
            fake_images = G(noise.float())
            print(fake_images.shape)
            print(fake_images[0].shape)
            D_input = T.cat((batch, fake_images))

            one = np.ones(batch_size)
            zero = np.zeros(batch_size)
            one = T.Tensor(one).to(device)
            zero = T.Tensor(zero).to(device)
            y = T.cat((one, zero))
            y = T.reshape(y, (128, 1)).to(device)

            D_output = D(D_input.float())
            D_loss = D.loss(D_output, y).to(device)
            D_loss_total += D_loss.item()
            D_loss.backward()

            D_optim = D.optimizer
            D_optim.step()
            # D_decay = T.optim.lr_scheduler.ExponentialLR(optimizer=D_optim, gamma=decay)
            # D_decay.step()

            # Adversarial Training
            G.optimizer.zero_grad()
            fake_images = G(noise.float())

            D_output = D(fake_images.float())
            y = np.ones(batch_size)
            y = T.Tensor(y)
            y = T.reshape(y, (64, 1)).to(device)


            A_loss = G.loss(D_output, y).to(device)
            A_loss_total += A_loss.item()
            A_loss.backward()

            G_optim = G.optimizer
            G_optim.step()
            # G_decay = T.optim.lr_scheduler.ExponentialLR(optimizer=G_optim, gamma=decay)
            # G_decay.step()

        print(f"For epoch {epoch} Discriminator Loss is : {D_loss_total / len(train)}")
        print(f"For epoch {epoch} Adversarial Loss is : {A_loss_total / len(train)}")
        print("///////////////////////////////////////////////////////////////////////////////")



    G.eval()
    images = []
    noise = np.random.uniform(-1.0,
                                  1.0,
                                  size=[batch_size, latent_size])
    noise = T.Tensor(noise).to(device)
    fake_images = G(noise.float())
    counter = 0

    for image in fake_images:

        img = image.cpu().detach()
        img = img.reshape((28,28))
        img = img * 255.0
        img = np.array(img)
        cv2.imwrite(f"generated_images/image{counter}.jpg", img)
        counter += 1