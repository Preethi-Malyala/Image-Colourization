import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

latent_size = 100
hidden_size = 256
epochs = 200
img_size = 28*28
batch_size = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
            ])

mnist_train = dsets.MNIST(root = '../data/',
                              train=True,
                              transform = transform,
                              download=True)

mnist_test = dsets.MNIST(root = '../data/',
                              train=False,
                              transform=transform)
#mnist_train = [(img, label),(img, label),............]
plt.imshow(mnist_train[7][0].squeeze(0))
print(mnist_train[7][1])

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                              batch_size = batch_size,
                                              shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                             batch_size = 10,
                                             shuffle=False)

class Generator(nn.Module):
          def __init__(self, latent_size, hidden_size, img_size):
            super(Generator, self).__init__()
            self.layer1 = nn.Linear(in_features = latent_size, out_features = hidden_size)
            self.layer2 = nn.Linear(in_features = hidden_size, out_features = hidden_size)
            self.layer3 = nn.Linear(in_features = hidden_size, out_features = img_size)

            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()

          def forward(self, x):
            out = self.relu(self.layer1(x))
            out = self.relu(self.layer2(out))
            out = self.layer3(out)
            out = self.tanh(out)

            return out

class Discriminator(nn.Module):
          def __init__(self, img_size, hidden_size):
            super(Discriminator, self).__init__()
            self.layer1 = nn.Linear(in_features = img_size, out_features = hidden_size)
            self.layer2 = nn.Linear(in_features = hidden_size, out_features = hidden_size)
            self.layer3 = nn.Linear(in_features = hidden_size, out_features = 1)

            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

          def forward(self, x):
            out = self.relu(self.layer1(x))
            out = self.relu(self.layer2(out))
            out = self.layer3(out)
            out = self.sigmoid(out)

            return out

G = Generator(latent_size, hidden_size, img_size).to(device)
D = Discriminator(img_size, hidden_size).to(device)

#BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
Criterion = nn.BCELoss()
D_optim = torch.optim.Adam(D.parameters(), lr=0.0002)
G_optim = torch.optim.Adam(G.parameters(), lr=0.0002)

total_step = len(train_loader)

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):
        imgs = imgs.reshape(batch_size, 28*28).to(device)  #28*28 = 784
        z = torch.randn(batch_size, latent_size).to(device)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        ''' Train a discriminator'''
        D_real_out = D(imgs)
        real_score = D_real_out
        D_real_loss = Criterion(D_real_out, real_labels)  # -y*log(y')

        fake_imgs = G(z)
        D_fake_out = D(fake_imgs)
        fake_score = D_fake_out
        D_fake_loss = Criterion(D_fake_out, fake_labels)    #- (1)*log(1-y')

        D_loss = D_real_loss + D_fake_loss # -y*log(y') - (1-y)*log(1-y')

        D_optim.zero_grad()
        D_loss.backward()
        D_optim.step()

        '''Train a Generator'''
        z = torch.randn(batch_size, latent_size).to(device)
        fake_imgs = G(z)
        D_out = D(fake_imgs)

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        G_loss = Criterion(D_out, real_labels) #-y*log(y')    y' = D(G(z)) y = 1   -> -log(D(G(z)))

        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()


        if((i+1)%200==0):
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                .format(epoch, epochs, i+1, total_step, D_loss.item(), G_loss.item(),
                    real_score.mean().item(), fake_score.mean().item()))