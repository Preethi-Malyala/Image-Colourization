import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#hyperparameters
lr = 2e-4  # could also use two lrs, one for gen and one for disc
batch_size = 128
image_size = 64  #The input is a 3x64x64 input image
channels_img = 1 #number of channels in the output image (taking 3 will cause error)
noise_dim = 100
epochs = 5
disc_feature_size = 64
gen_feature_size = 64
beta1 = 0.5
#the output is a 3x64x64 RGB image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
])

mnist_train = dsets.MNIST(root = 'data/',
                      train=True,
                      transform = transform,
                      download=True)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                      batch_size = batch_size,
                                      shuffle=True)

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


#converting noise to data-space means ultimately creating a RGB image with the same size as the training images (i.e. 3x64x64)
#assuming gen_feature_size=64


class Generator(nn.Module):
    def __init__(self,noise_dim,channels_img,gen_feature_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            #torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1)
            nn.ConvTranspose2d( noise_dim, gen_feature_size * 16, 4, 1, 0),  #gen_feature_size*16=1024 as shown in paper
            nn.BatchNorm2d(gen_feature_size *16),
            nn.ReLU(True),
            # state size. (gen_feature_size *16) x 4 x 4
            nn.ConvTranspose2d(gen_feature_size *16, gen_feature_size *8, 4, 2, 1),
            nn.BatchNorm2d(gen_feature_size *8),
            nn.ReLU(True),
            # state size. (gen_feature_size *8) x 8 x 8
            nn.ConvTranspose2d( gen_feature_size *8, gen_feature_size *4, 4, 2, 1),
            nn.BatchNorm2d(gen_feature_size *4),
            nn.ReLU(True),
            # state size. (gen_feature_size *4) x 16 x 16
            nn.ConvTranspose2d( gen_feature_size *4, gen_feature_size *2, 4, 2, 1),
            nn.BatchNorm2d(gen_feature_size *2),
            nn.ReLU(True),
            # state size. (gen_feature_size *2) x 32 x 32
            nn.ConvTranspose2d( gen_feature_size *2, channels_img, 4, 2, 1),
            nn.Tanh()
            # state size. (channels_img) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# visualising the outputs from each layer in generator

netG = Generator(noise_dim,channels_img,gen_feature_size=64).to(device)
netG.apply(initialize_weights)


# Here, Discriminator takes a 3x64x64 input image, processes it through a series of Conv2d, BatchNorm2d, and LeakyReLU layers,
# and outputs the final probability through a Sigmoid activation function

class Discriminator(nn.Module):
    def __init__(self, channels_img,disc_feature_size):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (channels_img) x 64 x 64
            nn.Conv2d(channels_img, disc_feature_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_feature_size) x 32 x 32
            nn.Conv2d(disc_feature_size, disc_feature_size * 2, 4, 2, 1),
            nn.BatchNorm2d(disc_feature_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_feature_size*2) x 16 x 16
            nn.Conv2d(disc_feature_size * 2, disc_feature_size * 4, 4, 2, 1),
            nn.BatchNorm2d(disc_feature_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_feature_size*4) x 8 x 8
            nn.Conv2d(disc_feature_size * 4,disc_feature_size * 8, 4, 2, 1),
            nn.BatchNorm2d(disc_feature_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (disc_feature_size*8) x 4 x 4
            nn.Conv2d(disc_feature_size * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# visualising the outputs from each layer in discriminator

netD = Discriminator(channels_img,disc_feature_size).to(device)
netD.apply(initialize_weights)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, noise_dim , 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))



# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(epochs):
    # For each batch in the dataloader
    for i, data in enumerate(train_loader , 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, noise_dim, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(train_loader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1