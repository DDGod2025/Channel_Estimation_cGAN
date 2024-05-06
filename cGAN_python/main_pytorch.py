import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
import scipy.io as sio
import h5py
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime

from models.generator import Generator
from models.discriminator import Discriminator
from utils.losses import discriminator_loss, generator_loss, calculate_nmse

now = datetime.datetime.now()
file_name = now.strftime('%Y-%m-%d_%H-%M-%S') + '_cGAN'
dataset_filename = '../Data_Generation_matlab/Gan_Data/Gan_0_dBIndoor2p4_64ant_32users_8pilot.mat'
useIO = False

# SummaryWriter
log_dir = f'logs/{file_name}'
writer = SummaryWriter(log_dir)

# Set the device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# G and D
generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-5)

H_DeepMIMO = h5py.File(dataset_filename, 'r')

Y_sign_train = H_DeepMIMO['input_da']
Y_sign_test = H_DeepMIMO['input_da_test']

H_train = H_DeepMIMO['output_da']
H_test = H_DeepMIMO['output_da_test']

Y_sign_train = np.transpose(Y_sign_train, (3, 0, 2, 1))  # N x 2 x 64 x 8
Y_sign_test = np.transpose(Y_sign_test, (3, 0, 2, 1))  # N x 2 x 64 x 8
H_train = np.transpose(H_train, (3, 0, 2, 1))  # N x 2 x 64 x 32
H_test = np.transpose(H_test, (3, 0, 2, 1))  # N x 2 x 64 x 32

print('Y_shape: ', Y_sign_train.shape)
print('H_shape: ', H_train.shape)

# Convert NumPy arrays to PyTorch tensors
Y_sign_train = torch.from_numpy(Y_sign_train).float().to(device)
Y_sign_test = torch.from_numpy(Y_sign_test).float().to(device)

H_train = torch.from_numpy(H_train).float().to(device)
H_test = torch.from_numpy(H_test).float().to(device)

# Create a TensorDataset
train_dataset = torch.utils.data.TensorDataset(Y_sign_train, H_train)

# Define the data loaders
batch_size = 1
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Training loop
num_epochs = 30
loss_num = 0
size = len(train_dataset)
# Training GAN
generator.train()
discriminator.train()

for epoch in tqdm(range(num_epochs)):
    for i, (Y_sign, target) in enumerate(train_loader):
        generator_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()
        
        H_gen = generator(Y_sign)                          # input -> generated_target
        disc_real_output = discriminator(target)           # [input, target] -> disc output
        disc_generated_output = discriminator(H_gen)       # [input, generated_target] -> disc output

        gen_loss = generator_loss(disc_generated_output, H_gen, target)          # gen loss
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)  # disc loss
        
        gen_loss.backward(retain_graph=True)
        disc_loss.backward()
        
        generator_optimizer.step()
        discriminator_optimizer.step()

        # print loss
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{int(size/batch_size)}], G Loss: {gen_loss.item():.4f}, D Loss: {disc_loss.item():.4f}')
            writer.add_scalar("D_loss",disc_loss.item(), loss_num)
            writer.add_scalar("G_loss",gen_loss.item(), loss_num)
            loss_num += 1
            
    # eval
    generator.eval()

    # channel to img
    with torch.no_grad():
        H_gen = generator(Y_sign_test)   
        NMSE = 10 * torch.log10(calculate_nmse(H_gen, H_test))
        print(f'Epoch [{epoch+1}/{num_epochs}], NMSE: {NMSE.item():.4f}')
        writer.add_scalar("NSME",NMSE.cpu().data.numpy(),epoch)

        # norm
        H_fake_mod = torch.norm(H_gen, dim=1)   #  [N, 64, 32]
        H_real_mod = torch.norm(H_test, dim=1)  # [N, 64, 32]
        # mean
        H_fake_mean = torch.mean(H_fake_mod, dim=0)  # [64, 32]
        H_real_mean = torch.mean(H_real_mod, dim=0)  # [64, 32]
        # Tensor to NumPy
        H_fake_mean_np = H_fake_mean.cpu().numpy()
        H_real_mean_np = H_real_mean.cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(9, 5))  # Adjust the figure size

        # Plot the first image and colorbar
        im1 = axes[0].imshow(H_real_mean_np)
        axes[0].set_title('Target', fontsize=14, fontweight='bold')
        cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.1, pad=0.04)

        # Plot the second image and colorbar
        im2 = axes[1].imshow(H_fake_mean_np)
        axes[1].set_title('H_gen', fontsize=14, fontweight='bold')
        cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.1, pad=0.04)

        writer.add_figure('H_img', fig, epoch)  
         
writer.close()
# Save the generator network
torch.save(generator.state_dict(), 'results/generator.pth')