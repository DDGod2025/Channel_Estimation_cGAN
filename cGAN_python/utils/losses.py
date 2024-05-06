import torch
import torch.nn.functional as F

BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()
L1Loss = torch.nn.L1Loss()

def discriminator_loss(disc_real_output, disc_generated_output):
    """disc_real_output = [real_target]
       disc_generated_output = [generated_target]
    """
    real_loss = BCEWithLogitsLoss(disc_real_output, torch.ones_like(disc_real_output))  # label=1
    generated_loss = BCEWithLogitsLoss(disc_generated_output, torch.zeros_like(disc_generated_output))  # label=0
    total_disc_loss = real_loss.mean() + generated_loss.mean()
    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target, l2_weight=100):
    """
        disc_generated_output: output of Discriminator when input is from Generator
        gen_output:  output of Generator (i.e., estimated H)
        target:  target image
        l2_weight: weight of L2 loss
    """
    # GAN loss
    gen_loss = BCEWithLogitsLoss(disc_generated_output, torch.ones_like(disc_generated_output))
    # L2 loss
    l2_loss = L1Loss(target, gen_output)
    total_gen_loss = gen_loss.mean() + l2_weight * l2_loss.mean()
    return total_gen_loss

def calculate_nmse(gen_output, output):
    mse = F.mse_loss(gen_output, output)
    nmse = mse / torch.mean(output ** 2)
    return nmse