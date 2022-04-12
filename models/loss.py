import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg19, vgg16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vgg_model = vgg19(pretrained=True).to(device).features
vgg_model.eval()

L1_loss = nn.L1Loss().to(device)
BCE_loss = nn.BCELoss().to(device)


def vggloss(image_a, image_b):
    vgg_a = vgg_model(image_a)
    vgg_b = vgg_model(image_b)
    vgg_loss = L1_loss(vgg_a, vgg_b)
    return vgg_loss


def lsgan_loss(discriminator, real, fake):
    
    real_logit = discriminator(real)
    fake_logit = discriminator(fake)

    g_loss = torch.mean((fake_logit - 1) ** 2)
    d_loss = 0.5 * (torch.mean((real_logit - 1) ** 2) + torch.mean(fake_logit ** 2))

    return d_loss, g_loss


def total_variation_loss(image, k_size=1):
    H, W = image.shape[2:]
    tv_h = torch.mean((image[:, :, k_size:, :] - image[:, :, :H - k_size, :]) ** 2)
    tv_w = torch.mean((image[:, :, :, k_size:] - image[:, :, :, :W - k_size]) ** 2)
    tv_loss = (tv_h + tv_w) / (3 * H * W)
    return tv_loss


# def wgan_loss(discriminator, real, fake, patch=True, 
#               channel=32, name='discriminator', lambda_=2):
#     real_logits = discriminator(real, patch=patch, channel=channel, name=name, reuse=False)
#     fake_logits = discriminator(fake, patch=patch, channel=channel, name=name, reuse=True)

#     d_loss_real = - torch.mean(real_logits)
#     d_loss_fake = torch.mean(fake_logits)

#     d_loss = d_loss_real + d_loss_fake
#     g_loss = - d_loss_fake

#     """ Gradient Penalty """
#     # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
#     # alpha = tf.random_uniform([tf.shape(real)[0], 1, 1, 1], minval=0.,maxval=1.)
#     alpha = torch.Tensor(real.shape[0], 1, 1, 1).uniform_(0., 1.)
#     differences = fake - real # This is different from MAGAN
#     interpolates = real + (alpha * differences)
#     inter_logit = discriminator(interpolates, channel=channel, name=name, reuse=True)
#     # gradients = tf.gradients(inter_logit, [interpolates])[0]
#     gradients = torch.autograd.grad(inter_logit, interpolates)[0]
#     slopes = torch.sqrt(torch.reduce_sum(torch.square(gradients), reduction_indices=[1]))
#     gradient_penalty = torch.mean((slopes - 1.) ** 2)
#     d_loss += lambda_ * gradient_penalty
    
#     return d_loss, g_loss


# def gan_loss(discriminator, real, fake, scale=1,channel=32, patch=False, name='discriminator'):

#     real_logit = discriminator(real, scale, channel, name=name, patch=patch, reuse=False)
#     fake_logit = discriminator(fake, scale, channel, name=name, patch=patch, reuse=True)

#     real_logit = F.sigmoid(real_logit)
#     fake_logit = F.sigmoid(fake_logit)
    
#     g_loss_blur = -torch.mean(torch.log(fake_logit)) 
#     d_loss_blur = -torch.mean(torch.log(real_logit) + torch.log(1. - fake_logit))

#     return d_loss_blur, g_loss_blur


if __name__ == '__main__':
    pass


    