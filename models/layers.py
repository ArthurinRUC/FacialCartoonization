import torch
import numpy as np
from torch import device, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm


VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg19:
    
    def __init__(self, vgg19_npy_path=None, device=None):
        
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        print('Finished loading vgg19.npy')

        self.device = device

        self.conv1_1 = self.conv_layer("conv1_1")
        self.conv1_2 = self.conv_layer("conv1_2")

        self.conv2_1 = self.conv_layer("conv2_1")
        self.conv2_2 = self.conv_layer("conv2_2")

        self.conv3_1 = self.conv_layer("conv3_1")
        self.conv3_2 = self.conv_layer("conv3_2")
        self.conv3_3 = self.conv_layer("conv3_3")
        self.conv3_4 = self.conv_layer("conv3_4")

        self.conv4_1 = self.conv_layer("conv4_1")
        self.conv4_2 = self.conv_layer("conv4_2")
        self.conv4_3 = self.conv_layer("conv4_3")
        self.conv4_4 = self.conv_layer("conv4_4", False)

        self.conv = nn.Sequential(
            self.conv1_1,
            self.conv1_2,
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv2_1,
            self.conv2_2,
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv3_1,
            self.conv3_2,
            self.conv3_3,
            self.conv3_4,
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv4_1,
            self.conv4_2,
            self.conv4_3,
            self.conv4_4,
        )


    def build_conv4_4(self, rgb, include_fc=False):
        
        img = (rgb+1) * 127.5
        red, green, blue = img.split(1, dim=1)
        bgr = torch.cat([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]], dim=1)

        return self.conv(bgr)


    def max_pool(self, bottom, name):
        return F.max_pool2d(bottom, kernel_size=2, stride=2)


    def conv_layer(self, name, relu=True):
        filt = self.get_conv_filter(name)
        conv_biases = self.get_bias(name)

        conv = nn.Conv2d(filt.shape[1], filt.shape[0], (filt.shape[2], filt.shape[3]), stride=1, padding='same')
        conv.weight = nn.Parameter(filt.to(self.device))
        conv.bias = nn.Parameter(conv_biases.to(self.device))

        conv_relu = nn.Sequential(
            conv,
            nn.ReLU(inplace=True),
        )
        
        if relu:
            return conv_relu
        else:
            return conv

    def get_conv_filter(self, name):
        return torch.Tensor(self.data_dict[name][0]).permute(3, 2, 0, 1) # or 3210 

    def get_bias(self, name):
        return torch.Tensor(self.data_dict[name][1])

    def get_fc_weight(self, name):
        return torch.Tensor(self.data_dict[name][0]).permute(3, 2, 0, 1)


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, epsilon=1e-5):
        super(AdaptiveInstanceNorm, self).__init__()

        self.epsilon = epsilon

    def forward(self, content, style):
        c_mean = torch.mean(content, dim=[2, 3], keepdim=True)
        c_var = torch.var(content, dim=[2, 3], keepdim=True)
        s_mean = torch.mean(style, dim=[2, 3], keepdim=True)
        s_var = torch.var(style, dim=[2, 3], keepdim=True)
        c_std, s_std = torch.sqrt(c_var + self.epsilon), torch.sqrt(s_var + self.epsilon)

        return s_std * (content - c_mean) / c_std + s_mean


class SelfAttention(nn.Module):

    def __init__(self, name='attention'):
        super(SelfAttention, self).__init__()

        self.name = name

    def forward(self, inputs):
        N, C, H, W = inputs.shape
        f = nn.Conv2d(inputs, C//8, [1, 1])
        g = nn.Conv2d(inputs, C//8, [1, 1])
        s = nn.Conv2d(inputs, 1, [1, 1])
        f_flatten = torch.reshape(f, shape=[f.shape[0], f.shape[1], -1])
        g_flatten = torch.reshape(g, shape=[g.shape[0], g.shape[1], -1])
        s_flatten = torch.reshape(s, shape=[s.shape[0], s.shape[1], -1])

        beta = torch.matmul(f_flatten, g_flatten.transpose(2, 3))
        beta = F.softmax(beta, dim=1)

        att_map = torch.matmul(beta, s_flatten)
        att_map = torch.reshape(att_map, shape=[N, 1, H, W])

        gamma = nn.Parameter(torch.FloatTensor(0), requires_grad=True)

        output = att_map * gamma + inputs
        return att_map, output


class ResBlock(nn.Module):

    def __init__(self, in_channels=32, out_channels=32, name='resblock') -> None:
        super(ResBlock, self).__init__()

        self.name = name
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, [3, 3], padding='same'),
            # nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, [3, 3], padding='same'),
            # nn.InstanceNorm2d(out_channels, affine=True)
        )

    def forward(self, input):
        x = self.Conv(input)
        return x + input


class DownSample(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.ConvReLU_2 = nn.Sequential(
            nn.ZeroPad2d(padding=(0, 1, 0, 1)),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2),
            # nn.InstanceNorm2d(in_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            # nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.ConvReLU_2(x)
        return x



class UpSample(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.FirstConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            # nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

        self.ConvReLU = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            # nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.FirstConv(x1)

        x1 = F.upsample(x1, scale_factor=2, mode='bilinear', align_corners=False)
        
        assert x1.shape[:-2] == x2.shape[:-2]
        x = x1 + x2

        x = self.ConvReLU(x)
        return x1



class UnetGenerator(nn.Module):

    def __init__(self, channel=32, num_blocks=4, in_channels=3, out_channels=3) -> None:
        super().__init__()

        self.FirstConv = nn.Sequential(
            nn.Conv2d(in_channels, channel, 7, padding="same"),
            # nn.InstanceNorm2d(channel, affine=True),
            nn.LeakyReLU(inplace=True),
        )
        self.DownSample1 = DownSample(channel, channel*2)
        self.DownSample2 = DownSample(channel*2, channel*4)
        self.ResBlock = nn.Sequential()
        for i in range(num_blocks):
            self.ResBlock.add_module(f'ResBlock_{i}', ResBlock(channel*4, channel*4))
        self.UpSample1 = UpSample(channel*4, channel*2)
        self.UpSample2 = UpSample(channel*2, channel)
        self.LastConv = nn.Sequential(
            nn.Conv2d(channel, out_channels, 7, padding="same"),
            nn.Tanh(),
        )

        self.num_blocks = num_blocks


    def forward(self, x):
        x1 = self.FirstConv(x)
        x2 = self.DownSample1(x1)
        x3 = self.DownSample2(x2)
        x3 = self.ResBlock(x3)
        x4 = self.UpSample1(x3, x2)
        x5 = self.UpSample2(x4, x1)
        x = self.LastConv(x5)
        return x


class ConvSpectralNorm(nn.Module):

    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)),
            # nn.InstanceNorm2d(out_channel, affine=True),
            nn.LeakyReLU(inplace=True),
            spectral_norm(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding="same")),
            # nn.InstanceNorm2d(out_channel, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):

    def __init__(self, in_channel, channel=32, patch=True) -> None:
        super().__init__()

        self.ConvSN1 = ConvSpectralNorm(in_channel, channel)
        self.ConvSN2 = ConvSpectralNorm(channel, channel*2)
        self.ConvSN3 = ConvSpectralNorm(channel*2, channel*4)
        self.LastConv = nn.Sequential(
            spectral_norm(nn.Conv2d(channel*4, 1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.ConvSN1(x)
        x = self.ConvSN2(x)
        x = self.ConvSN3(x)
        x = self.LastConv(x)

        return x

# class Encoder(nn.Module):

#     def __init__(self, channel=32, in_channels=3) -> None:
#         super().__init__()

#         ## consider BN layers

#         self.Encoder = nn.Sequential(
#             nn.Conv2d(in_channels, channel, kernel_size=7),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(channel, channel*2, kernel_size=3, stride=2),
#             nn.Conv2d(channel*2, channel*2, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(channel*2, channel*4, kernel_size=3, stride=2),
#             nn.Conv2d(channel*4, channel*4, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#         )
    
#     def forward(self, x):
#         return self.Encoder(x)

# class Decoder(nn.Module):

#     def __init__(self, channel=32, out_channels=3) -> None:
#         super().__init__()

#         ## consider BN layers

#         self.Decoder = nn.Sequential(
#             nn.ConvTranspose2d(channel*4, channel*2, kernel_size=3, stride=2),
#             nn.Conv2d(channel*2, channel*2, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             nn.ConvTranspose2d(channel*2, channel, kernel_size=3, stride=2),
#             nn.Conv2d(channel, channel, kernel_size=3),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(channel, out_channels, kernel_size=7),
#         )
    
#     def forward(self, x):
#         return self.Decoder(x)


# class Generator(nn.Module):

#     def __init__(self, channel=32, num_blocks=4, in_channels=3, out_channels=3) -> None:
#         super(Generator, self).__init__()

#         self.num_blocks = num_blocks
#         self.Encoder = Encoder(channel, in_channels)
#         self.ResBlock = ResBlock(channel*4, channel*4)
#         self.Decoder = Decoder(channel, out_channels)

#     def forward(self, x):
#         x = self.Encoder(x)
#         for _ in range(self.num_blocks):
#             x = self.ResBlock(x)
#         x = self.Decoder(x)
#         return x      
    
    
# def disc_bn(x, scale=1, channel=32, is_training=True, 
#             name='discriminator', patch=True, reuse=tf.AUTO_REUSE):
    
#     with tf.variable_scope(name, reuse=reuse):
        
#         for idx in range(3):
#             x = slim.convolution2d(x, channel*2**idx, [3, 3], stride=2, activation_fn=None)
#             x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
#             x = tf.nn.leaky_relu(x)
            
#             x = slim.convolution2d(x, channel*2**idx, [3, 3], activation_fn=None)
#             x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
#             x = tf.nn.leaky_relu(x)

#         if patch == True:
#             x = slim.convolution2d(x, 1, [1, 1], activation_fn=None)
#         else:
#             x = tf.reduce_mean(x, axis=[1, 2])
#             x = slim.fully_connected(x, 1, activation_fn=None)
        
#         return x


# def disc_sn(x, scale=1, channel=32, patch=True, name='discriminator', reuse=tf.AUTO_REUSE):
#     with tf.variable_scope(name, reuse=reuse):

#         for idx in range(3):
#             x = layers.conv_spectral_norm(x, channel*2**idx, [3, 3], 
#                                           stride=2, name='conv{}_1'.format(idx))
#             x = tf.nn.leaky_relu(x)
            
#             x = layers.conv_spectral_norm(x, channel*2**idx, [3, 3], 
#                                           name='conv{}_2'.format(idx))
#             x = tf.nn.leaky_relu(x)
        
        
#         if patch == True:
#             x = layers.conv_spectral_norm(x, 1, [1, 1], name='conv_out'.format(idx))
            
#         else:
#             x = tf.reduce_mean(x, axis=[1, 2])
#             x = slim.fully_connected(x, 1, activation_fn=None)
        
#         return x


# def disc_ln(x, channel=32, is_training=True, name='discriminator', patch=True, reuse=True):
#     with tf.variable_scope(name, reuse=reuse):

#         for idx in range(3):
#             x = slim.convolution2d(x, channel*2**idx, [3, 3], stride=2, activation_fn=None)
#             x = tf.contrib.layers.layer_norm(x)
#             x = tf.nn.leaky_relu(x)
            
#             x = slim.convolution2d(x, channel*2**idx, [3, 3], activation_fn=None)
#             x = tf.contrib.layers.layer_norm(x)
#             x = tf.nn.leaky_relu(x)

#         if patch == True:
#             x = slim.convolution2d(x, 1, [1, 1], activation_fn=None)
#         else:
#             x = tf.reduce_mean(x, axis=[1, 2])
#             x = slim.fully_connected(x, 1, activation_fn=None)
        
#         return x

            
if __name__ == '__main__':
    pass
    
   
