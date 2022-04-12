import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    
    def __init__(self, vgg19_npy_path=None):
        
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        print('Finished loading vgg19.npy')

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
        conv.weight = nn.Parameter(filt)
        conv.bias = nn.Parameter(conv_biases)

        conv_relu = nn.Sequential(
            conv,
            nn.ReLU(inplace=True),
        )
        
        if relu:
            return conv_relu
        else:
            return conv

    # def fc_layer(self, bottom, name):
    #     with tf.variable_scope(name):
    #         shape = bottom.get_shape().as_list()
    #         dim = 1
    #         for d in shape[1:]:
    #             dim *= d
    #         x = tf.reshape(bottom, [-1, dim])

    #         weights = self.get_fc_weight(name)
    #         biases = self.get_bias(name)

    #         # Fully connected layer. Note that the '+' operation automatically
    #         # broadcasts the biases.
    #         fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

    #         return fc

    def get_conv_filter(self, name):
        return torch.Tensor(self.data_dict[name][0]).permute(3, 2, 0, 1) # or 3210 

    def get_bias(self, name):
        return torch.Tensor(self.data_dict[name][1])

    def get_fc_weight(self, name):
        return torch.Tensor(self.data_dict[name][0]).permute(3, 2, 0, 1)

vgg = Vgg19('vgg19_no_fc.npy')
img = torch.ones(1, 3, 128, 128)
output = vgg.build_conv4_4(img)
print(output.shape)
print(output[0][0][:3][:3])