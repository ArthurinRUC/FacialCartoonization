import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def forward(self, x):
        assert x.dim() == 4
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        in_channels = x.shape[1]
        out_channels = in_channels
        weights = 1 / ((2 * self.r + 1) ** 2)
        box_kernel = weights * torch.ones((out_channels, 1, 2 * self.r + 1, 2 * self.r + 1))
        box_kernel = box_kernel.to(device)
        output = F.conv2d(x, box_kernel, groups=in_channels, padding='same')
        return output


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=0.1):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(
            Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b

if __name__ == '__main__':
    pass
