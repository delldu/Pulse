import torch
from torch import nn
from torch.nn import functional as F
import pdb


class BicubicDownSample(nn.Module):
    def bicubic_kernel(self, x, a=-0.50):
        """
        This equation is exactly copied from the website below:
        https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic
        """
        abs_x = torch.abs(x)
        # x = tensor(-1.9844)
        # a = -0.5

        if abs_x <= 1.0:
            return (a + 2.0) * torch.pow(abs_x, 3.0) - (a + 3.0) * torch.pow(abs_x, 2.0) + 1
        elif 1.0 < abs_x < 2.0:
            return a * torch.pow(abs_x, 3) - 5.0 * a * torch.pow(abs_x, 2.0) + 8.0 * a * abs_x - 4.0 * a
        else:
            return 0.0

    def __init__(self, factor=4, cuda=True, padding="reflect"):
        super().__init__()
        self.factor = factor
        size = factor * 4

        k = torch.tensor(
            [self.bicubic_kernel((i - torch.floor(torch.tensor(size / 2)) + 0.5) / factor) for i in range(size)],
            dtype=torch.float32,
        )
        # factor = 32
        # cuda = True
        # padding = 'reflect'

        k = k / torch.sum(k)
        # k = torch.einsum('i,j->ij', (k, k))
        k1 = torch.reshape(k, shape=(1, 1, size, 1))
        self.k1 = torch.cat([k1, k1, k1], dim=0)
        k2 = torch.reshape(k, shape=(1, 1, 1, size))
        self.k2 = torch.cat([k2, k2, k2], dim=0)
        self.cuda = ".cuda" if cuda else ""
        self.padding = padding
        for param in self.parameters():
            param.requires_grad = False
        # (Pdb) k1.size(), k2.size()
        # ([1, 1, 128, 1], [1, 1, 1, 128])
        # (Pdb) self.k1.size(), self.k2.size()
        # ([3, 1, 128, 1], [3, 1, 1, 128])
        self.filters1 = self.k1.type("torch{}.FloatTensor".format(self.cuda))
        self.filters2 = self.k2.type("torch{}.FloatTensor".format(self.cuda))

    def forward(self, x):
        # x = torch.from_numpy(x).type('torch.FloatTensor')

        filter_height = self.factor * 4
        filter_width = self.factor * 4
        stride = self.factor

        pad_along_height = max(filter_height - stride, 0)
        pad_along_width = max(filter_width - stride, 0)

        # compute actual padding values for each side
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        # downscaling performed by 1-d convolution
        x = F.pad(x, (0, 0, pad_top, pad_bottom), self.padding)
        x = F.conv2d(input=x, weight=self.filters1, stride=(stride, 1), groups=3)

        x = F.pad(x, (pad_left, pad_right, 0, 0), self.padding)
        x = F.conv2d(input=x, weight=self.filters2, stride=(1, stride), groups=3)

        # (Pdb) x.size() -- [1, 3, 32, 32]

        return x
