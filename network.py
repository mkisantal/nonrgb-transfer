from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np
import torch  # for debug only
import itertools
from backboned_unet import Unet  # https://github.com/mkisantal/backboned-unet


fixed_random_noise_vector = [0.33963535, -0.70369039,  0.62590457,  0.59152784,  0.4051563,
                             0.26512166,  0.25203669, -0.39983498,  0.66386131, -0.94438161]


class GeneratorResidualModule(nn.Module):

    """ Residual module based on torchvision.models.resnet.BasicBlock"""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(GeneratorResidualModule, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class PixelDAGenerator(nn.Module):

    """ Generator based on Pixel-Level Domain Adaptation paper (arXiv:1612.05424v2). """

    # TODO: consider adding random input vector

    def __init__(self, num_channels, latent_dim=10, im_size=224):
        super(PixelDAGenerator, self).__init__()
        self.fixed_noise = torch.tensor(fixed_random_noise_vector)
        self.noise_in = nn.Linear(latent_dim, im_size**2, bias=False)
        self.noise_bn = nn.BatchNorm1d(im_size**2)
        self.conv1 = nn.Conv2d(num_channels+1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64) # not included according to TF implementation
        self.relu = nn.ReLU(inplace=True)
        self.res_block1 = GeneratorResidualModule(64, 64)
        self.res_block2 = GeneratorResidualModule(64, 64)
        self.res_block3 = GeneratorResidualModule(64, 64)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x, z=None):
        if z is None:
            z = self.fixed_noise.repeat(x.shape[0], 1)
        z = self.noise_in(z.cuda())
        z = self.noise_bn(z)
        x = torch.cat([x, z.view([x.shape[0], 1, x.shape[2], x.shape[3]])], 1)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.final_conv(x)
        out = self.tanh(x)

        return out


class DiscriminatorModule(nn.Module):

    """ Conv-BN-LeakyRelu module for PixelDADiscriminator, and for PatchGAN. """

    def __init__(self, inplanes, kernel_size=3, stride=None, padding=1):
        super(DiscriminatorModule, self).__init__()
        planes = 64 if inplanes == 3 else 2*inplanes
        if stride is None:
            stride = 1 if inplanes == 3 else 2
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.lrelu = nn.LeakyReLU(.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class PixelDADiscriminator(nn.Module):

    """ Discriminator based on Pixel-Level Domain Adaptation paper (arXiv:1612.05424v2). """

    def __init__(self):
        super(PixelDADiscriminator, self).__init__()
        self.block1 = DiscriminatorModule(3)
        self.block2 = DiscriminatorModule(64)
        self.block3 = DiscriminatorModule(128)
        self.block4 = DiscriminatorModule(256)
        self.block5 = DiscriminatorModule(512)
        self.block6 = DiscriminatorModule(1024)
        self.block7 = DiscriminatorModule(2048)
        self.fc = nn.Linear(4096*4*4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_logit=True):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.fc(x.view(x.size(0), -1))
        out = self.sigmoid(x)
        if return_logit:
            return out, x
        else:
            return out


class PatchGanDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGanDiscriminator, self).__init__()
        # self.block1 = DiscriminatorModule(3, kernel_size=4, stride=2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.lrelu1 = nn.LeakyReLU(negative_slope=.2, inplace=True)
        self.block2 = DiscriminatorModule(64, kernel_size=4, stride=2)
        self.block3 = DiscriminatorModule(128, kernel_size=4, stride=2)
        self.block4 = DiscriminatorModule(256, kernel_size=4, stride=1)
        self.last_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.last_conv(x)
        out = self.sigmoid(x)
        return out, x


class PixelGanDiscriminator(nn.Module):
    def __init__(self):
        super(PixelGanDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.lrelu1 = nn.LeakyReLU(negative_slope=.2, inplace=True)
        self.block2 = DiscriminatorModule(64, kernel_size=1, stride=1, padding=0)
        self.last_conv = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.block2(x)
        x = self.last_conv(x)
        out = self.sigmoid(x)
        return out, x


class MultiChannelNet(nn.Module):

    """ Class for experiments with non-RGB inputs for pre-trained networks."""

    def __init__(self,
                 num_channels=3,
                 num_classes=10,
                 input_mode=None,
                 segmentation=True,
                 backbone_kwargs=None):
        super(MultiChannelNet, self).__init__()

        self.input_transform_module = None
        # self.rgb_net = models.resnet50(pretrained=True)
        if backbone_kwargs is not None:
            self.rgb_net = Unet(classes=num_classes)
        else:
            self.rgb_net = Unet(**backbone_kwargs)
        self.segmentation_mode = segmentation

        if input_mode == 'replace_conv1':
            conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if segmentation:
                self.rgb_net.backbone.conv1 = conv1
                nn.init.kaiming_normal_(self.rgb_net.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
            else:
                self.rgb_net.conv1 = conv1
                nn.init.kaiming_normal_(self.rgb_net.conv1.weight, mode='fan_out', nonlinearity='relu')
        if input_mode == 'domain_adapter':
            # defining network for input transformation (architecture based on PixelDA)
            self.input_transform_module = PixelDAGenerator(num_channels)
        self.conv1_replaced = input_mode == 'replace_conv1'

        # replace output layer
        if not segmentation:
            fc_in_features = self.rgb_net.fc.in_features
            self.rgb_net.fc = nn.Linear(fc_in_features, num_classes)

    def forward(self, x, z=None):

        """ Running inference on HSI with generator + classifier. """

        if self.input_transform_module is not None:
            three_channel_image = self.input_transform_module(x, z)
        else:
            three_channel_image = x
        output = self.rgb_net(three_channel_image)
        return output, three_channel_image

    def classify(self, x):

        """ Running only classification on 3 channel input"""

        x = self.rgb_net(x)
        return x

    def set_finetuning(self):

        """ Setting all model parameters trainable. """

        for subnetwork in [self.rgb_net, self.input_transform_module]:
            if subnetwork is not None:
                for param in subnetwork.parameters():
                    param.requires_grad = True
        return

    def set_feature_extracting(self):

        """ Freeze rgb net, replaced layers, generator and discriminator trainable. """

        if self.segmentation_mode:
            # only encoder part of U-net should be frozen
            self.rgb_net.freeze_encoder()
            if self.conv1_replaced:
                for param in self.rgb_net.backbone.conv1.parameters():
                    param.requires_grad = True
        else:
            for param in self.rgb_net.parameters():
                param.requires_grad = False
            for param in self.rgb_net.fc.parameters():
                param.requires_grad = True
            if self.conv1_replaced:
                for param in self.rgb_net.conv1.parameters():
                    param.requires_grad = True
        for subnetwork in [self.input_transform_module]:
            if subnetwork is not None:
                for param in subnetwork.parameters():
                    param.requires_grad = True
        return

    def get_transformed_input(self, x, pil=False):

        """ For inspecting the image we are feeding to the pre-trained network. """

        if self.input_transform_module is not None:
            x = self.input_transform_module(x)
        if not pil:
            return x
        else:
            x += x.min()
            x *= 255.0 / x.max()
            img = np.uint8(x.cpu().numpy()).transpose()
            return Image.fromarray(img)

    def get_params_for_opt(self, training_phase='generator'):

        """ Parameter iterators for initializing pytorch optimizers. """

        # TODO: remove this, use get_pretrained_parameters and get_random_initialized_parameters instead.
        if training_phase == 'generator':
            if self.input_transform_module is not None:
                return itertools.chain(self.input_transform_module.parameters(), self.rgb_net.parameters())
            else:
                return self.rgb_net.parameters()
        if training_phase == 'discriminator':
            return self.domain_discriminator.parameters()

    def get_pretrained_parameters(self):

        """ Returning iterator for pretrained parametes, to be used with PyTorch optimizer"""

        if self.segmentation_mode:
            return self.rgb_net.get_pretrained_parameters()
        else:
            raise NotImplementedError('Only segmentation for now.')

    def get_random_initialized_parameters(self):

        """ Returning iterator for new, randomly initialized parametes, to be used with PyTorch optimizer"""

        if self.segmentation_mode:
            iterators = [self.rgb_net.get_random_initialized_parameters()]
            if self.input_transform_module is not None:
                iterators.append(self.input_transform_module.parameters())
            return itertools.chain(*iterators)
        else:
            raise NotImplementedError('Only segmentation for now.')


if __name__ == '__main__':
    net = MultiChannelNet(num_channels=13,
                          num_classes=10,
                          input_mode='replace_conv1')
    hsi = torch.ones([1, 13, 224, 224])
    rgb = torch.ones([1, 3, 224, 224])
    # print(net)
    # net.set_feature_extracting()
    # net.set_finetuning()

    # input = torch.ones([1, 3, 224, 224])
    #
    # net2 = PixelDADiscriminator()
    # print(net2)
    #
    # net2(input)

    print('done.')
