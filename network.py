from torchvision import models
import torch.nn as nn


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

    """ Generator based on Pixel-Level Domain Adaptation Paper """

    def __init__(self, num_channels):
        super(PixelDAGenerator, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        # self.bn1 = nn.BatchNorm2d(64) # not included according to TF implementation
        self.relu = nn.ReLU(inplace=True)
        self.res_block1 = GeneratorResidualModule(64, 64)
        self.res_block2 = GeneratorResidualModule(64, 64)
        self.res_block3 = GeneratorResidualModule(64, 64)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.final_conv(x)
        out = self.tanh(x)

        return out


class MultiChannelNet(nn.Module):

    """ Class for experiments with non-RGB inputs for pre-trained networks."""

    def __init__(self,
                 num_channels=3,
                 num_classes=10,
                 input_mode=None):
        super(MultiChannelNet, self).__init__()

        self.input_transform_module = None
        self.rgb_net = models.resnet50(pretrained=True)

        if input_mode == 'replace_conv1':
            self.rgb_net.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(self.rgb_net.conv1.weight, mode='fan_out', nonlinearity='relu')
        if input_mode == 'domain_adapter':
            # defining network for input transformation (architecture based on PixelDA)
            self.input_transform_module = PixelDAGenerator(num_channels)

        self.conv1_replaced = input_mode == 'replace_conv1'

        # replace output layer
        fc_in_features = self.rgb_net.fc.in_features
        self.rgb_net.fc = nn.Linear(fc_in_features, num_classes)

    def forward(self, x):

        """ Running inference on model. """

        if self.input_transform_module is not None:
            x = self.input_transform_module(x)
        x = self.rgb_net(x)
        return x

    def set_finetuning(self):

        """ Setting all model parameters trainable. """

        for param in self.rgb_net.parameters():
            param.requires_grad = True
        if self.input_transform_module is not None:
            for param in self.input_transform_module.parameters():
                param.requires_grad = True
        return

    def set_feature_extracting(self):

        """ Freeze model parameters, except for the last and optionally the first layers. """

        for param in self.rgb_net.parameters():
            param.requires_grad = False
        for param in self.rgb_net.fc.parameters():
            param.requires_grad = True
        if self.conv1_replaced:
            for param in self.rgb_net.conv1.parameters():
                param.requires_grad = True
        if self.input_transform_module is not None:
            for param in self.input_transform_module.parameters():
                param.requires_grad = True
        return


if __name__ == '__main__':
    net = MultiChannelNet(num_channels=6,
                          num_classes=11,
                          input_mode=None)
    # print(net)
    net.set_feature_extracting()
    net.set_finetuning()
    print('done.')
