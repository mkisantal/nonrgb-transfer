from torchvision import models
import torch.nn as nn


class MultiChannelNet(nn.Module):

    """ Class for experiments with non-RGB inputs for pre-trained networks."""

    def __init__(self,
                 num_channels=3,
                 num_classes=10,
                 replace_rgb_input=False):
        super(MultiChannelNet, self).__init__()

        self.input_transform = None
        self.rgb_net = models.resnet50(pretrained=True)

        if replace_rgb_input:
            self.rgb_net.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(self.rgb_net.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.conv1_replaced = replace_rgb_input


        # replace output layer
        fc_in_features = self.rgb_net.fc.in_features
        self.rgb_net.fc = nn.Linear(fc_in_features, num_classes)

    def forward(self, x):

        """ Running inference on model. """

        if self.input_transform is not None:
            x = self.input_transform(x)
        x = self.rgb_net(x)
        return x

    def set_finetuning(self):

        """ Setting all model parameters trainable. """

        for param in self.rgb_net.parameters():
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
        return


if __name__ == '__main__':
    net = MultiChannelNet(num_channels=6,
                          num_classes=11,
                          replace_rgb_input=True)
    # print(net)
    net.set_feature_extracting()
    net.set_finetuning()
    print('done.')
