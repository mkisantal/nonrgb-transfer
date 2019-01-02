from torchvision import models
import torch.nn as nn


class MultiChannelNet(nn.Module):

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

        # replace output layer
        fc_in_features = self.rgb_net.fc.in_features
        self.rgb_net.fc = nn.Linear(fc_in_features, num_classes)

    def forward(self, x):

        if self.input_transform is not None:
            x = self.input_transform(x)

        x = self.rgb_net(x)

        return x


if __name__ == '__main__':
    net = MultiChannelNet(num_channels=6,
                          num_classes=11,
                          replace_rgb_input=True)
    print(net)
    print('done.')
