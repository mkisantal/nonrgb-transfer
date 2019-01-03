from torchvision.datasets import DatasetFolder
import rasterio
from functools import partial


def hsi_loader(path, channels=None):

    """ Loader for hyperspectral tif images, returns selected channels. """

    dataset_reader = rasterio.open(path)
    image = dataset_reader.read()
    if channels is not None:
        return image[channels, :, :]
    else:
        return image


class HsiImageFolder(DatasetFolder):

    """ ImageFolder dataset for hyperspectral images. """

    def __init__(self, root, transform=None, target_transform=None, channels=None):
        super(HsiImageFolder, self).__init__(root=root,
                                             loader=partial(hsi_loader, channels),
                                             extensions=['tif'],
                                             transform=transform,
                                             target_transform=target_transform)
        self.imgs = self.samples
        return


if __name__ == '__main__':

    loader = HsiImageFolder(root=r"C:\datasets\EuroSATallBands\ds\images\remote_sensing\otherDatasets\sentinel_2\tif",
                            channels=[1, 2, 3])
    print('done.')
