from torchvision.datasets import DatasetFolder
import rasterio
from functools import partial
from torchvision import transforms
import torch
from scipy.ndimage.interpolation import zoom

def hsi_loader(channels, path):

    """ Loader for hyperspectral tif images, returns selected channels. """

    dataset_reader = rasterio.open(path)
    image = dataset_reader.read()
    if channels is not None:
        image = image[channels, :, :]
    # for now image transforms are done here, as PIL does not work with >3 channels
    # resize to from 64 to 224; don't use order >1 to avoid mixing channels
    image = zoom(image, zoom=[1, 3.5, 3.5], order=1, prefilter=False)
    image = torch.tensor(image)
    # float in [0, 1]



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



transforms.ToTensor

torch.from_numpy()


