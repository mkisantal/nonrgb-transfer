from torchvision.datasets import DatasetFolder
import rasterio
from functools import partial
from torchvision import transforms
import torch
from scipy.ndimage.interpolation import zoom
import numpy as np
import os
from random import shuffle, randint
from shutil import copyfile
import json
from multiprocessing import Pool
import tqdm



# getting statistics ready
try:
    with open('spectrum_stats.json', 'r') as f:
        spectrum_stats = json.load(f)
except FileNotFoundError:
    msg = 'Channelwise statistics for multispectral image normalization not available.'
    msg += '\nCalculate statistics by running hsi_stats.py!'
    raise ImportError(msg)

spectrum_means = np.expand_dims(np.expand_dims(np.array([x['mean'] for x in spectrum_stats]), 1), 1).astype('float32')
spectrum_vars = np.expand_dims(np.expand_dims(np.array([x['var'] for x in spectrum_stats]), 1), 1).astype('float32')


def hsi_loader(chs, path, normalize=True):

    """ Loader for hyperspectral tif images, preprocesses and returns selected channels. """

    dataset_reader = rasterio.open(path)
    image = dataset_reader.read().astype('float32')
    if chs is not None:
        image = image[chs, :, :]
    else:
        chs = np.arange(0, 13)
    # for now image transforms are done here, as PIL does not work with >3 channels
    # resize to from 64 to 224; don't use order >1 to avoid mixing channels
    image = zoom(image, zoom=[1, 3.5, 3.5], order=1, prefilter=False)  # 20 times slower than RGB processing!!
    # normalize to zero mean and unit variance
    if normalize:
        image -= spectrum_means[chs, :, :]
        image /= spectrum_vars[chs, :, :]

    return torch.tensor(image)


def npy_hsi_loader(chs, path, normalize=True):

    """ Loader for converted npy hyperspectral images, preprocesses and returns selected channels. """

    # TODO: merge this fcn with hsi_loader if possible

    image = np.load(path)
    if chs is not None:
        image = image[chs, :, :]
    else:
        chs = np.arange(0, 13)
    # for now image transforms are done here, as PIL does not work with >3 channels
    # resize to from 64 to 224; don't use order >1 to avoid mixing channels
    # image = zoom(image, zoom=[1, 3.5, 3.5], order=1, prefilter=False) # Done at conversion instead.
    # normalize to zero mean and unit variance
    if normalize:
        image -= spectrum_means[chs, :, :]
        image /= spectrum_vars[chs, :, :]

    return torch.tensor(image)


def split_dataset(root_dir, split=[.8, .2], convert=False, dataset_suffix=''):

    """ Splitting dataset to train (val) and test sets. Optionally converting to .npy for faster loading. """

    dataset_name = os.path.basename(os.path.normpath(root_dir))

    split = np.array(split, dtype=np.float)
    split /= split.sum()

    if split.shape[0] == 2:
        split_names = ['train', 'test']
    elif split.shape[0] == 3:
        split_names = ['train', 'val', 'test']
    else:
        raise ValueError('Split should be length 2 (train, test) or 3 (train, val, test)')

    # create root folders for dataset partitions
    split_roots = []
    for name in split_names:
        split_root = os.path.join(root_dir, '..', '{}_{}{}'.format(dataset_name, dataset_suffix, name))
        os.makedirs(split_root)
        split_roots.append(split_root)

    # loop over classes
    category_folders = os.listdir(root_dir)
    for category in category_folders:
        category_path = os.path.join(root_dir, category)
        print('{}: {}'.format('category', category))

        # create category folders in partitions
        split_category_paths = []
        for split_root in split_roots:
            split_category_path = os.path.join(split_root, category)
            os.makedirs(split_category_path)
            split_category_paths.append(split_category_path)

        # list source images
        image_names = os.listdir(category_path)
        source_image_paths = []
        for image_name in image_names:
            source_image_paths.append(os.path.join(category_path, image_name))

        # divide images to partitions
        destination_image_paths = []
        shuffle(image_names)
        num_images_in_split = [int(x*len(image_names)) for x in split]
        print('{}: {}'.format('Number of images in splits:', num_images_in_split))
        split_limits = np.append([0], np.cumsum(num_images_in_split))
        split_limits[-1] = len(image_names)
        for i in range(split_limits.shape[0]-1):
            for j in range(split_limits[i], split_limits[i+1]):
                destination_image_paths.append(os.path.join(split_category_paths[i], image_names[j]))

        if convert:
            src_dst_dicts = [{'src': src, 'dst': dst[:-4]} for src, dst in zip(source_image_paths, destination_image_paths)]
            with Pool(8) as p:
                for _ in tqdm.tqdm(p.imap(process_and_copy_image, src_dst_dicts), total=len(src_dst_dicts)):
                    pass
        else:
            for src, dst in zip(source_image_paths, destination_image_paths):
                copyfile(src, dst)
        print('\n\n')

    return


def process_and_copy_image(src_dst_dict):

    """ Loading HSI tif, converting to npy, more other slow processing steps. """

    dataset_reader = rasterio.open(src_dst_dict['src'])
    image = dataset_reader.read().astype('float32')
    image = zoom(image, zoom=[1, 3.5, 3.5], order=1, prefilter=False)  # scipy zoom is slow, better do it here
    # TODO: optionally more pre-processing steps
    np.save(src_dst_dict['dst'], image)
    return True


class HsiImageFolder(DatasetFolder):

    """ ImageFolder dataset for hyperspectral images. """

    def __init__(self, root, transform=None, target_transform=None, channels=None, npy=False):
        super(HsiImageFolder, self).__init__(root=root,
                                             loader=partial(npy_hsi_loader, channels) if npy
                                               else partial(hsi_loader, channels),
                                             extensions=['npy'] if npy else ['tif'],
                                             transform=transform,
                                             target_transform=target_transform)
        self.imgs = self.samples
        return


def load_single_image(idx=None, category=None, root=None, channels=None, normalize=True):
    if root is None:
        root = '/home/mate/dataset/EuroSATallBands'

    if category is None:
        category = randint(0, 9)
    category_list = os.listdir(root)
    category_list.sort()
    image_list = os.listdir(os.path.join(root, category_list[category]))
    if idx is None:
        idx = randint(0, len(image_list))
    path = os.path.join(root, category_list[category], image_list[idx])
    return hsi_loader(channels, path, normalize=normalize)


if __name__ == '__main__':

    # tests
    # split_dataset(r"C:\datasets\EuroSATallBands", split=[.8, .2], convert=True, dataset_suffix='npy_')
    # split_dataset("/home/mate/dataset/EuroSATallBands", split=[.8, .2], convert=True, dataset_suffix='npy_')
    # a = hsi_loader([1], r'C:\datasets\EuroSATallBands_train\Residential\Residential_1006.tif')
    print('done.')




