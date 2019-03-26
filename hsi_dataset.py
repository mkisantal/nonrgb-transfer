from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset
import rasterio
import tifffile
from functools import partial
import torch
from scipy.ndimage.interpolation import zoom
import numpy as np
import os
from random import shuffle, randint
from shutil import copyfile
import json
from multiprocessing import Pool
import tqdm
from torch.utils.data import DataLoader

# getting statistics ready
try:
    with open('spectrum_stats.json', 'r') as f:
        spectrum_stats = json.load(f)
except FileNotFoundError:
    msg = 'Channelwise statistics for multispectral image normalization not available.'
    msg += '\nCalculate statistics by running hsi_stats.py!'
    raise ImportError(msg)

spectrum_means = np.expand_dims(np.expand_dims(np.array([x['mean'] for x in spectrum_stats]), 1), 1).astype('float32')
spectrum_stds = np.expand_dims(np.expand_dims(np.array([x['std'] for x in spectrum_stats]), 1), 1).astype('float32')


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
        image /= spectrum_stds[chs, :, :]

    return torch.tensor(image)


def hsi_loader_gfc(chs, path, dataset_stats=None):

    """ Loader for GFC hyperspectral tif images, preprocesses and returns selected channels. """

    image = tifffile.imread(path)
    image = np.float32(image).swapaxes(0, -1)  # converting to channel first

    if chs is not None:
        image = image[chs, :, :]
    else:
        chs = np.arange(0, 8)
    # for now image transforms are done here, as PIL does not work with >3 channels
    # resize; don't use order >1 to avoid mixing channels
    # image = zoom(image, zoom=[1, 3.5, 3.5], order=1, prefilter=False)  # 20 times slower than RGB processing!!
    # normalize to zero mean and unit variance
    if dataset_stats is not None:
        image -= dataset_stats['spectrum_means'][chs, :, :]
        image /= dataset_stats['spectrum_stds'][chs, :, :]

    return torch.tensor(image)


def segmentation_gt_loader(path):

    """ Loading segmentation ground truth images for segmentation. """

    image = tifffile.imread(path)
    indices1 = [2, 5, 6, 9, 17, 65]
    indices2 = [0, 1, 2, 3, 4, 255]
    for from_index, to_index in zip(indices1, indices2):
        image[image == from_index] = to_index
    return image


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
        image /= spectrum_stds[chs, :, :]

    return torch.tensor(image)


def split_dataset(root_dir, split=[.8, .2], convert=False, dataset_suffix='', classification=True,
                  image_limit=None, semi_supervised=False):

    """ Splitting dataset to train (val) and test sets. Assumes that samples are in subfolders by classes. """

    dataset_name = os.path.basename(os.path.normpath(root_dir))

    split = np.array(split, dtype=np.float)
    split /= split.sum()

    if split.shape[0] == 2:
        split_names = ['train', 'val']
    elif split.shape[0] == 3:
        split_names = ['train', 'val', 'test']
    else:
        raise ValueError('Split should be length 2 (train, test) or 3 (train, val, test)')

    if semi_supervised:
        split_names.append('minitrain')
        if image_limit is None:
            raise ValueError('You need to specify \'per_category_image_limit\' to create semi-supervised dataset.')

    # create root folders for dataset partitions
    split_roots = []
    for name in split_names:
        split_root = os.path.join(root_dir, '..', '{}_{}{}'.format(dataset_name, dataset_suffix, name))
        if not os.path.exists(split_root):
            os.makedirs(split_root)
        split_roots.append(split_root)

    if classification:
        split_classification(root_dir, split, split_roots, semi_supervised, image_limit, convert)
    else:
        split_segmentation(root_dir, split, split_roots, semi_supervised, image_limit, convert)

    return


def split_segmentation(root_dir, split, split_roots, semi_supervised, image_limit, convert):

    """ Splitting the dataset to train, val and test sets. Making sure scenes are separated! """

    image_names = os.listdir(root_dir)
    shuffle(image_names)

    multiview_dict = {}
    for v, image_name in enumerate(image_names):
        view_code = image_name[:7]
        if view_code not in multiview_dict.keys():
            multiview_dict[view_code] = []
        multiview_dict[view_code].append(image_name)

    scenes = list(multiview_dict.keys())
    shuffle(scenes)

    source_image_paths = []
    destination_image_paths = []

    num_images_in_split = [int(x * len(image_names)) for x in split]
    print('{}: {}'.format('Estimated number of images in splits: ', num_images_in_split))
    split_limits = np.append([0], np.cumsum(num_images_in_split))
    split_limits[-1] = len(image_names)

    view_counter, scene_counter = 0, -1
    for i in range(split_limits.shape[0] - 1):
        scene_counter += 1  # always move multiview pointer when new subset started
        if len(scenes) == scene_counter:
            break
        view_counter = 0
        for j in range(split_limits[i], split_limits[i + 1]):
            if len(multiview_dict[scenes[scene_counter]]) == view_counter:
                scene_counter += 1
                view_counter = 0
                if len(scenes) == scene_counter:
                    break
            view = multiview_dict[scenes[scene_counter]][view_counter]
            source_image_paths.append(os.path.join(root_dir, view))
            destination_image_paths.append(os.path.join(split_roots[i], view))
            view_counter += 1
            if semi_supervised and view_counter < image_limit:
                source_image_paths.append(os.path.join(root_dir, view))
                destination_image_paths.append(os.path.join(split_roots[-1], view))
            if image_limit is not None and j - split_limits[i] == image_limit - 1:
                break

    src_dst_dicts = [{'src': src, 'dst': dst} for src, dst in zip(source_image_paths, destination_image_paths)]
    with Pool(8) as p:
        for _ in tqdm.tqdm(p.imap(copy_image, src_dst_dicts), total=len(src_dst_dicts)):
            pass
    return


def split_classification(root_dir, split, split_roots, semi_supervised, per_category_image_limit, convert):
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
        shuffle(image_names)
        for i, image_name in enumerate(image_names):
            source_image_paths.append(os.path.join(category_path, image_name))
            if semi_supervised and i < per_category_image_limit:
                # duplicate source img for adding to minitrain too
                source_image_paths.append(os.path.join(category_path, image_name))

        # divide images to partitions
        destination_image_paths = []
        num_images_in_split = [int(x * len(image_names)) for x in split]
        print('{}: {}'.format('Number of images in splits:', num_images_in_split))
        split_limits = np.append([0], np.cumsum(num_images_in_split))
        split_limits[-1] = len(image_names)
        for i in range(split_limits.shape[0] - 1):
            for j in range(split_limits[i], split_limits[i + 1]):
                destination_image_paths.append(os.path.join(split_category_paths[i], image_names[j]))
                if semi_supervised and i == 0 and j < per_category_image_limit:
                    # add the image to minitrain as well
                    destination_image_paths.append(os.path.join(split_category_paths[-1], image_names[j]))

        if convert:
            src_dst_dicts = [{'src': src, 'dst': dst[:-4]} for src, dst in zip(source_image_paths,
                                                                               destination_image_paths)]
            with Pool(8) as p:
                for _ in tqdm.tqdm(p.imap(process_and_copy_image, src_dst_dicts), total=len(src_dst_dicts)):
                    pass
        else:
            for src, dst in zip(source_image_paths, destination_image_paths):
                copyfile(src, dst)
        print('\n\n')
    return


def copy_image(src_dst_dict):

    """ Simple function for parallelization. """

    copyfile(src_dst_dict['src'], src_dst_dict['dst'])
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


class HsiSegmentationDataset(Dataset):

    """ Dataset for HSI semantic segmentation datasets. """

    def __init__(self, root, gt_root, channels=None):
        print(root)
        super(HsiSegmentationDataset, self).__init__()

        self.root, self.gt_root = root, gt_root
        self.image_list = os.listdir(root)
        self.channels = channels
        shuffle(self.image_list)

        with open('gfc_channel_stats.json', 'r') as file:
            ds_stats = json.load(file)
        self.dataset_stats = dict()
        self.dataset_stats['spectrum_means'] = np.expand_dims(np.expand_dims(np.array(
            [x['mean'] for x in ds_stats]), 1), 1).astype('float32')
        self.dataset_stats['spectrum_stds'] = np.expand_dims(np.expand_dims(np.array(
            [x['std'] for x in ds_stats]), 1), 1).astype('float32')

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        gt_name = image_name[:-7] + 'CLS.tif'
        hsi_path = os.path.join(self.root, image_name)
        gt_path = os.path.join(self.gt_root, gt_name)
        image = hsi_loader_gfc(chs=self.channels, path=hsi_path, dataset_stats=self.dataset_stats)
        segmentation = segmentation_gt_loader(gt_path)
        return image, segmentation


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
    # split_dataset("/home/mate/dataset/EuroSATallBands", split=[.8, .2], convert=True, dataset_suffix='semi_',
    #               per_category_image_limit=300, semi_supervised=True)
    # a = hsi_loader([1], r'C:\datasets\EuroSATallBands_train\Residential\Residential_1006.tif')
    # split_dataset(root_dir='/home/mate/datasets/grss/Track1-MSI', classification=False, image_limit = None,
    #               dataset_suffix='A_')

    dataset = HsiSegmentationDataset(root=r'C:\datasets\grss\debug', gt_root=r'c:\datasets\grss\Track1-Truth')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    i = iter(dataloader)
    img, segm = next(i)
    print(np.unique(segm.numpy()))
    print('done.')




