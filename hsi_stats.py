import rasterio
import os
from multiprocessing import Pool
import tqdm
import json
import numpy as np
import tifffile


def check_image_stats_eurosat(path):

    """ Check channelwise statistics for multispectral image. """

    image_stats = []
    img = rasterio.open(path).read()
    img = np.float64(img)
    for ch in range(img.shape[0]):
        image_stats.append({'min': img[ch, :, :].min().item(),
                            'max': img[ch, :, :].max().item(),
                            'mean': img[ch, :, :].mean().item(),
                            'var': img[ch, :, :].var().item(),
                            'std': img[ch, :, :].std().item(),
                            'sum_square_over_n': (img[ch, :, :]**2).mean().item()})
    return image_stats


def check_image_stats(path):

    """ Check channelwise statistics for multispectral image. """

    image_stats = []
    img = tifffile.imread(path)
    img = np.float64(img)
    for ch in range(img.shape[-1]):
        image_stats.append({'min': img[:, :, ch].min().item(),
                            'max': img[:, :, ch].max().item(),
                            'mean': img[:, :, ch].mean().item(),
                            'var': img[:, :, ch].var().item(),
                            'std': img[:, :, ch].std().item(),
                            'sum_square_over_n': (img[:, :, ch]**2).mean().item()})
    return image_stats


def check_dataset_stats(root_dir, save=False, eurosat=False, filename=None):

    """ Getting channelwise statistics for the dataset. """

    # collect all images
    image_paths = []
    if eurosat:
        class_folders = os.listdir(root_dir)
        for folder in class_folders:
            class_path = os.path.join(root_dir, folder)
            image_names = os.listdir(class_path)
            for image_name in image_names:
                image_paths.append(os.path.join(class_path, image_name))
        example_image = rasterio.open(image_paths[0]).read()
        num_channels = example_image.shape[0]
    else:
        image_names = os.listdir(root_dir)
        for image_name in image_names:
            image_paths.append(os.path.join(root_dir, image_name))
        example_image = tifffile.imread(image_paths[0])
        num_channels = example_image.shape[-1]
    num_images = len(image_paths)
    print('Calculating statistics on {} images.'.format(num_images))

    # checking image statistics separately
    all_image_stats = []
    check_image = check_image_stats_eurosat if eurosat else check_image_stats
    with Pool(1) as p:
        for stats in tqdm.tqdm(p.imap(check_image, image_paths), total=len(image_paths)):
            all_image_stats.append(stats)

    # aggregating statistics for the whole dataset
    channelwise_stats = []
    for i in range(num_channels):
        channelwise_stats.append({'min': 10e9, 'max': -1, 'mean': 0, 'var': 0, 'std': 0})
    means, ssqn = [[] for _ in range(num_channels)], [[] for _ in range(num_channels)]

    for j, stats in enumerate(all_image_stats):
        for i in range(num_channels):
            channelwise_stats[i]['min'] = min(stats[i]['min'], channelwise_stats[i]['min'])
            channelwise_stats[i]['max'] = max(stats[i]['max'], channelwise_stats[i]['max'])
            channelwise_stats[i]['mean'] += (stats[i]['mean'] - channelwise_stats[i]['mean']) / (j + 1)
            means[i].append(stats[i]['mean'])
            ssqn[i].append(stats[i]['sum_square_over_n'])
    for ch, channel in enumerate(channelwise_stats):
        sqm = (np.array(means[ch]).mean()**2).item()  # squared mean
        var = 1/num_images * np.array(ssqn[ch]).sum() - sqm  # dataset variance from image variances
        channel['var'], channel['std'] = var, var**.5
        print('min: {} \t max: {} \t mean: {:.02f} '.format(channel['min'], channel['max'], channel['mean'],) +
              '\t var: {:.02f} \t std: {:.02f}'.format(channel['var'], channel['std']))

    if save:
        filename = filename if filename is not None else 'spectrum_stats.json'
        with open(os.path.join(filename), 'w') as fout:
            json.dump(channelwise_stats, fout)

    return channelwise_stats


if __name__ == '__main__':

    # check_dataset_stats('/home/mate/dataset/EuroSATallBands_train', save=True)
    check_dataset_stats('/home/mate/datasets/grss/Track1-MSI_A_train', save=True, eurosat=False,
                        filename='gfc_channel_stats.json')
    # check_dataset_stats('C:\datasets\grss\debug', save=True, eurosat=False)
    # check_dataset_stats('C:\datasets\debug', save=True, eurosat=True)
    # check_dataset_stats('/datasets/EuroSATallBands_test', save=True)


