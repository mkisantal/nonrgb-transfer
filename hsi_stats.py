import rasterio
import os
from multiprocessing import Pool
import tqdm
import json
import numpy as np


def load_channels(path):

    """ Load all channels for multispectral image. """

    channels = []
    img = rasterio.open(path).read()
    for ch in range(img.shape[0]):
        reshaped_channel = np.reshape(img[ch, :, :], [1, -1])
        channels.append(reshaped_channel)
    return channels


def check_dataset_stats(root_dir, save=False):

    """ Getting channelwise statistics for the dataset. """

    class_folders = os.listdir(root_dir)
    image_paths = []
    for folder in class_folders:
        class_path = os.path.join(root_dir, folder)
        image_names = os.listdir(class_path)
        for image_name in image_names:
            image_paths.append(os.path.join(class_path, image_name))

    print(len(image_paths))
    # inspect an image to count channels
    channelwise_stats = []
    image = rasterio.open(image_paths[0]).read()
    num_channels = image.shape[0]
    for i in range(num_channels):
        channelwise_stats.append({'min': 10e9, 'max': -1, 'mean': 0, 'var': 0, 'std': 0})

    all_image_channels = []
    with Pool(8) as p:
        for channels in tqdm.tqdm(p.imap(load_channels, image_paths), total=len(image_paths)):
            all_image_channels.append(channels)

    # for image_path in image_paths:
    #     all_image_channels.append(load_channels(image_path))

    separated_channels = [[] for _ in range(num_channels)]
    for image_channels in all_image_channels:
        for i in range(num_channels):
            separated_channels[i].append(image_channels[i])

    joined_channels = []
    for images in separated_channels:
        joined_channels.append(np.hstack(images))

    for i, ch in enumerate(joined_channels):
        channelwise_stats[i]['min'] = ch.min().item()
        channelwise_stats[i]['max'] = ch.max().item()
        channelwise_stats[i]['mean'] = ch.mean().item()
        channelwise_stats[i]['var'] = ch.var().item()
        channelwise_stats[i]['std'] = ch.std().item()
        # summarizing statistics

    for channel in channelwise_stats:
        print('min: {} \t max: {} \t mean: {:.02f} \t std: {:.02f}'.format(channel['min'], channel['max'],
                                                                           channel['mean'], channel['std']))

    if save:
        with open(os.path.join('spectrum_stats.json'), 'w') as fout:
            json.dump(channelwise_stats, fout)

    return channelwise_stats


if __name__ == '__main__':

    check_dataset_stats('/home/mate/dataset/EuroSATallBands_train', save=True)
    # check_dataset_stats('/datasets/EuroSATallBands_test', save=True)
