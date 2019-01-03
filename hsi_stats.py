import rasterio
import os
from multiprocessing import Pool
import tqdm
import json


def check_image_stats(path):

    """ Check channelwise statistics for multispectral image. """

    image_stats = []
    img = rasterio.open(path).read()
    for ch in range(img.shape[0]):
        image_stats.append({'min': img[ch, :, :].min().item(),
                            'max': img[ch, :, :].max().item(),
                            'mean': img[ch, :, :].mean().mean(),
                            'var': img[ch, :, :].var().item()})
    return image_stats


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
        channelwise_stats.append({'min': 10e9, 'max': -1, 'mean': 0, 'var': 0})

    all_image_stats = []
    with Pool(8) as p:
        for stats in tqdm.tqdm(p.imap(check_image_stats, image_paths), total=len(image_paths)):
            all_image_stats.append(stats)

    # summarizing statistics
    for j, stats in enumerate(all_image_stats):
        for i in range(num_channels):
            channelwise_stats[i]['min'] = min(stats[i]['min'], channelwise_stats[i]['min'])
            channelwise_stats[i]['max'] = max(stats[i]['max'], channelwise_stats[i]['max'])

            channelwise_stats[i]['mean'] += (stats[i]['mean'] - channelwise_stats[i]['mean']) / (j+1)
            channelwise_stats[i]['var'] += (stats[i]['var'] - channelwise_stats[i]['var']) / (j+1)

    for channel in channelwise_stats:
        print('min: {} \t max: {} \t mean: {:.02f} \t var: {:.02f}'.format(channel['min'], channel['max'],
                                                                           channel['mean'], channel['var']))

    if save:
        with open(os.path.join('spectrum_stats.json'), 'w') as fout:
            json.dump(channelwise_stats, fout)

    return channelwise_stats


if __name__ == '__main__':

    # check_dataset_stats(r"C:\datasets\EuroSATallBands\ds\images\remote_sensing\otherDatasets\sentinel_2\tif",
    #                     save=True)
    check_dataset_stats('/home/mate/dataset/EuroSATallBands/ds/images/remote_sensing/otherDatasets/sentinel_2/tif',
                        save=True)

