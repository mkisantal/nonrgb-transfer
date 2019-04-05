import numpy as np
from PIL import Image
from hsi_dataset import load_single_image, HsiSegmentationDataset
import torch
from albumentations import Compose, CenterCrop


def to_color_image(torch_tensor, pil=True):

    """ Converting to valid RGB image. If HSI, then RGB channels are selected. """

    tensor = np.squeeze(torch_tensor.clone().cpu().numpy())
    if tensor.shape[0] != 3:
        tensor = tensor[[3, 2, 1], :, :]
    np_img = tensor.transpose()
    np_img -= np_img.min()
    np_img *= 255/np_img.max()
    norm_image = np.uint8(np_img)
    if pil:
        return Image.fromarray(norm_image)
    else:
        return norm_image


def inspect_images(model, device, num_samples=5, channels=None, gen_only=False):

    """ Visualizing learned HSI Image embedding. """

    # TODO: adapt this for new dataset
    # TODO: calculate whole batch at once?
    model.eval()
    rows = []
    for image_id in range(num_samples):
        rgb_images = []
        converted_images = []
        for category_id in range(10):
            x = load_single_image(image_id, category_id, channels=channels)
            original_rgb = load_single_image(image_id, category_id, normalize=False)
            rgb_images.append(to_color_image(original_rgb, pil=False))
            x = x.unsqueeze(0).to(device)
            with torch.set_grad_enabled(False):
                if gen_only:
                    transformed_image=model(x)
                else:
                    transformed_image = model.module.get_transformed_input(x)
            converted_images.append(to_color_image(transformed_image, pil=False))
        appended_rgb = np.hstack(rgb_images)
        appended_conv = np.hstack(converted_images)
        row = np.vstack([appended_rgb, appended_conv])
        rows.append(row)
    single_image = np.vstack(rows)
    return Image.fromarray(single_image)


def inspect_images2(model, device, num_samples=2, channels=None, gen_only=False):

    """ Visualizing learned segmentation MSI image embedding. """

    model.eval()
    rows = []

    transforms = Compose([CenterCrop(224, 224)])
    dataset = HsiSegmentationDataset(root='/home/mate/datasets/grss/Track1-MSI_A_val',
                                     gt_root='/home/mate/datasets/grss/Train-Track1-Truth',
                                     channels=None, transforms=transforms)

    for index in range(num_samples):
        rgb_images = []
        converted_images = []
        for category_id in range(num_samples * 2):
            with torch.set_grad_enabled(False):
                img_npy, gt_tensor = dataset[index + 10 * category_id]
                image_in = torch.tensor(img_npy).unsqueeze(0).to(device)
                out, transformed = model(image_in)

                rgb_images.append(to_color_image(image_in, pil=False))
                converted_images.append(to_color_image(transformed, pil=False))
        appended_rgb = np.hstack(rgb_images)
        appended_conv = np.hstack(converted_images)
        row = np.vstack([appended_rgb, appended_conv])
        rows.append(row)
    single_image = np.vstack(rows)
    return Image.fromarray(single_image)
