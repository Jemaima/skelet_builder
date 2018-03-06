import config
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import skimage.io
import skimage.transform


def draw_nodes(nodes, img, type):
    img_with_nodes = np.copy(img)
    if type == 'rgb':
        scaler = config.rgb_img_size[0] / img_with_nodes.shape[0]
        for _, node in nodes.iterrows():
            cv2.circle(img_with_nodes, (int(node['rgb_x'] // scaler),
                                       int(node['rgb_y'] // scaler)), 3, [0, 255, 170], -1)

    elif type == 'depth':
        scaler = config.depth_img_size[0]/img_with_nodes.shape[0]
        for _, node in nodes.iterrows():
            for index, node in nodes.iterrows():
                cv2.circle(img_with_nodes, (int((node['depth_x']+config.depth_img_padding[2]) // scaler),
                                            int((node['depth_y'] +config.depth_img_padding[0])// scaler)), 3, 100, -1)
    return img_with_nodes


def draw_limbs(img, joints):
    img_with_nodes = np.copy(img)
    scaler = config.rgb_img_size[0] / img_with_nodes.shape[0]
    for person in range(joints.shape[0]):
        for lid, (p0, p1) in enumerate(config.limbs): #initial_limbs):
            y0, x0 = (joints[person][p0-1]//scaler).astype(np.int)
            y1, x1 = (joints[person][p1-1]//scaler).astype(np.int)
            cv2.line(img_with_nodes, (x0, y0), (x1, y1), config.limbs_color[lid], 2)
    return img_with_nodes


def detect_objects_heatmap(heatmap):
    data = 256 * heatmap
    data_max = filters.maximum_filter(data, 3, mode='reflect')
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, 3, mode='reflect')
    diff = ((data_max - data_min) > 5)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    objects = np.zeros((num_objects, 2), dtype=np.int32)
    for oid, (dy, dx) in enumerate(slices):
        objects[oid, :] = [(dy.start + dy.stop - 1) / 2, (dx.start + dx.stop - 1) / 2]
    return objects


def gaussian_kernel(h, w, sigma_h, sigma_w):
    yx = np.mgrid[-h // 2:h // 2, -w // 2:w // 2] ** 2
    return np.exp(-yx[0, :, :] / sigma_h ** 2 - yx[1, :, :] / sigma_w ** 2)


def prepare_input_posenet(img, objects, size_person, size, sigma=config.person_detection_sigma, max_num_objects=config.max_person_num, border=400):
    result = np.zeros((max_num_objects, size[0], size[1], 4))
    padded_image = np.zeros((1, size_person[0] + border, size_person[1] + border, 4))
    padded_image[0, border // 2:-border // 2, border // 2:-border // 2, :3] = img
    assert len(objects) < max_num_objects
    for oid, (yc, xc) in enumerate(objects):
        dh, dw = size[0] // 2, size[1] // 2
        y0, x0, y1, x1 = np.array([yc - dh, xc - dw, yc + dh, xc + dw]) + border // 2
        result[oid, :, :, :4] = padded_image[:, y0:y1, x0:x1, :]
        result[oid, :, :, 3] = gaussian_kernel(size[0], size[1], sigma, sigma)
    return np.split(result, [3], 3)


def detect_parts_heatmaps(heatmaps, centers, size, num_parts=config.n_joints):
    parts = np.zeros((len(centers), num_parts, 2), dtype=np.int32)
    for oid, (yc, xc) in enumerate(centers):
        part_hmap = skimage.transform.resize(np.clip(heatmaps[oid], -1, 1), size,
                                             mode='reflect')
        for pid in range(num_parts):
            y, x = np.unravel_index(np.argmax(part_hmap[:, :, pid]), size)
            parts[oid, pid] = y + yc - size[0] // 2, x + xc - size[1] // 2
    return [pd.DataFrame(p*2.872340425531915, index=np.arange(config.n_joints), columns=['rgb_y', 'rgb_x']) for p in parts]