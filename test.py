# from __future__ import print_function
# from __future__ import division

import matplotlib.pyplot as plt

import numpy as np

import skimage.io
import skimage.transform

import cv2

import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cpm_workable as cpm
import data_import as di
import data_processing as dp

import config

person_net_path = 'initial_models/person_net.ckpt'
pose_net_path = 'initial_models/pose_net.ckpt'


def main():

    b_rgb, b_d, b_d_m, b_joints, b_fm = di.generate_random_batch()
    b = di.KinectFrame(b_rgb[0], b_d[0], b_d_m[0], b_joints[0], b_fm[0])
    # b = di.loadKinectFrame('S001C003P003R002A010', 47)

    tf.reset_default_graph()

    with tf.variable_scope('CPM'):
        # input dims for the person network
        PH, PW = 376, 656
        image_in = tf.placeholder(tf.float32, [1, PH, PW, 3])

        heatmap_person = cpm.inference_person(image_in)
        heatmap_person_large = tf.image.resize_images(heatmap_person, [PH, PW])

        # input dims for the pose network
        N, H, W = config.max_person_num, 376, 376
        pose_image_in = tf.placeholder(tf.float32, [N, H, W, 3])
        pose_centermap_in = tf.placeholder(tf.float32, [N, H, W, 1])
        heatmap_pose = cpm.inference_pose(pose_image_in, pose_centermap_in)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    image = b.rgb
    image = skimage.transform.resize(image, [PH, PW], mode='constant',
                                     preserve_range=True).astype(np.uint8)
    restorer = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'CPM/PersonNet'))
    with tf.Session(config=tf_config) as sess:
        restorer.restore(sess, person_net_path)
        b_image = image[np.newaxis] / 255.0 - 0.5
        hmap_person = sess.run(heatmap_person_large, {image_in: b_image})

    print('done detecting')

    # TODO: make this in tf as well?
    hmap_person = np.squeeze(hmap_person)
    centers = dp.detect_objects_heatmap(hmap_person)
    b_pose_image, b_pose_cmap = dp.prepare_input_posenet(b_image[0], centers, [PH, PW], [H, W])

    restorer = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'CPM/PoseNet'))

    with tf.Session(config=tf_config) as sess:
        restorer.restore(sess, pose_net_path)
        feed_dict = {
            pose_image_in: b_pose_image,
            pose_centermap_in: b_pose_cmap
        }
        _hmap_pose = sess.run(heatmap_pose, feed_dict)

    plt.figure()
    person1 = np.transpose(_hmap_pose[0],(2,0,1))
    for i in range(config.n_joints):
        plt.subplot(4, 4, i+1)
        plt.imshow(person1[i])

    parts = (dp.detect_parts_heatmaps(_hmap_pose, centers, [H, W])) # *[config.rgb_img_size[0],config.rgb_img_size[1]]/[PH, PW]).astype(np.int)
    plt.figure()
    plt.imshow(dp.draw_nodes(parts[0], b.rgb, 'rgb'))
    plt.figure()
    plt.imshow(dp.draw_limbs(b.rgb, np.array([p.as_matrix() for p in parts])))
    plt.show()
if __name__ == "__main__":
    main()
