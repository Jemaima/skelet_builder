import config
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import data_processing as dp

node_params = ['x',
               'y',
               'z',
               'depth_x',
               'depth_y',
               'rgb_x',
               'rgb_y',
               'orientation_w',
               'orientation_x',
               'orientation_y',
               'orientation_z',
               'state',
               ]

data_list = os.listdir(config.depth_data_folder)


class KinectFrame():
    def __init__(self, rgb=None, depth=None, depth_masked=None, joints=None, fms=None):
        self.joints = joints
        self.rgb = rgb
        self.depth = depth
        self.depth_masked = depth_masked
        if (fms is None and joints is None):
            self.fms = _generate_feature_map(self.joints)
        else:
            self.fms = fms


class KinectEpisode():

    def __init__(self, filename):
        print(filename + 'processing...')
        self.joints, self.n_frames = _import_skeleton_points(filename)
        print("Joints loaded")
        self.rgb = _import_rgb(filename, _n_frames=self.n_frames)
        print("RGB loaded")
        self.depth = _import_depth(filename)
        print("Depth loaded")
        self.depth_masked = _import_depth_masked(filename)
        print("Depth masked loaded")
        # self.fms = _generate_feature_map(self.joints)


def _import_rgb(_filename, _frame_num=-1, _n_frames=0):
    cap = cv2.VideoCapture(config.rgb_data_folder + _filename + "_rgb.avi")
    n = 0;
    frame_row = []
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    # Read until video is completed
    while (n < _n_frames):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if _frame_num != -1:
            if n == _frame_num:
                cap.release()
                return cv2.cvtColor(cv2.resize(frame, (0, 0), fx=1 / config.rgb_img_scale_factor, fy=1 / config.rgb_img_scale_factor),cv2.COLOR_RGB2GRAY) if not config.is_color else cv2.resize(frame, (0, 0), fx=1 / config.rgb_img_scale_factor, fy=1 / config.rgb_img_scale_factor)

        else:
            if config.is_color:
                frame_row.append(
                cv2.resize(frame, (0, 0), fx=1 / config.rgb_img_scale_factor, fy=1 / config.rgb_img_scale_factor))
            else:
                frame_row.append(
                    cv2.cvtColor(cv2.resize(frame, (0, 0), fx=1 / config.rgb_img_scale_factor, fy=1 / config.rgb_img_scale_factor),cv2.COLOR_RGB2GRAY))

        n += 1
    cap.release()
    return np.array(frame_row)


def _import_depth(_filename, _frame_num=-1):
    if _frame_num != -1:
        frame = cv2.imread(
            config.depth_data_folder + _filename + '//' + os.listdir(config.depth_data_folder + _filename)[_frame_num],
            0)
        frame = cv2.copyMakeBorder(frame, config.depth_img_padding[0], config.depth_img_padding[1], config.depth_img_padding[2], config.depth_img_padding[3], cv2.BORDER_REPLICATE)
        return cv2.resize(frame,(config.ph_depth_width ,config.ph_depth_height))
    else:
        frame_row = []
        for file in os.listdir(config.depth_data_folder + _filename):
            frame_row.append(cv2.imread(
                config.depth_data_folder + _filename + '//' + file,
                0))
        return np.array(frame_row)


def _import_depth_masked(_filename, _frame_num=-1):
    if _frame_num != -1:
        frame = cv2.imread(
            config.depth_masked_data_folder + _filename + '//' +
            os.listdir(config.depth_masked_data_folder + _filename)[
                _frame_num], 0)
        return frame
    else:
        frame_row = []
        for file in os.listdir(config.depth_masked_data_folder + _filename):
            frame_row.append(cv2.imread(
                config.depth_masked_data_folder + _filename + '//' + file,
                0))
        return np.array(frame_row)


def _import_ir(_filename, _frame_num=-1):
    if _frame_num != -1:
        frame = cv2.imread(
            config.ir_data_folder + _filename + '//' + os.listdir(config.ir_data_folder + _filename)[_frame_num],
            0)
        return frame
    else:
        frame_row = []
        for file in os.listdir(config.ir_data_folder + _filename):
            frame_row.append(cv2.imread(
                config.ir_data_folder + _filename + '//' + file,
                0))
        return np.array(frame_row)


def _import_skeleton_points(_filename, _frame_num=-1):
    print(_filename + ':  frame: ' + str(_frame_num))
    f = open(config.skeleton_data_folder + _filename + '.skeleton', 'r')
    frame_row = []
    frame_cnt = int(f.readline())
    for frame in range(frame_cnt):
        tmp = []
        try:
            int(f.readline())
            f.readline()
            node_cnt = int(f.readline())
        except:
            node_cnt = int(f.readline())
        for i in range(node_cnt):
            if i + 1 in config.valuable_joints:
                tmp.append(pd.Series(f.readline()[:-1].split(' '), index=node_params).astype(dtype=np.float32))
            else:
                f.readline()

        if _frame_num != -1 and frame == _frame_num:
            f.close()
            return pd.DataFrame(tmp, index=np.arange(config.n_joints), columns=node_params), frame_cnt
        else:
            frame_row.append(pd.DataFrame(tmp, index=np.arange(config.n_joints), columns=node_params))
    f.close()
    return frame_row, frame_cnt


def _generate_feature_map(nodes):
    fm = []
    for index, node in nodes.iterrows():
        tmp = np.zeros([config.fm_height, config.fm_width])
        tmp[int(node['depth_y']) - config.radius:int(node['depth_y']) + config.radius,
        int(node['depth_x']) - config.radius:int(node['depth_x']) + config.radius] = [
            [np.exp(-4 * np.log(2) * ((x - config.radius) ** 2 + (y - config.radius) ** 2) / config.radius ** 2)
             for x in range(config.radius << 1)] for y in range(config.radius << 1)]
        fm.append(skimage.transform.resize(tmp,(config.ph_fms_height,config.ph_fms_width)))
    fm.append(np.sum(fm, axis=0))
    return np.array(fm)

def loadKinectFrame(filename, frame_num=-1):
    exc = pd.read_excel(config.exception_file, squeeze=True)
    if filename in exc.values:
        print(filename + ' has no skelet data.')
        return None
    print(filename + ' processing...')
    joints, n_frames = _import_skeleton_points(filename, frame_num)
    print("Joints loaded")
    rgb = _import_rgb(filename, frame_num, _n_frames=n_frames)
    print("RGB loaded")
    depth = _import_depth(filename, frame_num)
    print("Depth loaded")
    depth_masked = _import_depth_masked(filename, frame_num)
    print("Depth masked loaded/n")
    fms = _generate_feature_map(joints)
    return KinectFrame(rgb,depth,depth_masked,joints,fms)


def generate_random_batch(batch_size=config.batch_size):
    rgb_imgs = []
    depth_imgs = []
    depth_m_imgs = []
    fms = []
    joints = []

    for sample in range(batch_size):
        filename = np.random.choice(data_list)
        f = open(config.skeleton_data_folder + filename + '.skeleton', 'r')
        frame_count = int(f.readline())
        frame_num = np.random.choice(frame_count)
        f.close()
        data = loadKinectFrame(filename,frame_num)
        joints.append(data.joints)
        rgb_imgs.append(data.rgb)
        depth_imgs.append(data.depth)
        depth_m_imgs.append(data.depth_masked)
        fms.append(data.fms)
    return rgb_imgs, depth_imgs, depth_m_imgs, joints, fms


def main():

    b_rgb, b_d, b_d_m, b_joints, b_fm = generate_random_batch(2)
    b = KinectFrame(b_rgb[0], b_d[0], b_d_m[0], b_joints[0], b_fm[0])

    # b = loadKinectFrame('S001C003P003R002A010', 0)
    #
    cv2.imwrite('pic1.jpg', b.rgb)

    # check nodes
    im = dp.draw_nodes(b.joints, b.rgb, 'rgb')
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(im) if config.is_color else plt.imshow(im, 'gray')

    im = dp.draw_nodes(b.joints, b.depth, 'depth')
    plt.subplot(2, 1, 2)
    plt.imshow(im)

    im2 = dp.draw_limbs(b.rgb, np.array([b.joints[['rgb_y', 'rgb_x']].as_matrix()]))
    plt.figure()
    plt.imshow(im2)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.histogram(b.depth, 255)[0])
    plt.subplot(2, 1, 2)
    plt.imshow(b.depth, 'gray')
    #
    # plt.figure()
    # plt.imshow(np.sum(b.fms, axis=0))
    # plt.show()


if __name__ == "__main__":
    main()
