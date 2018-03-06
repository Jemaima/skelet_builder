import numpy as np

MODE = ''
# =================== data parameters ==================

max_person_num = 16
data_set_path = "Z://NTU_RGB_D_dataset//"
depth_data_folder = data_set_path + "nturgb+d_depth//"
depth_masked_data_folder = data_set_path + "nturgb+d_depth_masked//"
rgb_data_folder = data_set_path + "nturgb+d_rgb//"
ir_data_folder = data_set_path + "nturgb+d_ir//"
skeleton_data_folder = data_set_path + "nturgb+d_skeletons//"
exception_file = data_set_path + "exceptions.xlsx"

# rgb image config
is_color = True
rgb_img_size = (1080, 1920, 3)

ph_img_height = 360
ph_img_width = 640

rgb_img_scale_factor = rgb_img_size[0]//ph_img_height

# depth image config
depth_img_size = (424, 512)
ph_depth_height = 360
ph_depth_width = 640
depth_img_padding = (0,0,121,121)


depth_img_scale_factor = 1

ph_fms_height = 46
ph_fms_width = 46


# =================== joints parameters ==================
if MODE =='kinect':
    valuable_joints = np.arange(20) + 1
else:
    valuable_joints = [3, 4,  5, 6, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19]


n_joints = len(valuable_joints)
fm_channel = n_joints + 1

limbs = np.array([1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14]).reshape((-1,2))
limbs_color = [[0, 0, 255],
               [0, 255, 0],
               [255, 0, 0],
               [0, 170, 255],
               [0, 255, 170],
               [170, 255, 0],
               [255, 170, 0],
               [255, 0, 170],
               [170, 0, 255],
               [255, 255, 0],
               [255, 0, 255],
               [0, 255, 255],
               [170, 170, 0],
               [170, 0, 170],
               [0, 170, 170],
               ]
# training
batch_size = 3
full_training = False
l_r = 0.001
# =================== modify parameters ==================

initialize = True  # True: train from scratch (should also

steps = "30000"  # if 'initialize = False', set steps to
# where you want to restore
toDistort = False
# iterations config
max_iteration = 30000
checkpoint_iters = 1000
summary_iters = 50
validate_iters = 1000
# checkpoint path and filename
logdir = ".//log//"
params_dir = ".//models//"
load_filename = "cpm" + '-' + steps
save_filename = "cpm"

# ========================================================

# feature map config
fm_width = depth_img_size[1]  # img_width >> 1
fm_height = depth_img_size[0]  # img_height >> 1
sigma = 5.0
alpha = 1.0
radius = 20
person_detection_sigma = 25

# random distortion
degree = 8

# solver config
wd = 5e-4
stddev = 5e-2
use_fp16 = False
moving_average_decay = 0.999

