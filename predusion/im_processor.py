# process images save in the format of
#- catergory
#  - sub_dir_1
#    - im_1
#    - im_2
#    - im_3
#  - sub_dir_2
#  - label.json
#note images under sub_dir are listed in the order of time. label.json is a dictionary {sub_dir_1: label_1, sub_dir_2: label_2}

import os
import requests
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
from imageio import imread
from scipy.misc import imresize
import hickle as hkl
from kitti_settings import *
import cv2

if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

def process_data(categories=[], desired_im_sz=(128, 160), val_recordings=[], test_recordings=[], out_data_head='', scale=None):
    '''
    input:
      out_data_head (str):
      example recording looks like this
        test_recordings = [(c, 'sub_dir_' + str(idi)) for c in categories for idi in range(9)] where 'sub_dir_' + str(idi) is the sub_dir_1 for example

    output:
      output files are 1. out_data_head + '_X_train.hkl' 2. out_data_head + '_source_train.hkl' 3. out_data_head + '_X_test.hkl' 4. out_data_head + '_source_test.hkl' 5. out_data_head + '_label.hkl'
    '''
    splits = {s: [] for s in ['train', 'test', 'val']} # splits = {'train': [], 'test': [], 'val': []}
    splits['val'] = val_recordings
    splits['test'] = test_recordings
    not_train = splits['val'] + splits['test']

    label_list = {} # a dictionary {source_folder: label} This label contains all of testing, training, validation set
    label_name_list = []
    for c in categories:  # Randomly assign recordings to training and testing. Cross-validation done across entire recordings.
        c_dir = os.path.join(DATA_DIR, 'raw', c + '/')
        print(c_dir)
        folders= list(os.walk(c_dir, topdown=False))[-1][-2]
        splits['train'] += [(c, f) for f in folders if (c, f) not in not_train]

        label_dir = os.path.join(DATA_DIR, 'raw/', c, 'label.json')
        label = hkl.load(label_dir)
        label_list = {**label_list, **label}

        label_name_dir = os.path.join(DATA_DIR, 'raw/', c, 'label_name.json')
        label_name = hkl.load(label_name_dir)
        label_name_list = [*label_name_list, *label_name]

    hkl.dump(label_list, os.path.join(DATA_DIR, out_data_head + '_label' + '.hkl'))
    hkl.dump(label_name_list, os.path.join(DATA_DIR, out_data_head + '_label_name' + '.hkl'))

    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from

        for category, folder in splits[split]:
            im_dir = os.path.join(DATA_DIR, 'raw/', category, folder + '/') # DATA_DIR/raw/hermann/id2/
            print(im_dir)
            files = list(os.walk(im_dir, topdown=False))[-1][-1]
            im_list += [im_dir + f for f in sorted(files)]
            source_list += [category + '-' + folder] * len(files) # for example source[i] = ['hermann-id2', 'hermann-id2', ...] the length is the number of images in the corresponding zip file

        print( 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
        X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
        for i, im_file in enumerate(im_list):
            im = imread(im_file, pilmode='RGB') # this will ignore the transparency channel of the image
            X[i] = process_im(im, desired_im_sz, scale=scale)

        hkl.dump(X, os.path.join(DATA_DIR, out_data_head + '_X_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, out_data_head + '_sources_' + split + '.hkl'))

# resize and crop image
def process_im(im, desired_sz, scale=None):
    '''
    First step: 
    '''
    im_temp = im.copy()
    if im_temp.shape[0] / im_temp.shape[1] > desired_sz[0] / desired_sz[1]:
        target_ds = float(desired_sz[1])/im_temp.shape[1]
        im_temp = imresize(im_temp, (int(np.round(target_ds * im_temp.shape[0])), desired_sz[1]))
        d = int((im_temp.shape[0] - desired_sz[0]) / 2)
        im_temp = im_temp[d:d+desired_sz[0], :]
    else:
        target_ds = float(desired_sz[0])/im_temp.shape[0]
        im_temp = imresize(im_temp, (desired_sz[0], int(np.round(target_ds * im_temp.shape[1]))))
        d = int((im_temp.shape[1] - desired_sz[1]) / 2)
        im_temp = im_temp[:, d:d+desired_sz[1]]

    im_norm = cv2.normalize(im_temp, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    im_norm = im_norm.astype(np.uint8)

    if not(scale is None):
        im_norm = rescale_im(im_norm, scale=scale)

    return im_norm


def rescale_im(im, scale=1):
    # create a black background
    im_bg = np.zeros(im.shape, dtype=np.uint8)
    scale_shape = (np.array(im.shape[:2] ) * scale).astype(np.uint8)
    # resize the image
    im_scale = imresize(im, size=tuple(scale_shape), interp='nearest')
    # add the image to a gray background
    center = np.array(im.shape[:2], dtype=np.uint8)//2
    center_coner = center - np.array(scale_shape, dtype=np.uint8) // 2
    im_bg[center_coner[0]: center_coner[0] + scale_shape[0], center_coner[1]: center_coner[1] + scale_shape[1], :] = im_scale
    return im_bg

if __name__ == '__main__':
    process_data()
