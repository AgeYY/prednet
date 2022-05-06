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

if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

def process_data(categories=[], desired_im_sz=(128, 160), val_recordings=[], test_recordings=[], out_data_head=''):
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
    for c in categories:  # Randomly assign recordings to training and testing. Cross-validation done across entire recordings.
        c_dir = os.path.join(DATA_DIR, 'raw', c + '/')
        print(c_dir)
        folders= list(os.walk(c_dir, topdown=False))[-1][-2]
        splits['train'] += [(c, f) for f in folders if (c, f) not in not_train]

        label_dir = os.path.join(DATA_DIR, 'raw/', c, 'label.json')
        label = hkl.load(label_dir)
        label_list = {**label_list, **label}

    hkl.dump(label_list, os.path.join(DATA_DIR, out_data_head + '_label' + '.hkl'))

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
            X[i] = process_im(im, desired_im_sz)

        hkl.dump(X, os.path.join(DATA_DIR, out_data_head + '_X_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, out_data_head + '_sources_' + split + '.hkl'))

# resize and crop image
def process_im(im, desired_sz):
    '''
    First step: 
    '''
    im_temp = im.copy()
    if im.shape[0] / im.shape[1] > desired_sz[0] / desired_sz[1]:
        target_ds = float(desired_sz[1])/im.shape[1]
        im = imresize(im, (int(np.round(target_ds * im.shape[0])), desired_sz[1]))
        d = int((im.shape[0] - desired_sz[0]) / 2)
        im = im[d:d+desired_sz[0], :]
    else:
        target_ds = float(desired_sz[0])/im.shape[0]
        im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
        d = int((im.shape[1] - desired_sz[1]) / 2)
        im = im[:, d:d+desired_sz[1]]

    return im

if __name__ == '__main__':
    process_data()
