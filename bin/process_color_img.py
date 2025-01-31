''' image is saved with RGB ranges 0 to 255'''
import os
import requests
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
from imageio import imread
from scipy.misc import imresize
import hickle as hkl
from kitti_settings import *


desired_im_sz = (128, 160)

# Recordings used for validation and testing.
# Were initially chosen randomly such that one of the city recordings was used for validation and one of each category was used for testing.
val_recordings = []
categories = ['square_10_64'] # please use nt = 9

if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

def my_process_data():
    '''
    this is same as process_data but I chanaged the fold to test if I can use my own images
    '''
    splits = {s: [] for s in ['train', 'test', 'val']} # splits = {'train': [], 'test': [], 'val': []}. In this code, actually only 'test' is a non-empty list
    for c in categories:  # Randomly assign recordings to training and testing. Cross-validation done across entire recordings.
        c_dir = os.path.join(DATA_DIR, 'raw', c + '/')
        folders= list(os.walk(c_dir, topdown=False))[-1][-2]
        splits['test'] += [(c, f) for f in folders]

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

        hkl.dump(X, os.path.join(DATA_DIR, 'color_img_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, 'color_img_sources_' + split + '.hkl'))

# resize and crop image
def process_im(im, desired_sz):
    '''
    crop the images to desired_sz
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
    my_process_data()
