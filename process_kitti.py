'''
Code for downloading and processing KITTI data (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import requests
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
#from imageio import imread
from cv2 import imread, resize
import hickle as hkl
from kitti_settings import *


desired_im_sz = (128, 160)
categories = ['city', 'residential', 'road']

# Recordings used for validation and testing.
# Were initially chosen randomly such that one of the city recordings was used for validation and one of each category was used for testing.
val_recordings = [('hermann', 'id0')]
test_recordings = [('hermann', 'id1')]
categories = ['hermann']

if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

def my_process_data():
    '''
    this is same as process_data but I chanaged the fold to test if I can use my own images
    '''

    splits = {s: [] for s in ['train', 'test', 'val']} # splits = {'train': [], 'test': [], 'val': []}
    splits['val'] = val_recordings
    splits['test'] = test_recordings
    not_train = splits['val'] + splits['test']
    for c in categories:  # Randomly assign recordings to training and testing. Cross-validation done across entire recordings.
        c_dir = os.path.join(DATA_DIR, 'raw', c + '/')
        folders= list(os.walk(c_dir, topdown=False))[-1][-2]
        splits['train'] += [(c, f) for f in folders if (c, f) not in not_train]

    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from
        for category, folder in splits[split]:
            im_dir = os.path.join(DATA_DIR, 'raw/', category, folder + '/') # DATA_DIR/raw/hermann/id2/
            files = list(os.walk(im_dir, topdown=False))[-1][-1]
            im_list += [im_dir + f for f in sorted(files)]
            source_list += [category + '-' + folder] * len(files) # for example source[i] = ['hermann-id2', 'hermann-id2', ...] the length is the number of images in the corresponding zip file

        print( 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
        X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
        for i, im_file in enumerate(im_list):
            im = imread(im_file) # this will ignore the transparency channel of the image
            X[i] = resize(im, (desired_im_sz[1], desired_im_sz[0]))

        hkl.dump(X, os.path.join(DATA_DIR, 'my_X_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, 'my_sources_' + split + '.hkl'))

if __name__ == '__main__':
    my_process_data()
