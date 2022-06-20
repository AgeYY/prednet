# create a dataset. Each observation is one image labeled by a few latent vairables
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
#import hickle as hkl
import torch
from torch.utils.data import Dataset, DataLoader

#class Moving_square(): # generate a moving square along the x direction
#    def __init__(self, width = 200, color_bag=(0, 0, 0)):
#        self.images = []
#        self.color_bag = color_bag # background of the images
#        self.width = width # image width
#        self.init_im()
#
#    def init_im(self, im_bag=None):
#        self.im_bag = im_bag
#
#    def create_video(self, init_pos = [0, 0], speed=2, step=20, color_rect=(255, 255, 255), color_bag=(0, 0, 0), size_rect=20, shape='rectangle', shape_para={}):
#        self.init_pos = init_pos
#        self.size_rect = size_rect
#        self.speed = int(speed) # unit: pixels per frame
#        self.step = step # number of time steps
#
#        if self.im_bag is None:
#            self.im_bag = Image.new('RGB', (self.width, self.width), color=color_bag)
#
#        for i in range(step):
#            im_temp = self.im_bag.copy()
#            draw = ImageDraw.Draw(im_temp)
#            curr_pos = (self.init_pos[0] + self.speed * i, self.init_pos[1]) # moving along the x direction
#            if np.array(self.size_rect).shape == (): # if size_rect is a scalar
#                rect_para = (curr_pos[0] - self.size_rect // 2, curr_pos[1] - self.size_rect // 2, curr_pos[0] + self.size_rect // 2, curr_pos[1] + self.size_rect // 2) # initial position
#            else: # size_rect is (width, length)
#                rect_para = (curr_pos[0] - self.size_rect[0] // 2, curr_pos[1] - self.size_rect[1] // 2, curr_pos[0] + self.size_rect[0] // 2, curr_pos[1] + self.size_rect[1] // 2) # initial position
#
#            if shape == 'rectangle':
#                draw.rectangle(rect_para, fill=color_rect)
#            elif shape == 'arc':
#                draw.arc(rect_para, shape_para['start'], shape_para['end'], fill=color_rect)
#            elif shape == 'text':
#                draw.text(curr_pos, shape_para['text'], fill=color_rect)
#
#            self.images.append(im_temp)
#
#    def save_image(self, save_dir='./kitti_data/raw/moving_bars'):
#        if not os.path.exists(save_dir): os.makedirs(save_dir) # if doesn't exist, create the dir
#        [self.images[i].save(save_dir + 'im_' + '{0:03}'.format(i) + '.jpg') for i in range(len(self.images))]
#
#    def clear_image(self):
#        self.images=[]
#
#    def overlap_video(self, images, video):
#        '''
#        images is a sequence of frames where a moving square moving on a blackbackground
#        '''
#        video_size = [imresize(im, (self.width, self.width)) for im in video]
#
#        n_frame = min(len(images), video.shape[0])
#        images_temp = []
#        for i in range(n_frame):
#            images_temp_max = np.maximum(images[i], video_size[i])
#            images_temp.append(Image.fromarray(images_temp_max))
#        self.images = images_temp
#
#    def create_video_batch_on_video(self, video, init_pos = [0, 0], speed=[], step=20, color_rect=(255, 255, 255), color_bag=(0, 0, 0), save_dir_head='./kitti_data/raw/', category='moving_bar', sub_dir_head='sp_', size_rect=20):
#        '''
#        similar as create_video_batch, but instead of static background, the moving square will move on a video. The number of frames would be the minimum of frames in the moving bar video and the input video. UGLY CODE, NEEDS TO BE IMPROVED.
#        '''
#        for sp in speed:
#            self.create_video(init_pos=init_pos, speed=sp, step=step, size_rect=size_rect, color_rect=color_rect, color_bag=(0, 0, 0), shape='rectangle')
#            self.overlap_video(self.images, video) # overlap the moving square to a video
#
#            save_dir_label =save_dir_head + category + '/' + sub_dir_head + str(sp) + '/'
#            self.save_image(save_dir_label)
#            self.clear_image()
#        label= {category + '-' + sub_dir_head + str(sp): sp  for sp in speed} # source_folder : label. source_folder format is the same as process_kitti.py
#        hkl.dump(label, save_dir_head + category + '/label.json')
#
#    def create_video_batch(self, init_pos = [0, 0], speed=[], step=20, color_rect=(255, 255, 255), color_bag=(0, 0, 0), save_dir_head='./kitti_data/raw/', category='moving_bar', sub_dir_head='sp_', size_rect=20, shape='rectangle', shape_para={}):
#        '''
#        save images and labels
#        '''
#        for sp in speed:
#            self.create_video(init_pos=init_pos, speed=sp, step=step, size_rect=size_rect, color_rect=color_rect, color_bag=color_bag, shape=shape, shape_para=shape_para)
#            save_dir_label =save_dir_head + category + '/' + sub_dir_head + str(sp) + '/'
#            self.save_image(save_dir_label)
#            self.clear_image()
#        label= {category + '-' + sub_dir_head + str(sp): sp  for sp in speed} # source_folder : label. source_folder format is the same as process_kitti.py
#        hkl.dump(label, save_dir_head + category + '/label.json')

#class Latent_image(Dataset):
import math
from PIL import Image, ImageDraw

#finds the straight-line distance between two points
def distance(ax, ay, bx, by):
    return math.sqrt((by - ay)**2 + (bx - ax)**2)

#rotates point `A` about point `B` by `angle` radians clockwise.
def rotated_about(ax, ay, bx, by, angle):
    radius = distance(ax,ay,bx,by)
    angle += math.atan2(ay-by, ax-bx)
    return (
        round(bx + radius * math.cos(angle)),
        round(by + radius * math.sin(angle))
    )

# latent variable angle, square_length, center
square_center = (50,50)
square_length = 40
angle = math.radians(45)

image = Image.new('L', (100, 100), 0)
draw = ImageDraw.Draw(image)

square_vertices = (
    (square_center[0] + square_length / 2, square_center[1] + square_length / 2),
    (square_center[0] + square_length / 2, square_center[1] - square_length / 2),
    (square_center[0] - square_length / 2, square_center[1] - square_length / 2),
    (square_center[0] - square_length / 2, square_center[1] + square_length / 2)
)

square_vertices = [rotated_about(x,y, square_center[0], square_center[1], math.radians(45)) for x,y in square_vertices]

draw.polygon(square_vertices, fill=255)

plt.imshow(image)
plt.show()
