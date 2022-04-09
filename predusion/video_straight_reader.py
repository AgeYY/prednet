# Show the video stimuli used in paper Hénaff et al. (2021) from https://osf.io/gwtcs/
import scipy.io
import os
import numpy as np
from predusion.immaker import process_im

class VS_reader():
    '''
    read video data from Hénaff et al. (2021)
    '''
    def __init__(self, path='./data/stim_matrix.mat'):
        '''
        more information about stim_matrix please data/readme.md
        '''
        self.stim_matrix = scipy.io.loadmat('./data/stim_matrix.mat')

    def read_frame(self, video_type='natural', video_cate='01', video_frame='01', scale='1x'):
        '''
        read only one frame
          video_type: natural, contrast, synthetic
          video_cate: from 01 to 10. They corresponds to [chironomus, bees, dogville, egomotion, prairiel, carnegie, walking, smile, water, leaves-wind]
          video_frame: the ith video from from 01 to 11
          scale: 1x or 2x
        output:
          out_im ([512, 512]): target frame, if multiple frame satisfy the requirement, only the first one would be returned. Correct input will have one frame anyway
        '''
        video_type_frame = video_type + video_frame
        video_cate = 'movie' + video_cate
        scale = 'zoom' + scale

        for i_scale, im_scale in enumerate(self.stim_matrix['image_paths']):

            for i_type, im_type in enumerate(im_scale):

                for i_cate, im_cate in enumerate(im_type):

                    for i_frame, im_frame in enumerate(im_cate):

                        if (video_type_frame in im_frame[0]) and (video_cate in im_frame[0]) and (scale in im_frame[0]):
                            return np.array(self.stim_matrix['stim_matrix'][i_scale, i_type, i_cate, :, :, i_frame])

    def read_video(self, video_type='natural', video_cate='01', scale='1x'):
        '''
        inefficient algorithm, but good enough
        output:
          video ([n_frames, 512, 512])
        '''
        frame_list = ['0' + str(i) for i in range(1, 10)]
        frame_list.append('10')
        frame_list.append('11')

        video = []
        for video_frame in frame_list:
            im = self.read_frame(video_type, video_cate, video_frame, scale)
            if not (im is None):
                video.append(im)

        return np.array(video)

    def read_video_ppd(self, video_type='natural', video_cate='01', scale='1x', imshape=(128, 160)):
        '''
        read and process the video (resize the image to the target imshape)
        output:
        video_ppd ([n_frames, 512, 512])
        '''
        video = self.read_video(video_type=video_type, video_cate=video_cate, scale=scale)
        video = video.reshape((1, *video.shape))

        assert len(video.shape) == 4, 'The shape of video should be (n_video, n_frame, imsize[0], imsize[1])'
        video_ppd = np.zeros((video.shape[0], video.shape[1], *imshape))
        for i_seq, seq in enumerate(video):
            for i_im, im in enumerate(seq):
                video_ppd[i_seq, i_im] = process_im(im, imshape)

        return video_ppd[0]

    def read_video_all(self, video_type='natural', scale='1x'):
        '''
        real all category of videos
        output:
          video (array, [n_video, n_frames, 512, 512] )
        '''
        cate_list = ['0' + str(i) for i in range(1, 10)]
        cate_list.append('10')

        video = []
        for video_cate in cate_list:
            a_video = self.read_video(video_type, video_cate, scale)
            if not (a_video is None):
                video.append(a_video.copy())

        return np.array(video)

    def read_video_all_ppd(self, video_type='natural', scale='1x', imshape=(128, 160)):
        '''
        read and process the video (resize the image to the target imshape)
        output:
          video (array, [n_video, n_frames, *imshape] )
        '''
        video = self.read_video_all(video_type=video_type, scale=scale)

        video_ppd = np.zeros((video.shape[0], video.shape[1], *imshape))
        for i_seq, seq in enumerate(video):
            for i_im, im in enumerate(seq):
                video_ppd[i_seq, i_im] = process_im(im, imshape)

        return video_ppd

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dreader = VS_reader()
    im = dreader.read_frame(video_type='natural', video_cate='03', video_frame='05', scale='2x')
    plt.figure()
    plt.imshow(im)
    plt.show()

    video = dreader.read_video(video_type='synthetic', video_cate='03', scale='2x')
    print(video.shape)
    print(type(video))
    for im in video:
        plt.imshow(im)
        plt.show()
