Data from paper Hénaff et al. (2021) from https://osf.io/gwtcs/ 10 videos are obtained from a few dataset. Then they are down sampled in the temporal direction so that the change of consecutive frames are large enough, thus the curveture become meaningful. Second, only center of the videos clipped out.

Load data using scipy.io.loadmat(). The output is a dictionary with keys

# stim\_info.mat
meta information about the stimuli.

- artificial\_movie\_contrast [2 (scaled x1 and scaled x2?), 10 (n\_original videos), 11 (n\_frames per video)]: 10 is the number of original movie, 11 is the number of frames in a movie.
- artificial\_movie\_frame [1, 20]: A single video frame for each of the 20 videos
- artificial\_movie\_labels [1, 20]: 10 original videos. Each video is represented as 1x scale or 2x scale so there are 20 videos in total.
- artificial\_movie\_luminance [2, 10, 11]:

- contrast\_movie\_contrast [2, 10, 11]
- contrast\_movie\_frame [1, 4]
- contrast\_movie\_labels [1, 4]
- contrast\_movie\_luminance [2, 10, 11]

- natural\_movie\_contrast [2, 10, 11]: contrast information for each movie frame
- natural\_movie\_frame [1, 20]
- natural\_movie\_labels [1, 20]
- natural\_movie\_luminance [2, 10, 11]: illuminance information for each movie frame

# stim\_matrix.mat
pixel values of each image

- artificial\_movie\_labels (1, 20)
- contrast\_movie\_labels (1, 4)
- image\_paths (2, 3, 10, 11): a typical path looks like '/Users/yoonbai/Research/perceptualStraightening/LN\_LN\_model/stimulus/original\_images/stimSelected-zoom2x/movie05-prairie1/synthetic06.png'
- natural\_movie_labels (1, 20): a typical label looks like 'natural-01-chironomus-1x'
- stim\_matrix (2, 3, 10, 512, 512, 11): same as below
- stim\_matrix\_blurred (2 (scale x1 and scale x2), 3 (artificial, contrast, natural movie), 10 (10 original movie), 512 (image width), 512 (image width), 11 (n frames in a movie)):

Available video categories are:
_omit here_
07 is a person walking on the street, visually nice
They are numbered from 01 to 10
