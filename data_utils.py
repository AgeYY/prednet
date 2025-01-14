import hickle as hkl
import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator

# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):
    def __init__(self, data_file, source_file, nt, label_file=None,
                 batch_size=8, shuffle=False, seed=None,
                 output_mode='error', sequence_start_mode='all', N_seq=None,
                 data_format=K.image_data_format()):
        self.X = hkl.load(data_file)  # X will be like (n_images, nb_cols, nb_rows, nb_channels) = (n_images, 128, 160, 3). data type is np.ndarray
        self.sources = hkl.load(source_file) # source for each image so when creating sequences can assure that consecutive frames are from same video. Shape is (n_images,), type is np.ndarray
        if not (label_file is None):
            self.label = hkl.load(label_file)
        else:
            self.label = {}
        self.nt = nt
        self.batch_size = batch_size
        self.data_format = data_format
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode
        self.source_name = []

        if self.data_format == 'channels_first':
            self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.im_shape = self.X[0].shape

        if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
            self.possible_label = [self.label[self.sources[i]] for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]]
        elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
            curr_location = 0
            possible_starts = []
            possible_label = []
            while curr_location < self.X.shape[0] - self.nt + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                    self.source_name.append(self.sources[curr_location]) # the source_name correspond to this location
                    possible_starts.append(curr_location)
                    possible_label.append(self.label[self.source_name[-1]])
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts
            self.possible_label = possible_label

        if shuffle:
            idx = range(len(self.possible_starts))
            idx_permut = np.random.permutation(idx).astype(int)
            self.possible_starts = [self.possible_starts[i] for i in idx_permut]
            self.possible_label = [self.possible_label[i] for i in idx_permut]
        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
            self.possible_label = self.possible_label[:N_seq]
        self.N_sequences = len(self.possible_starts)
        super(SequenceGenerator, self).__init__(len(self.possible_starts), batch_size, shuffle, seed)

    def __getitem__(self, null):
        return self.next()

    def next(self):
        with self.lock:
            current_index = (self.batch_index * self.batch_size) % self.n
            index_array, current_batch_size = next(self.index_generator), self.batch_size
        batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(index_array):
            idx = self.possible_starts[idx]
            batch_x[i] = self.preprocess(self.X[idx:idx+self.nt])
        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        return X.astype(np.float32) / 255

    def create_all(self, out_label=False):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx:idx+self.nt])
        if out_label:
            label_all = self.possible_label.copy()
            return X_all, label_all
        else:
            return X_all
