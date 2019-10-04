"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import pickle
import numpy as np
import caffe
import sys
import h5py
sys.path.insert(0, '/mnt/lustre/wangjiayun/repos/splatnet')
from splatnet.utils import rotate_3d
from splatnet.configs import SN_CATEGORIES, SN_CATEGORY_NAMES, SN_NUM_PART_CATEGORIES, MODELNET_DATA_DIR #
import pdb

def category_mask(category):
    """
    Get a mask vector for part categories corresponding to a specific shape category
    :param category: (string) shape categoiy name or id
    :return: [50] numpy array
    """
    if type(category) == str:
        if category.startswith('0'):
            category = SN_CATEGORIES.index(category)
        else:
            category = SN_CATEGORY_NAMES.index(category)
    mask = np.zeros(sum(SN_NUM_PART_CATEGORIES))
    mask[sum(SN_NUM_PART_CATEGORIES[:category]):sum(SN_NUM_PART_CATEGORIES[:category + 1])] = 1
    return mask


def points_test_single_area(subset, 
                           dims='x_y_z',    # combinations of 'x', 'y', 'z', 'nx', 'ny', 'nz' and 'one'
                           read_cache=True, write_cache=True, cache_dir='',
                           shuffle=False,
                           root=MODELNET_DATA_DIR):
    if not cache_dir:
        cache_dir = os.path.join(root, 'cache')

    if read_cache or write_cache:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        #os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, '{}.cache'.format(subset))

    if read_cache and os.path.exists(cache_path):
        with open(cache_path, mode='rb') as f:
            feat_list, label_list, hash_list = pickle.load(f)
    else:
        feat_list, label_list, hash_list = [], [], []

    if not feat_list:
        #hash_list = sorted([s[:-4] for s in filter(lambda s: s.endswith('.ply'), os.listdir(root))])
        if subset == "train":
            hash_list = sorted([s[:-3] for s in filter(lambda s: s.endswith('.h5') and "train" in s, os.listdir(root))])
        elif subset == "val":
            hash_list = sorted([s[:-3] for s in filter(lambda s: s.endswith('.h5') and "test" in s, os.listdir(root))])
        else:
            hash_list = sorted([s[:-3] for s in filter(lambda s: s.endswith('.h5') and "test" in s, os.listdir(root))])
        feat_list = []
        label_list = []
        for s in hash_list:
            data = h5py.File(os.path.join(root, s + '.h5'), 'r')
            #data = np.loadtxt(os.path.join(data_dir, s + '.npy'), skiprows=14)
            feat_list = feat_list + list(data["data"])
            label_list = label_list + list(data["label"])        
        if write_cache:
            with open(cache_path, mode='wb') as f:
                pickle.dump((feat_list, label_list, hash_list), f)

    # adding 'one' as an additional feature
    feat_dict = dict(zip('x_y_z_r_g_b_one'.split('_'), range(7)))
    feat_idxs = [feat_dict[f] for f in dims.split('_')]
    if 'one' in dims.split('_'):
        feat_list = [np.concatenate((d, np.ones((len(d), 1))), axis=1) for d in feat_list]

    feat_list = [d[:, feat_idxs] for d in feat_list]

    # shuffle
    if shuffle:
        idx = np.random.permutation(len(hash_list))
        feat_list, label_list, hash_list = feat_list[idx], label_list[idx], hash_list[idx]
    return feat_list, label_list, hash_list


def points_all_categories(subset,
                          dims='x_y_z',  # combinations of 'x', 'y', 'z', 'nx', 'ny', 'nz' and 'one'
                          read_cache=True, write_cache=True, cache_dir='',
                          shuffle=False,
                          root=MODELNET_DATA_DIR):

    if not cache_dir:
        cache_dir = os.path.join(root, 'cache')

    if read_cache or write_cache:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        #os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, '{}.cache'.format(subset))

    if read_cache and os.path.exists(cache_path):
        with open(cache_path, mode='rb') as f:
            feats, object_labels, part_labels, shape_ids = pickle.load(f)
    else:
        feats, object_labels, part_labels, shape_ids = [], [], [], []

    if not feats:
        for i, c in enumerate(SN_CATEGORIES):
            c_feats, c_object_labels, c_part_labels, c_shape_ids = points_test_single_area(subset,
                                                                                         category=c,
                                                                                         dims='x_y_z_nx_ny_nz',
                                                                                         read_cache=True,
                                                                                         write_cache=False,
                                                                                         cache_dir=cache_dir,
                                                                                         shuffle=False,
                                                                                         root=root)
            c_part_labels = [v + sum(SN_NUM_PART_CATEGORIES[:i]) for v in c_part_labels]
            feats.extend(c_feats)
            object_labels.extend(c_object_labels)
            part_labels.extend(c_part_labels)
            shape_ids.extend(c_shape_ids)

        if write_cache:
            with open(cache_path, mode='wb') as f:
                pickle.dump((feats, object_labels, part_labels, shape_ids), f)

    # adding 'one' as an additional feature
    feat_dict = dict(zip('x_y_z_nx_ny_nz_one'.split('_'), range(7)))
    feat_idxs = [feat_dict[f] for f in dims.split('_')]
    if 'one' in dims.split('_'):
        feats = [np.concatenate((d, np.ones((len(d), 1))), axis=1) for d in feats]
    feats = [d[:, feat_idxs] for d in feats]

    # shuffle
    if shuffle:
        idx = np.random.permutation(len(shape_ids))
        feats, object_labels, part_labels, shape_ids = feats[idx], object_labels[idx], part_labels[idx], shape_ids[idx]

    return feats, object_labels, part_labels, shape_ids


class InputShapenet(caffe.Layer):
    def _restart(self):
        # make a deep copy
        data = [d.copy() for d in self.data_copy]
        label = [l.copy() for l in self.label_copy]

        # duplicate if necessary to fill batch
        num_samples = len(data)
        if num_samples < self.batch_size:
            idx = np.concatenate((np.tile(np.arange(num_samples), (self.batch_size // num_samples, )),
                                  np.random.permutation(num_samples)[:(self.batch_size % num_samples)]), axis=0)
            data, label = [data[i] for i in idx], [label[i] for i in idx]
            num_samples = self.batch_size

        # shuffle samples
        idx = np.random.permutation(num_samples)
        data, label = [data[i] for i in idx], [label[i] for i in idx]

        # sample to a fixed length
        """for i in range(num_samples):
            k = len(data[i])
            idx = np.concatenate((np.tile(np.arange(k), (self.sample_size // k, )),
                                  np.random.permutation(k)[:(self.sample_size % k)]), axis=0)
            data[i] = data[i][idx, :]
            label[i] = label[i][idx]"""
            
        # reshape and reset index
        self.data = data #data[:, self.feat_dims].reshape(num_samples, self.sample_size, -1, 1).transpose(0, 2, 3, 1)
        self.label = label #label.reshape(num_samples, self.sample_size, -1, 1).transpose(0, 2, 3, 1)
        self.index = 0
        
    def _data_processing_batch(self, data, label):
        data = np.concatenate(data, axis=0)     # (NxS) x C
        label = np.concatenate(label, axis=0)   # (NxS)
        # data aug. # TODO should this be done at batch level?
        if self.jitter_rotation > 0:
            rotations = (('x', (2 * np.random.rand() - 1) * self.jitter_rotation * np.pi / 180.0),
                         ('y', (2 * np.random.rand() - 1) * self.jitter_rotation * np.pi / 180.0),
                         ('z', (2 * np.random.rand() - 1) * self.jitter_rotation * np.pi / 180.0))
            if 'x' in self.raw_dims:
                feat_idx = self.raw_dims.index('x')
                data[:, feat_idx:feat_idx + 3] = rotate_3d(data[:, feat_idx:feat_idx + 3], rotations)
            if 'r' in self.raw_dims:
                feat_idx = self.raw_dims.index('r')
                data[:, feat_idx:feat_idx + 3] = rotate_3d(data[:, feat_idx:feat_idx + 3], rotations)
        if self.jitter_stretch > 0:
            stretch_strength = (2 * np.random.rand(3) - 1) * self.jitter_stretch + 1
            if 'x' in self.raw_dims:
                feat_idx = self.raw_dims.index('x')
                data[:, feat_idx:feat_idx + 3] *= stretch_strength
            if 'r' in self.raw_dims:
                feat_idx = self.raw_dims.index('r')
                data[:, feat_idx:feat_idx + 3] *= stretch_strength
                data[:, feat_idx:feat_idx + 3] /= np.sqrt(
                    np.power(data[:, feat_idx:feat_idx + 3], 2).sum(axis=1, keepdims=True))
        if self.jitter_xyz > 0 and 'x' in self.raw_dims:
            feat_idx = self.raw_dims.index('x')
            data[:, feat_idx:feat_idx + 3] += ((2 * np.random.rand(3) - 1) * self.jitter_xyz)
        return data, label

    def setup(self, bottom, top):
        params = dict(subset='train', batch_size=32, sample_size=3000,
                      feat_dims='x_y_z',        # choose from 'x', 'y', 'z', 'nx', 'ny', 'nz' and 'one'
                      jitter_xyz=0.01,          # random displacements
                      jitter_stretch=0.1,       # random stretching (uniform random within +- this value)
                      jitter_rotation=10,       # random rotation along three axis (in degrees)
                      root=MODELNET_DATA_DIR)
        params.update(eval(self.param_str))
        self.batch_size = params['batch_size']
        self.sample_size = params['sample_size']
        self.jitter_xyz = params['jitter_xyz']
        self.jitter_stretch = params['jitter_stretch']
        self.jitter_rotation = params['jitter_rotation']

        self.raw_dims = []
        for feat_group in [['x', 'y', 'z'], ['r', 'g', 'b'], ['one']]:
            if np.any([f in feat_group for f in params['feat_dims'].split('_')]):
                self.raw_dims.extend(feat_group)
        self.feat_dims = [self.raw_dims.index(f) for f in params['feat_dims'].split('_')]
        params['root'] = MODELNET_DATA_DIR
        data, label, _ = points_test_single_area(params['subset'], 
                                                   dims='_'.join(self.raw_dims), root=params['root'])
        self.data_copy = data
        self.label_copy = label
        self.top_names = ['data', 'label']
        self.top_channels = [len(self.raw_dims), 1]

        if len(top) != len(self.top_names):
            raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                            (len(self.top_names), len(top)))

        self._restart()

    def reshape(self, bottom, top):
        for top_index, name in enumerate(self.top_names):
            if top_index == 0:
                shape = (self.batch_size, self.top_channels[top_index], 1, self.sample_size)
            else:
                shape = (self.batch_size, self.top_channels[top_index], 1, 1)
            top[top_index].reshape(*shape)
            
            """filelist = [] #np.array
            batch_idx = np.choice(len(filelist), BATCH_SIZE)
            file_batch = filelist[batch_idx]
            lam = lambda x: filepath2_image(x) # filepath => image
            lam2 = lambda x: img_precess(x) # image => processed_image"""
            

    def forward(self, bottom, top):
        temp_data, temp_label = self._data_processing_batch( self.data[self.index:self.index+self.batch_size], self.label[self.index:self.index+self.batch_size] )
        temp_data = temp_data.reshape(self.batch_size, self.sample_size, -1, 1).transpose(0, 2, 3, 1)
        temp_label = temp_label.reshape(self.batch_size, 1, -1, 1).transpose(0, 2, 3, 1)
        top[0].data[...] = temp_data
        top[1].data[...] = temp_label
        self.index += self.batch_size
        if self.index + self.batch_size > len(self.data):
            self._restart()

    def backward(self, top, propagate_down, bottom):
        pass


class InputShapenetAllCategories(caffe.Layer):
    def _restart(self):
        # make a deep copy
        data = [d.copy() for d in self.data_copy]
        label = [l.copy() for l in self.label_copy]
        label_mask = [l.copy() for l in self.label_mask_copy]

        # duplicate if necessary to fill batch
        num_samples = len(data)
        if num_samples < self.batch_size:
            idx = np.concatenate((np.tile(np.arange(num_samples), (self.batch_size // num_samples, )),
                                  np.random.permutation(num_samples)[:(self.batch_size % num_samples)]), axis=0)
            data, label_mask, label = [data[i] for i in idx], [label_mask[i] for i in idx], [label[i] for i in idx]
            num_samples = self.batch_size

        # shuffle samples
        idx = np.random.permutation(num_samples)
        data, label_mask, label = [data[i] for i in idx], [label_mask[i] for i in idx], [label[i] for i in idx]

        # sample to a fixed length
        for i in range(num_samples):
            k = len(data[i])
            idx = np.concatenate((np.tile(np.arange(k), (self.sample_size // k, )),
                                  np.random.permutation(k)[:(self.sample_size % k)]), axis=0)
            data[i] = data[i][idx, :]
            label[i] = label[i][idx]

        data = np.concatenate(data, axis=0)     # (NxS) x C
        label = np.concatenate(label, axis=0)   # (NxS)
        label_mask = np.concatenate([l.reshape(1, -1, 1, 1) for l in label_mask], axis=0)     # N x 50 x 1 x 1

        # data aug.
        if self.jitter_rotation > 0:
            rotations = (('x', (2 * np.random.rand() - 1) * self.jitter_rotation * np.pi / 180.0),
                         ('y', (2 * np.random.rand() - 1) * self.jitter_rotation * np.pi / 180.0),
                         ('z', (2 * np.random.rand() - 1) * self.jitter_rotation * np.pi / 180.0))
            if 'x' in self.raw_dims:
                feat_idx = self.raw_dims.index('x')
                data[:, feat_idx:feat_idx + 3] = rotate_3d(data[:, feat_idx:feat_idx + 3], rotations)
            if 'nx' in self.raw_dims:
                feat_idx = self.raw_dims.index('nx')
                data[:, feat_idx:feat_idx + 3] = rotate_3d(data[:, feat_idx:feat_idx + 3], rotations)
        if self.jitter_stretch > 0:
            stretch_strength = (2 * np.random.rand(3) - 1) * self.jitter_stretch + 1
            if 'x' in self.raw_dims:
                feat_idx = self.raw_dims.index('x')
                data[:, feat_idx:feat_idx + 3] *= stretch_strength
            if 'nx' in self.raw_dims:
                feat_idx = self.raw_dims.index('nx')
                data[:, feat_idx:feat_idx + 3] *= stretch_strength
                data[:, feat_idx:feat_idx + 3] /= np.sqrt(
                    np.power(data[:, feat_idx:feat_idx + 3], 2).sum(axis=1, keepdims=True))
        if self.jitter_xyz > 0 and 'x' in self.raw_dims:
            feat_idx = self.raw_dims.index('x')
            data[:, feat_idx:feat_idx + 3] += ((2 * np.random.rand(3) - 1) * self.jitter_xyz)

        # reshape and reset index
        self.data = data[:, self.feat_dims].reshape(num_samples, self.sample_size, -1, 1).transpose(0, 2, 3, 1)
        self.label = label.reshape(num_samples, self.sample_size, -1, 1).transpose(0, 2, 3, 1)
        self.label_mask = label_mask
        self.index = 0

    def setup(self, bottom, top):
        params = dict(subset='train', batch_size=32, sample_size=3000,
                      feat_dims='x_y_z',        # choose from 'x', 'y', 'z', 'one'
                      jitter_xyz=0.01,          # random displacements
                      jitter_stretch=0.1,       # random stretching (uniform random within +- this value)
                      jitter_rotation=10,       # random rotation along three axis (in degrees)
                      root=MODELNET_DATA_DIR,
                      output_mask=False)
        params.update(eval(self.param_str))
        self.batch_size = params['batch_size']
        self.sample_size = params['sample_size']
        self.jitter_xyz = params['jitter_xyz']
        self.jitter_stretch = params['jitter_stretch']
        self.jitter_rotation = params['jitter_rotation']
        self.output_mask = params['output_mask']

        self.raw_dims = []
        for feat_group in [['x', 'y', 'z'], ['nx', 'ny', 'nz'], ['one']]:
            if np.any([f in feat_group for f in params['feat_dims'].split('_')]):
                self.raw_dims.extend(feat_group)
        self.feat_dims = [self.raw_dims.index(f) for f in params['feat_dims'].split('_')]

        data, category, label, _ = points_all_categories(params['subset'],
                                                         dims='_'.join(self.raw_dims), root=params['root'])
        self.data_copy = data
        self.label_copy = label
        self.label_mask_copy = [category_mask(c) for c in category]
        self.top_names = ['data', 'label', 'label_mask']
        self.top_channels = [len(self.raw_dims), 1, sum(SN_NUM_PART_CATEGORIES)]

        if not self.output_mask:
            self.top_names, self.top_channels = self.top_names[:2], self.top_channels[:2]

        if len(top) != len(self.top_names):
            raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                            (len(self.top_names), len(top)))

        self._restart()

    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size, self.top_channels[0], 1, self.sample_size)
        top[1].reshape(self.batch_size, self.top_channels[1], 1, self.sample_size)
        if self.output_mask:
            top[2].reshape(self.batch_size, self.top_channels[2], 1, 1)

    def forward(self, bottom, top):
        top[0].data[...] = self.data[self.index:self.index+self.batch_size]
        top[1].data[...] = self.label[self.index:self.index+self.batch_size]
        if self.output_mask:
            top[2].data[...] = self.label_mask[self.index:self.index+self.batch_size]

        self.index += self.batch_size
        if self.index + self.batch_size > len(self.data):
            self._restart()

    def backward(self, top, propagate_down, bottom):
        pass

