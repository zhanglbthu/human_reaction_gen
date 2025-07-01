from os.path import join as pjoin
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import random
import codecs as cs
import os.path as osp
import copy
from abc import ABCMeta, abstractmethod

import warnings
warnings.filterwarnings(action='ignore', module='mmcv', category=UserWarning)

from .pipeline import *


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

class VimoMotionDataset(Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        motion_file_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                motion_file_list.append(line.strip())

        for motion_file in tqdm(motion_file_list):
            try:
                motion = np.load(pjoin(opt.data_root, motion_file))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size + 1)
                self.data.append(motion)
            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                               joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (
                                                                          joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion
    

class VimoMotionDatasetEval(Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 20

        data_dict = {}
        motion_file_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                motion_file_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(motion_file_list):
            try:
                motion = np.load(pjoin(opt.data_root, name))
                if (len(motion)) < min_motion_len or (len(motion) > 200):
                    continue
                
                data_dict[name] = {'motion': motion,
                                    'length': len(motion)}
                new_name_list.append(name)
                length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length = data['motion'], data['length']
        

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            # subtract one unit length
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            # maintain the original length but make it a multiple of the unit length
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        
        return motion, m_length
    

PIPELINES = Registry('pipeline')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


class VimoBaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, opt, mean, std,
                 data_prefix,
                 ann_file,
                 pipeline,
                 start_index=0,
                 modality='RGB',
                 ):
        super().__init__()
        self.opt = opt
        self.mean = mean
        self.std = std
        self.max_motion_length = opt.max_motion_length
        self.ann_file = ann_file
        self.data_prefix = osp.realpath(
            data_prefix) if data_prefix is not None and osp.isdir(
                data_prefix) else data_prefix
        self.start_index = start_index
        self.modality = modality

        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)

    def prepare_data(self, idx):
        """Prepare the data given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results = self.pipeline(results)

        motion = results['label']
        m_length = len(motion)

        if self.opt.unit_length < 10: 
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length 
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length 
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        # del results['label']
        # results['motion'] = motion
        # results['motion_length'] = m_length
        # return results

        return results['imgs'], motion, m_length, results['filename']

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        return self.prepare_data(idx)
    
    def inv_transform(self, data):
        return data * self.std + self.mean


class VimoDataset(VimoBaseDataset):
    def __init__(self, opt, mean, std, data_prefix, ann_file, pipeline, start_index=0, **kwargs):
        super().__init__(opt, mean, std, data_prefix, ann_file, pipeline, start_index=start_index, **kwargs)

    def load_annotations(self):
        video_infos = []
        ann_file_path = osp.join(self.data_prefix, self.ann_file)
        with open(ann_file_path, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_name, motion_name = line_split
                video_name = osp.join(self.data_prefix, video_name)
                motion_name = osp.join(self.data_prefix, motion_name)
                motion = np.load(motion_name)
                video_infos.append(dict(filename=video_name, label=motion, tar=False))
        return video_infos


img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
config_input_size = 224
config_num_frames = 16
config_num_clip = 1
config_num_crop = 1
scale_resize = int(256 / 224 * config_input_size)

train_pipeline = [
dict(type='DecordInit'),
dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config_num_frames, test_mode=True),   # TSN sampling strategy https://github.com/open-mmlab/mmaction2/issues/1379#issuecomment-1010938270
dict(type='DecordDecode'),
#dict(type='Resize', scale=(-1, scale_resize)),
dict(type='Resize', scale=(config_input_size, config_input_size), keep_ratio=False),
dict(type='Normalize', **img_norm_cfg),
dict(type='FormatShape', input_format='NCHW'),
dict(type='Collect', keys=['imgs', 'label', 'filename'], meta_keys=[]),
dict(type='ToTensor', keys=['imgs'])
]

val_pipeline = [
dict(type='DecordInit'),
dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config_num_frames, test_mode=True),
dict(type='DecordDecode'),
#dict(type='Resize', scale=(-1, scale_resize)),
#dict(type='CenterCrop', crop_size=config_input_size),
dict(type='Resize', scale=(config_input_size, config_input_size), keep_ratio=False),
dict(type='Normalize', **img_norm_cfg),
dict(type='FormatShape', input_format='NCHW'),
dict(type='Collect', keys=['imgs', 'label', 'filename'], meta_keys=[]),
dict(type='ToTensor', keys=['imgs'])
]

test_pipeline = [
dict(type='DecordInit'),
dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config_num_frames, test_mode=True),
dict(type='DecordDecode'),
#dict(type='Resize', scale=(-1, scale_resize)),
#dict(type='CenterCrop', crop_size=config_input_size),
dict(type='Resize', scale=(config_input_size, config_input_size), keep_ratio=False),
dict(type='Normalize', **img_norm_cfg),
dict(type='FormatShape', input_format='NCHW'),
dict(type='Collect', keys=['imgs'], meta_keys=[]),
dict(type='ToTensor', keys=['imgs'])
]