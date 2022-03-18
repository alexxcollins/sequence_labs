import os
from pathlib import Path
import numpy as np
import random
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn

class DatasetUCF11(data.Dataset):
    
    def __init__(self, data_path, seq_length, max_per_cat, transforms):
        self.actions = ['shooting','biking','diving' ,'golf',
                        'riding' ,'juggle' ,'swing','tennis',
                        'jumping' ,'spiking' ,'walk']
        self.action_dict = dict(zip(self.actions, range(11)))
        self.data_path = Path(data_path)
        self.seq_length = seq_length
        self.max_per_cat = max_per_cat
        self.list_of_videos = self.list_videos(self.data_path,
                                              self.max_per_cat)
        # outputs are lists, MUST be inthe same order: image + label 
        self.list_of_frames = []
        self.transforms = transforms
            
    def list_videos(self, path, max_per_cat):
        # code will break if videos have less than self.seq_length of frames
        # because it will attempt to return jagged tensors
        list_of_videos = []
        # randomly choose (without replacement) max_per_cat videos from each action category
        for act in self.actions:
            vids = sorted(path.glob('*{}*'.format(act)))
            k = min(max_per_cat, len(vids))
            list_of_videos.extend(random.sample(vids, k))
        # remove videos with fewer frames than seq_length
        for vid in list_of_videos:
            if len(sorted(vid.glob('[!.]*'))) <= self.seq_length:
                list_of_videos.remove(vid)
        
        
        return list_of_videos
        
    # returns a tensor with all frames in one video
    def load_video(self, path): 
        # sorted: definitely must be in the increasing order
#             self.list_of_frames = sorted(os.listdir(path), key=lambda x:int(x.split('.')[0]))
        self.list_of_frames = sorted(path.glob('*.jpg'),
                                    key = lambda x: int(x.stem))
        video_sequence = []
        # add random starting frame
        rand_start = torch.randint(0, len(self.list_of_frames)-self.seq_length, (1,)).item() 
        for num in range(rand_start, self.seq_length+rand_start):
            # should it be RGB?
            im = self.transforms(Image.open(self.list_of_frames[num]).convert("RGB"))
            video_sequence.append(im)

        # this returns FloatTensor seq_length x 3 x H x W
        video_sequence = torch.stack(video_sequence, dim=0)
        return video_sequence

      #one datapoint is one video, this method must point to all videos for each Class, NOT individual frames! 
    def __len__(self):
        return len(self.list_of_videos)

    # access a directory in the train dataset
    # idx refers to the dir's index
    # activities are enumerated in alphabetic order
    def __getitem__(self, idx):
        # Load data
        vid = self.list_of_videos[idx]
        X = self.load_video(vid)
        # use video file name to find action_string and then use self.action_dict to ret idx
        act_idx = self.action_dict[vid.name.split('_')[1]]
        y = torch.tensor(act_idx, dtype=torch.long).reshape(1,)
        return idx, X, y
    

