import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn


class DatasetUCF11(data.Dataset):

      def __init__(self, data_path, seq_length, transforms):
          self.data_path = data_path
          # outputs are lists, MUST be inthe same order: image + label 
          self.seq_length = seq_length
          self.list_of_frames = []
          self.transforms = transforms

      # returns a tensor with all frames in one video
      def load_video(self, path): 
          # sorted: definitely must be in the increasing order
          self.list_of_frames = sorted(os.listdir(path), key=lambda x:int(x.split('.')[0]))
          video_sequence = []
          # add random starting frame
          rand_start = torch.randint(0, len(self.list_of_frames), (1,)).item() 
          for num in range(rand_start, min(self.seq_length+rand_start, len(self.list_of_frames))):
              # should it be RGB?
              im = self.transforms(Image.open(os.path.join(path, self.list_of_frames[num])).convert("RGB"))
              video_sequence.append(im)

          # this returns FloatTensor seq_length x 3 x H x W
          video_sequence = torch.stack(video_sequence, dim=0)
          return video_sequence

      #one datapoint is one video, this method must point to all videos for each Class, NOT individual frames! 
      def __len__(self):
          return len(os.listdir(self.data_path))

      # access a directory in the train dataset
      # idx refers to the dir's index
      # activities are enumerated in alphabetic order
      def __getitem__(self, idx):
          # Load data
          vid = os.listdir(self.data_path)[idx]
          X = self.load_video(os.path.join(self.data_path, vid))
          # not it's float for some reason
          if 'basketball' in vid:
              y = torch.tensor([0], dtype=torch.long)
          elif 'biking' in vid:
              y = torch.tensor([1], dtype=torch.long)
          elif 'diving' in vid:
              y = torch.tensor([2], dtype=torch.long)
          elif 'golf' in vid:
              y = torch.tensor([3], dtype=torch.long)
          elif 'horse' in vid:
              y = torch.tensor([4], dtype=torch.long)
          elif 'soccer' in vid:
              y = torch.tensor([5], dtype=torch.long)
          elif 'swing' in vid:
              y = torch.tensor([6], dtype=torch.long)
          elif 'tennis' in vid:
              y = torch.tensor([7], dtype=torch.long)
          elif 'trampoline' in vid:
              y = torch.tensor([8], dtype=torch.long)
          elif 'volleyball' in vid:
              y = torch.tensor([9], dtype=torch.long)
          elif 'walking' in vid:
              y = torch.tensor([10], dtype=torch.long)
          return idx, X, y
