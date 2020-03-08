"""
Reads processed dataset json & numpy files.

JSON format:
{
    'data':
        [{'video_idx', 'video_name', 'video_length', 'label_idx'}],

    'memmap_size': tuple[int, int, int]     # (total_videos, max_video_len, emb_dim)
}

Numpy array format:
- shape = [total_videos, max_video_len, emb_dim]
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json


class ActionDataset(Dataset):
    def __init__(self, json_file, embedding_file, max_video_len=False):
        # Parse JSON
        json_data = self._read_json(json_file)

        # DataFrame
        self.data_df = pd.read_json(json_data['data'])

        # Setup Video Data
        memmap_shape = tuple(json_data['memmap_shape'])     # [total_videos, max_video_len, emb_dim]

        self.embeddings = np.memmap(embedding_file, mode='r', dtype='float32', shape=memmap_shape)

        self.video_lengths = self.data_df['video_length'].tolist()
        self.labels = self.data_df['label_idx'].tolist()

        self.max_video_len = max_video_len if max_video_len else memmap_shape[1]
        self.embedding_dim = memmap_shape[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx])
        video_len = self.video_lengths[idx]
        label = self.labels[idx]

        return embedding, video_len, label

    @staticmethod
    def _read_json(json_file):
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        return json_data


if __name__ == '__main__':
    jsn = '/home/axe/Datasets/UCF_101/train_1_fps.json'
    npy = '/home/axe/Datasets/UCF_101/train_1_fps_res18.npy'

    dataset = ActionDataset(jsn, npy)
    print(dataset.__len__())

    dataloader = DataLoader(dataset, batch_size=1024)

    for batch in dataloader:
        emb_tensor, video_length, label_idx = batch[:]
        print('-')
