"""
Given the processed UCF-101 train/val set csv,
inserts the video-index & no. of frames to
each row (video_filename, class_label_idx)
and saves as json

Input CSV Format:
`video_name, label_idx`

Output JSON Format:
{
    'delete':
        [{'video_idx', 'video_name', 'video_length', 'label_idx'}],

    'memmap_size': tuple[int, int, int]     # (total_videos, max_video_len, emb_dim)
}

Computes embeddings from frame images (saved as numpy file):
- shape=[num_videos, max_frames, embedding_dim]

The `video_idx` in json file corresponds to the
embedding array's 0th axis (npy).

"""
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
from time import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import Compose, Resize, ToTensor, Normalize
from configs import CNN_NAMES, load_cnn
from utils import compute_max_frames_len


"""
python3 prepare_data.py -s train \
-f /home/axe/Datasets/UCF_101/frames_1_fps \
-c /home/axe/Datasets/UCF_101/train_temp.csv \
-o /home/axe/Datasets/UCF_101/processed_fps_1_res18 \
-m resnet18 -bs 1024 -nw 4
"""


class VideoFramesDataset(Dataset):
    """
    For computing frame embeddings
    """
    def __init__(self, frames_dir, data_df, transform=None):
        """
        Given frames directory and DataFrame (w/ relative path to frames),
        iterates over each row and reads frames from disk.

        For every video dir, the dataset is defined by the frame images.
        Thus, the total len = sum_{i=1:N}(num_frames_in_video_i)

        :param str frames_dir: video frames root directory
        :param pd.DataFrame data_df: delete containing frames folder names.
                                        Fields: `video_name, label_idx, num_frames`
        :param transform: image transforms (torchvision.transforms.transforms)
        """
        self.df = data_df
        self.root_dir = frames_dir
        self.transform = transform

        # Setup dataset (frame-path, label)
        self.frame_paths, self.labels = self.setup_data()

        # No. of frames per video
        self.num_frames = self.df['video_length'].tolist()

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        # Read image path & label index
        frame_path = self.frame_paths[idx]

        image = Image.open(frame_path).convert('RGB')

        # Resize((224, 224)); ToTensor(); Normalize(mean, std_dev)
        image = self.transform(image)       # uint8 --> float32

        return image

    def setup_data(self):
        """
        Given the video frames directory, compute the
        absolute paths to frames, along with corresponding labels

        :returns: frame paths & labels
        """
        # input delete
        video_names = self.df['video_name'].tolist()
        num_frames = self.df['video_length'].tolist()
        label_idxs = self.df['label_idx'].tolist()

        # output delete
        frame_path_list = []
        label_idx_list = []

        for video, n_frame, label in zip(video_names, num_frames, label_idxs):
            # Read frames
            frame_paths = sorted(glob.glob(os.path.join(self.root_dir, video, '*')))

            frame_path_list += frame_paths
            label_idx_list += [label] * n_frame

        return frame_path_list, label_idx_list


def _count_frames(folder, root_dir):
    frame_paths = glob.glob(os.path.join(root_dir, folder, '*'))

    return len(frame_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare delete for Action Recognition')

    # Dataset params
    parser.add_argument('-f',   '--frames_dir',     type=str,   help='input frames root directory', required=True)
    parser.add_argument('-c',   '--csv_file',       type=str,   help='input train/val temp csv file', required=True)
    parser.add_argument('-o',   '--out_dir',        type=str,   help='stores processed json & embeddings npy', required=True)

    # Model params
    parser.add_argument('-m',   '--model',          type=str,   help='pre-trained CNN (torchvision.models)', choices=CNN_NAMES)
    parser.add_argument('-s',   '--split',          type=str,   help='select split', choices=['train', 'val'], required=True)
    parser.add_argument('-bs',  '--batch_size',     type=int,   help='batch size for computing embeddings', default=128)

    # Misc params
    parser.add_argument('-nw',  '--num_workers',    type=int,   help='no. of worker threads for dataloader', default=1)
    parser.add_argument('-g',   '--gpu_id',         type=int,   help='cuda:gpu_id (torch.device)', default=0)

    args = parser.parse_args()

    start_time = time()

    # Set CUDA device
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    print('Selected Device: {}'.format(device))

    # Read csv (video_name, label)
    df = pd.read_csv(args.csv_file)

    total_videos = int(df['video_name'].count())

    # Add video index column (to be utilized by json)
    df['video_idx'] = range(total_videos)

    # Compute the sequence length (no. of frames) for each video (row)
    df['video_length'] = df['video_name'].apply(lambda x: _count_frames(x, args.frames_dir))

    # Image Mean & Std-Dev for Normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    dataset = VideoFramesDataset(args.frames_dir, df, Compose([Resize((224, 224)), ToTensor(), Normalize(mean, std)]))
    # dataset = VideoFramesDataset(args.frames_dir, df, Compose([Resize((224, 224)), ToTensor()]))    # for sanity check

    # Compute the max sequence length, needed for embedding array - [N, F, D]
    max_video_len = compute_max_frames_len(args.frames_dir)
    total_frames = dataset.__len__()

    print('Total Videos: {}  |  Total Frames: {}  |  Max Video length: {}'.
          format(total_videos, total_frames, max_video_len))

    dataloader = DataLoader(dataset, args.batch_size, num_workers=args.num_workers)

    # Load model
    model, emb_dim = load_cnn(args.model)
    model.to(device)

    # Create output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Processed JSON (output)
    data = df.to_json(orient='records')     # {'video_idx', 'video_name', 'video_length', 'label_idx'}
    memmap_shape = (total_videos, max_video_len, emb_dim)

    json_data = dict(data=data,
                     memmap_shape=memmap_shape,
                     split=args.split)

    json_file = os.path.join(args.out_dir, args.split + '.json')
    with open(json_file, "w") as f:
        json.dump(json_data, f)

    print('Processed json saved at {}'.format(json_file))

    # Embeddings file (output)
    embeddings_file = os.path.join(args.out_dir, args.split + '.npy')
    emb_temp_file = os.path.join(args.out_dir, args.split + '_temp.npy')

    # Embeddings [num_videos * frames_per_video, emb_dim]
    embeddings_temp = np.memmap(emb_temp_file, 'float32', 'w+', shape=(total_frames, emb_dim))
    video_lengths = df['video_length'].tolist()

    with torch.no_grad():
        i = 0
        for batch in dataloader:
            batch_size = batch.shape[0]
            frames = batch.to(device)

            # Forward pass --> to CPU --> to numpy
            emb = model(frames).cpu().detach().numpy()      # [batch_size, emb_dim]

            # Add to embeddings file
            embeddings_temp[i: i+batch_size, :] = emb

            i += batch_size

    # Reshape the embeddings array
    embeddings_final = np.memmap(embeddings_file, 'float32', 'w+', shape=memmap_shape)

    j = 0
    for video_idx, video_len in enumerate(video_lengths):
        embeddings_final[video_idx, :video_len, :] = embeddings_temp[j: j+video_len, :]

        j += video_len

    # Delete the temp file
    os.remove(emb_temp_file)

    print('The embeddings memmap file saved at {}'.format(embeddings_file))

    print('Total execution time {:.2f} secs'.format(time() - start_time))
