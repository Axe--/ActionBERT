"""
Given path to UCF dataset's video directory, performs train-val split (saved as csv),
and computes video frames to disk, for the given frame-rate.

Creates the following csv files in the `--out_dir`:

- train_temp.csv
- val_temp.csv

CSV format:
`video_name, label_idx`

These csv's are temporary, as we need to run `prepare_data.py`
to generate the final dataset file (csv).

Also creates a new directory within the `--out_dir`
for storing video frames, organized as follows:

├── out_dir
    │
    └── frames
        │
        ├── video_1
        │    ├── frame_1.jpg
        │    │   ...
        │    └── frame_n.jpg
        │
        └── video_k
             ├── frame_1.jpg
             │   ...
             └── frame_m.jpg
"""
import numpy as np
import argparse
import os
import glob
from utils import save_video_frames

"""
python3 prepare_ucf101.py \
-v /home/axe/Datasets/UCF_101/videos \
-o /home/axe/Datasets/UCF_101 \
-s 0.8 -fps 1
"""


def _filename(path):
    """
    Extracts filename from file path.

    >>> _filename('UCF_101/videos/Biking/v_Biking_g01_c01.avi')
    'Biking/v_Biking_g01_c01.avi'

    :param str path: path to video file
    :return: file name (containing class)
    """
    filename = path.split('/')[-2:]

    filename = '/'.join(filename)

    return filename


def train_val_split(cls_name, video_dir, cls_idx, split_ratio=0.8):
    """
    For the given video class sub-directory, performs train-val split
    and computes filenames along with class label idx.

    :param str cls_name: class label name
    :param str video_dir: video directory
    :param int cls_idx: class label index
    :param split_ratio: train-val set split ratio
    :returns: train & validation lists containing
                tuples of filenames & class idx
    """
    # Get all files in the given class directory
    paths = sorted(glob.glob(os.path.join(video_dir, cls_name, '*.avi')))

    # Set seed (reproducibility) & shuffle
    np.random.seed(0)
    np.random.shuffle(paths)

    split_idx = int(len(paths) * split_ratio)

    train_paths = paths[:split_idx]
    val_paths = paths[split_idx:]

    # Extract filenames from paths & also insert the class label index
    train_fname_cls_idxs = [(_filename(path), cls_idx) for path in train_paths]
    val_fname_cls_idxs = [(_filename(path), cls_idx) for path in val_paths]

    # E.g.  [['Biking/v_Biking_g01_c01.avi', 12], ...]
    return train_fname_cls_idxs, val_fname_cls_idxs


def _write_to_csv(fname_cls_idxs, out_file):
    """
    Given the filename-class_idx list,
    saves the tuple to csv file.

    :param fname_cls_idxs: (filename, class_idx) tuples
    :type fname_cls_idxs: list[tuple[str, int]]

    :param str out_file: path to output csv file
    """
    with open(out_file, 'w') as f:
        # Create columns
        f.write('video_name' + ',' + 'label_idx' + '\n')

        # Append data
        for fname_cls in fname_cls_idxs:
            class_idx = fname_cls[1]
            filename = fname_cls[0]     # e.g. Bike/v_Bike_c5.avi

            # Clip out the video extension & parent folder (e.g --> v_Bike_c5)
            video_name = filename.split('.')[0].split('/')[-1]

            f.write(video_name + ',' + str(class_idx) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    # Dataset params
    parser.add_argument('-v',   '--video_dir',      type=str,       help='input videos directory', required=True)
    parser.add_argument('-o',   '--out_dir',        type=str,       help='contains frame sub-dirs & split set files', required=True)
    parser.add_argument('-s',   '--split_ratio',    type=float,     help='train-val split ratio', default=0.8)
    parser.add_argument('-fps', '--frame_rate',     type=int,       help='frame-rate (FPS) for sampling videos', default=1)

    args = parser.parse_args()

    # Read all action class names
    classes = sorted(glob.glob(os.path.join(args.video_dir, '*')))
    classes = [cls.split('/')[-1] for cls in classes]

    assert len(classes) == 101, 'The UCF-101 dataset expects total 101 classes, but {} found!'.format(len(classes))

    # For each class in the videos dir, perform train-val split
    class2idx = {cls: i for i, cls in enumerate(classes)}

    # Store the video filename along with class label index
    train_data = []
    val_data = []

    for cls_idx, cls_name in enumerate(classes):
        # Get the train-val split filename-class_idx tuples
        train_fname_cls_idx, val_fname_cls_idx = train_val_split(cls_name, args.video_dir, cls_idx, args.split_ratio)

        train_data += train_fname_cls_idx
        val_data += val_fname_cls_idx

    # list of tuples - (video_filenames, class_idx)
    dataset = sorted(train_data + val_data)

    # Parse videos & save frames to disk
    save_frames_dir = os.path.join(args.out_dir, 'frames_{}_fps'.format(args.frame_rate))

    if not os.path.exists(save_frames_dir):
        os.makedirs(save_frames_dir)

    total = len(dataset)
    for i, sample in enumerate(dataset):
        filename = sample[0]

        video_path = os.path.join(args.video_dir, filename)

        # save frames
        save_video_frames(video_path, args.frame_rate, save_frames_dir)

        if i % 1000 == 0:
            print('{} / {}'.format(i, total))

    print('Done! Video Frames saved in {}'.format(save_frames_dir))

    # Save train & val splits as csv files
    train_csv = os.path.join(args.out_dir, 'train_temp_ucf101.csv')
    val_csv = os.path.join(args.out_dir, 'val_temp_ucf101.csv')

    _write_to_csv(train_data, train_csv)
    _write_to_csv(val_data, val_csv)

    print('Train & Validation sets saved in:\n{}\n{}\n'.format(train_csv, val_csv))
