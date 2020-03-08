"""
Util functions
"""
import os
import sys
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def compute_fps(video_path):
    """
    Given video, computes the frame rate (FPS).

    :param str video_path: path to video file (mp4, avi)
    :return: frames-per-second
    """
    cap = cv2.VideoCapture(video_path)

    # The last frame's index within 1 sec interval is the FPS value
    last_frame_idx = 0
    frame_timestamp = 0.0

    # Play the video till the first second
    while frame_timestamp < 1.0:
        ret, frame = cap.read()

        if not ret:
            break

        last_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    cap.release()

    fps = last_frame_idx - 1

    return fps


def get_video_name(path):
    filename = path.split('/')[-1]
    vid_name = filename.split('.')[0]

    return vid_name


def save_video_frames(video_path, fps_desired, save_dir, img_format='jpg', verbose=False):
    """
    Given the video, extracts video frames at the desired rate &
    saves frames to disk.

    :param str video_path: path to video file (mp4, avi)
    :param int fps_desired: frame rate (frames-per-sec)
    :param str save_dir: save frames at $save_dir/video_name/
    :param str img_format: image compression format (jpg, png)
    :param bool verbose: print info
    """
    video_name = get_video_name(video_path)
    frames_dir = os.path.join(save_dir, video_name)

    # Create directory (with video name as folder)
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Compute FPS of the given video
    fps_original = compute_fps(video_path)

    # Sample every n'th frame
    sample_rate = max(fps_original // fps_desired, 1)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if verbose:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('(W, H) = ({}, {})'.format(frame_width, frame_height))

    # Keep track of the sampled frame count
    sampled_frame_idx = 0

    # The filename length, for pre-pending 0's
    filename_len = len(str(total_frames)) + 1

    while True:
        ret, frame = cap.read()

        # terminate
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_idx % sample_rate == 0:
            # Prepend 0's to image filename (prefix)
            prefix = '0' * (filename_len - len(str(sampled_frame_idx)))

            frame_filename = ''.join([prefix, str(sampled_frame_idx), '.{}'.format(img_format)])

            # Save frame to disk
            frame_path = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

            sampled_frame_idx += 1

        # frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        # print('{:d} || {:.2f} || {}'.format(frame_idx, frame_timestamp, is_sampled))
    cap.release()

    if verbose:
        print('Frame length - sampled: {} | original: {}'.format(sampled_frame_idx, total_frames))
        print('Frame rate   - sampled: {} | original: {}'.format(fps_desired, fps_original))
        print('\nSaved at {}'.format(frames_dir))


def compute_validation_metrics(model, dataloader, device, size):
    """
    For the given model, computes accuracy & loss on validation/test set.

    :param nn.Module model: model
    :param dataloader: validation/test set dataloader
    :param device: cuda/cpu device where the model resides
    :param size: no. of samples (subset) to use
    :return: metrics {'accuracy', 'loss'}
    :rtype: dict
    """
    model.eval()
    with torch.no_grad():
        batch_size = dataloader.batch_size
        loss = 0.0
        num_correct = 0

        n_iters = max(1, size // batch_size)

        # Evaluate on mini-batches & then average over the total
        for i, batch in enumerate(dataloader):
            # Load to device, for the list of batch tensors
            batch = [d.to(device) for d in batch]
            inputs, label = batch[:-1], batch[-1]

            # Forward Pass
            label_logits = model(*inputs)

            # Compute Accuracy
            label_predicted = torch.argmax(label_logits, dim=1)
            correct = (label == label_predicted)
            num_correct += correct.sum().item()

            # Compute Loss
            loss += F.cross_entropy(label_logits, label, reduction='mean')

            if i >= n_iters:
                break

        # Total Samples
        total = n_iters * batch_size

        # Final Accuracy
        accuracy = 100.0 * (num_correct / total)

        # Final Loss (averaged over mini-batches - n_iters)
        loss = loss / n_iters

        metrics = {'accuracy': accuracy, 'loss': loss}

        return metrics


def plot_images(dataloader, idx2label=None, num_plots=4):
    """
    For plotting input data (after preprocessing with dataloader). \n
    Helper for sanity check.
    """
    for i, data in enumerate(dataloader):
        # Read dataset, select one random sample from the mini-batch
        batch_size = len(data)
        idx = np.random.choice(batch_size)
        img = data[idx]

        # PyTorch uses (3, H, W); thus permute channels to (H, W, 3)
        img = img.permute(1, 2, 0)

        if idx2label:
            # Map label index to class name
            label = idx2label[data['label'].item()]
            plt.text(220, 220, label, bbox=dict(fill=True, facecolor='white', edgecolor='blue', linewidth=2))

        # Plot Data
        plt.imshow(img)
        plt.show()

        i += 1

        if i >= num_plots:
            break


def setup_logs_file(parser, log_dir, file_name='train_log.txt'):
    """
    Generates log file and writes the executed python flags for the current run,
    along with the training log (printed to console). \n

    This is helpful in maintaining experiment logs (with arguments). \n

    While resuming training, the new output log is simply appended to the previously created train log file.

    :param parser: argument parser object
    :param log_dir: file path (to create)
    :param file_name: log file name
    :return: train log file
    """
    log_file_path = os.path.join(log_dir, file_name)

    log_file = open(log_file_path, 'a+')

    # python3 file_name.py
    log_file.write('python3 ' + sys.argv[0] + '\n')

    # Add all the arguments (key value)
    args = parser.parse_args()

    for key, value in vars(args).items():
        # write to train log file
        log_file.write('--' + key + ' ' + str(value) + '\n')

    log_file.write('\n\n')
    log_file.flush()

    return log_file


def print_and_log(msg, log_file):
    """
    :param str msg: Message to be printed & logged
    :param file log_file: log file
    """
    log_file.write(msg + '\n')
    log_file.flush()

    print(msg)


def str2bool(v):
    v = v.lower()
    assert v in ['true', 'false'], 'Option requires: "true" or "false"'
    return v == 'true'
