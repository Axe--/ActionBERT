import torch
import torch.nn as nn
import argparse
import os
import apex.amp as amp
from time import time
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataloader import BiLSTMDataset, TransformerDataset
from model import BiLSTM, Transformer
from utils import str2bool, print_and_log, setup_logs_file
from utils import compute_validation_metrics

"""
Train + Val:
python3 main.py --mode train --expt_dir ./results_log  --expt_name BERT --model bert \
--data_dir ~/Datasets/UCF_101/processed_fps_1_res18 --run_name res18_1fps_lyr_1_bs_256_lr_1e4 \
--num_layers 1 --batch_size 256 --epochs 300 --gpu_id 1 --opt_lvl 1  --num_workers 4 --lr 1e-4

Test:
"""


def main():
    parser = argparse.ArgumentParser(description='Video + QA setup for Social-IQ')

    # Experiment params
    parser.add_argument('--mode',           type=str,       help='train or test mode', required=True, choices=['train', 'test'])
    parser.add_argument('--expt_dir',       type=str,       help='root directory to save model & summaries', required=True)
    parser.add_argument('--expt_name',      type=str,       help='expt_dir/expt_name: organize experiments', required=True)
    parser.add_argument('--run_name',       type=str,       help='expt_dir/expt_name/run_name: organize training runs', required=True)

    # Model params
    parser.add_argument('--model',          type=str,       help='RNN vs Transformer', required=True, choices=['bilstm', 'bert'])
    parser.add_argument('--config_name',    type=str,       help='transformers pre-trained config name', default='bert-base-uncased')
    parser.add_argument('--use_pretrained', type=str2bool,  help='use pre-trained transformer', default='true')
    parser.add_argument('--num_layers',     type=int,       help='no. of layers in the RNN/Transformer', default=1)
    parser.add_argument('--num_cls',        type=int,       help='no. of class labels', default=101)
    parser.add_argument('--model_ckpt',     type=str,       help='resume train / perform inference; e.g. model_100.pth')

    # Data params
    parser.add_argument('--data_dir',       type=str,       help='root dir containing all delete files', required=True)
    parser.add_argument('--pred_output',    type=str,       help='prediction file (label, pred) pair on each line')

    # Training params
    parser.add_argument('--batch_size',     type=int,       help='batch size', default=8)
    parser.add_argument('--epochs',         type=int,       help='number of epochs', default=50)
    parser.add_argument('--lr',             type=float,     help='learning rate', default=1e-4)
    parser.add_argument('--log_interval',   type=int,       help='interval size for logging training summaries', default=100)
    parser.add_argument('--save_interval',  type=int,       help='save model after `n` weight update steps', default=1000)
    parser.add_argument('--val_size',       type=int,       help='validation set size for evaluating accuracy', default=2000)
    parser.add_argument('--use_val',        type=str2bool,  help='use validation set & metrics', default='true')

    # GPU params
    parser.add_argument('--gpu_id',         type=int,       help='cuda:gpu_id (0,1,2,..) if num_gpus = 1', default=0)
    parser.add_argument('--opt_lvl',        type=int,       help='Automatic-Mixed Precision: opt-level (O_)', default=1, choices=[0, 1, 2, 3])
    # parser.add_argument('--num_gpus',    type=int,   help='number of GPUs to use for training', default=1)

    # Misc params
    parser.add_argument('--num_workers',    type=int,       help='number of worker threads for Dataloader', default=1)

    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print('Selected Device: {}'.format(device))

    # Set CUDA device
    torch.cuda.set_device(device)
    # torch.cuda.get_device_properties(device).total_memory  # in Bytes

    # Train params
    n_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    # Train
    if args.mode == 'train':
        # Setup train log directory
        log_dir = os.path.join(args.expt_dir, args.expt_name, args.run_name)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print('Training Log Directory: {}\n'.format(log_dir))

        # TensorBoard summaries setup  -->  /expt_dir/expt_name/run_name/
        writer = SummaryWriter(log_dir)

        # Train log file
        log_file = setup_logs_file(parser, log_dir)

        # Dataset & Dataloader
        dataset_configs, Dataset = init_dataset_configs(args.model, args)

        train_dataset = Dataset(os.path.join(args.data_dir, 'train.json'),
                                os.path.join(args.data_dir, 'train.npy'),
                                **dataset_configs)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

        log_msg = 'Train Data Size: {}\n'.format(train_dataset.__len__())
        print_and_log(log_msg, log_file)

        # Configs inferred from dataset
        input_dim = train_dataset.embedding_dim
        max_video_len = train_dataset.max_video_len

        if args.use_val:
            # Use the same max video length as in the training dataset
            dataset_configs['max_video_len'] = max_video_len

            val_dataset = Dataset(os.path.join(args.data_dir, 'val.json'),
                                  os.path.join(args.data_dir, 'val.npy'),
                                  **dataset_configs)

            val_loader = DataLoader(val_dataset, batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

            # Total validation set size
            val_total_size = val_dataset.__len__()
            log_msg = 'Validation Data Size: {}\n'.format(val_total_size)

            # Min of the total & subset size
            val_size = min(val_total_size, args.val_size)
            log_msg += 'Validation Accuracy is computed using {} samples. See --val_size\n'.format(val_size)

            print_and_log(log_msg, log_file)

        # Build Model
        model_configs, Model = init_model_configs(args.model, args, input_dim, max_video_len)

        model = Model(model_configs, device)
        model.to(device)

        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr)

        model, optimizer = amp.initialize(model, optimizer, opt_level="O{}".format(args.opt_lvl))

        # Step & Epoch
        start_epoch = 1
        curr_step = 1
        # Load model checkpoint file (if specified) from `log_dir`
        if args.model_ckpt:
            model_ckpt_path = os.path.join(log_dir, args.model_ckpt)
            checkpoint = torch.load(model_ckpt_path)

            # Load model & optimizer
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load other info
            curr_step = checkpoint['curr_step']
            start_epoch = checkpoint['epoch']
            prev_loss = checkpoint['loss']

            log_msg = 'Resuming Training...\n'
            log_msg += 'Model successfully loaded from {}\n'.format(model_ckpt_path)
            log_msg += 'Training loss: {:2f} (from ckpt)\n'.format(prev_loss)

            print_and_log(log_msg, log_file)

        steps_per_epoch = len(train_loader)
        start_time = time()

        for epoch in range(start_epoch, start_epoch + n_epochs):
            for batch_data in train_loader:
                # Load to device, for the list of batch tensors
                batch_data = [d.to(device) for d in batch_data]
                inputs, label = batch_data[:-1], batch_data[-1]

                # Forward Pass
                label_logits = model(*inputs)

                # Compute Loss
                loss = criterion(label_logits, label)

                # Backward Pass
                optimizer.zero_grad()
                # loss.backward()
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

                optimizer.step()

                # Print Results - Loss value & Validation Accuracy
                if curr_step % args.log_interval == 0 or curr_step == 1:
                    # Validation set accuracy
                    if args.use_val:
                        validation_metrics = compute_validation_metrics(model, val_loader, device, val_size)

                        # Reset the mode to training
                        model.train()

                        log_msg = 'Validation Accuracy: {:.2f} %  || Validation Loss: {:.4f}'.format(
                                validation_metrics['accuracy'], validation_metrics['loss'])

                        print_and_log(log_msg, log_file)

                        # Add summaries to TensorBoard
                        writer.add_scalar('Val/Accuracy', validation_metrics['accuracy'], curr_step)
                        writer.add_scalar('Val/Loss', validation_metrics['loss'], curr_step)

                    # Add summaries to TensorBoard
                    writer.add_scalar('Train/Loss', loss.item(), curr_step)

                    # Compute elapsed & remaining time for training to complete
                    time_elapsed = (time() - start_time) / 3600
                    # total time = time_per_step * steps_per_epoch * total_epochs
                    total_time = (time_elapsed / curr_step) * steps_per_epoch * n_epochs
                    time_left = total_time - time_elapsed

                    log_msg = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} | time elapsed: {:.2f}h | time left: {:.2f}h'.format(
                            epoch, n_epochs, curr_step, steps_per_epoch, loss.item(), time_elapsed, time_left)

                    print_and_log(log_msg, log_file)

                # Save the model
                if curr_step % args.save_interval == 0:
                    save_path = os.path.join(log_dir, 'model_' + str(curr_step) + '.pth')

                    state_dict = {'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'curr_step': curr_step, 'loss': loss, 'epoch': epoch}

                    torch.save(state_dict, save_path)
                    # torch.save(model.state_dict(), save_path)

                    log_msg = 'Saving the model at the {} step to directory:{}'.format(curr_step, log_dir)
                    print_and_log(log_msg, log_file)

                curr_step += 1

            # Validation set accuracy on the entire set
            if args.use_val:
                # Total validation set size
                total_validation_size = val_dataset.__len__()
                validation_metrics = compute_validation_metrics(model, val_loader, device, total_validation_size)

                log_msg = '\nAfter {} epoch:\n'.format(epoch)
                log_msg += 'Validation Accuracy: {:.2f} %  || Validation Loss: {:.4f}\n'.format(
                    validation_metrics['accuracy'], validation_metrics['loss'])

                print_and_log(log_msg, log_file)

                # Reset the mode to training
                model.train()

        writer.close()
        log_file.close()

    # TODO: Test/Inference
    elif args.mode == 'test':
        pass


def init_dataset_configs(model_name, args):
    """
    Given model, sets up the dataset config & class.

    :param model_name: e.g. bilstm, bert, etc.
    :return: config dict & dataset class
    """
    config = {'max_video_len': None}    # inferred from train_dataset

    # If transformer, insert additional configs
    if 'bert' in model_name:
        config['model_name'] = args.model
        config['tok_config'] = args.config_name

    # Setup the Dataset class
    Dataset = TransformerDataset if 'bert' in model_name else BiLSTMDataset

    return config, Dataset


def init_model_configs(model_name, args, input_dim, max_video_len):
    """
    Given model, sets up the model config & class.

    :param model_name: e.g. bilstm, bert, etc.
    :return: config dict & model class
    """

    if model_name == 'bilstm':
        config = {'input_dim': input_dim,
                  'num_layers': args.num_layers,
                  'lstm_hidden': 1024,
                  'lstm_dropout': 0.1,
                  'fc_dim': 1024,
                  'num_classes': args.num_cls}
        Model = BiLSTM

    else:
        config = {'input_dim': input_dim,
                  'model_name': model_name,
                  'config_name': args.config_name,
                  'config_dict': dict(num_hidden_layers=args.num_layers),
                  'use_pretrained': args.use_pretrained,
                  'max_video_len': max_video_len,
                  'fc_dim': 1024,
                  'num_classes': args.num_cls}
        Model = Transformer

    return config, Model


if __name__ == '__main__':
    main()
