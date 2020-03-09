"""
Reads processed dataset json & numpy files.

JSON format:
{
    'delete':
        [{'video_idx', 'video_name', 'video_length', 'label_idx'}],

    'memmap_size': tuple[int, int, int]     # (total_videos, max_video_len, emb_dim)
}

Numpy array format:
- shape = [total_videos, max_video_len, emb_dim]
"""
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from configs import load_tokenizer, load_embedding_fn


class ConvEmbeddingDataset(Dataset):
    """
    Loads pre-computed embeddings from ConvNet
    """

    def __init__(self, json_file, embedding_file, max_video_len=None):
        # Parse JSON
        json_data = self._read_json(json_file)

        # DataFrame
        self.data_df = pd.read_json(json_data['delete'])

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
        pass

    @staticmethod
    def _read_json(json_file):
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        return json_data


class BiLSTMDataset(ConvEmbeddingDataset):

    def __init__(self, json_file, embedding_file, max_video_len=None):
        super().__init__(json_file, embedding_file, max_video_len)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx])
        video_len = self.video_lengths[idx]
        label = self.labels[idx]

        return embedding, video_len, label


class TransformerDataset(ConvEmbeddingDataset):
    """
    Prepares input in the following format:

    [CLS] Video [SEP] [PAD]
    """

    def __init__(self, json_file, embedding_file, max_video_len=None, model_name=None, tok_config=None):
        """
        :param json_file: processed dataset json
        :param embedding_file: embeddings file (npy)
        :param max_video_len: max video length (frames)
        :param str model_name: transformer model name
                                (e.g. 'bert', 'roberta', etc.)
        :param str tok_config: pre-trained tokenizer config
                                (e.g. 'bert-base-uncased', 'roberta-base', etc.)
        """
        super().__init__(json_file, embedding_file, max_video_len)

        # Load tokenizer
        self.tokenizer = load_tokenizer(model_name, tok_config)

        self.num_special_tokens = 1
        self.max_seq_len = self.max_video_len + self.num_special_tokens

    def __getitem__(self, idx):
        # Read delete
        video_emb = torch.tensor(self.embeddings[idx])
        video_len = self.video_lengths[idx]
        label = self.labels[idx]

        # Prepend CLS & Append PAD tokens; the UNK tokens serve as placeholder for video embedding
        token_ids, attention_mask = self.prepare_token_sequence(video_len)

        return video_emb, token_ids, attention_mask, label

    def prepare_token_sequence(self, video_len):
        """
        Generates token sequence in the following format:

        [CLS] [UNK] * `video len` [PAD]

        :param int video_len: actual video length
        :returns: token IDs & corresponding attention mask
        :rtype: tuple [list[int], list[int]]
        """
        # Pad tokens for video embeddings
        pad_len = self.max_video_len - video_len

        token_ids = [self.tokenizer.cls_token_id]
        token_ids += [self.tokenizer.unk_token_id] * video_len
        token_ids += [self.tokenizer.pad_token_id] * pad_len

        attention_mask = [1] * (self.max_seq_len - pad_len)
        attention_mask += [0] * pad_len

        # Convert to tensors
        token_ids = torch.tensor(token_ids)
        attention_mask = torch.tensor(attention_mask)

        return token_ids, attention_mask


if __name__ == '__main__':
    jsn = '/home/axe/Datasets/UCF_101/processed_fps_1_res18/train_fps_1_res18.json'
    npy = '/home/axe/Datasets/UCF_101/processed_fps_1_res18/train_fps_1_res18.npy'

    # dataset = BiLSTMDataset(jsn, npy)
    dataset = TransformerDataset(jsn, npy, model_name='bert', tok_config='bert-base-uncased')
    print(dataset.__len__())

    dataloader = DataLoader(dataset, batch_size=2)

    for batch in dataloader:
        # emb_tensor, v_len, label_idx = batch[:]
        print('-')
