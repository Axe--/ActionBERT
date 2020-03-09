"""
Action Recognition Models
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from configs import load_model, load_embedding_fn


class BiLSTM(nn.Module):
    def __init__(self, model_params, device=None):
        super().__init__()

        input_dim = model_params['input_dim']
        hidden_dim = model_params['lstm_hidden']
        num_layers = model_params['num_layers']
        lstm_dropout = model_params['lstm_dropout']
        fc_dim = model_params['fc_dim']
        num_cls = model_params['num_classes']

        # BiLSTM
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                              dropout=lstm_dropout, bidirectional=True)

        # Logit layer
        self.fc = nn.Sequential(nn.Linear(2 * hidden_dim, fc_dim),
                                nn.Dropout(0.5),
                                nn.Tanh(),
                                nn.Linear(fc_dim, num_cls))
        # Params
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x_input, x_seq_len):
        # x_input: [batch_size, seq_len, input_dim]

        x = pack_padded_sequence(x_input, x_seq_len, batch_first=True, enforce_sorted=False)

        outputs, (hidden, cell) = self.bilstm(x)        # outputs: [sum_{i=0}^batch (seq_lens[i]), 2 * hidden_dim]

        # hidden: [num_layers * 2, batch_size, hidden_dim]
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)   # [num_layers, 2, batch_size, hidden_dim]

        # Skip hidden states of intermediate layers
        hidden = hidden[-1]                                             # [2, batch_size, hidden_dim]

        # Extract the forward & backward hidden states
        forward_h = hidden[0]
        backward_h = hidden[1]

        # Concatenate hidden states
        final_hidden = torch.cat([forward_h, backward_h], dim=1)       # [batch_size, 2*hidden_dim]

        logits = self.fc(final_hidden)                                 # [batch_size, num_cls]

        return logits


# ---------------------------------------------------------------
class Transformer(nn.Module):
    """
    Adapts HuggingFace's Transformer for handling video embeddings
    """
    def __init__(self, model_params, device=None):
        super(Transformer, self).__init__()

        input_dim = model_params['input_dim']
        model_name = model_params['model_name']         # e.g. bert, roberta, etc.
        config_name = model_params['config_name']       # e.g. bert-base-uncased, roberta-base, etc.
        config_dict = model_params['config_dict']       # custom config params
        use_pretrained = model_params['use_pretrained']
        fc_dim = model_params['fc_dim']
        num_cls = model_params['num_classes']

        self.max_len = model_params['max_video_len']

        self.device = device

        # Load transformer for the given name & config
        self.transformer = load_model(model_name, config_dict, config_name, use_pretrained)

        hidden_dim = self.get_hidden_size()

        # Project video embedding to transformer dim
        self.projection_layer = nn.Linear(input_dim, hidden_dim)

        # Load the embedding function for encoding token ids
        self.embedding_fn = load_embedding_fn(model_name, config_name)

        # Logit layer
        self.fc = nn.Sequential(nn.Linear(hidden_dim, fc_dim),
                                nn.Dropout(0.5),
                                nn.Tanh(),
                                nn.Linear(fc_dim, num_cls))

    def forward(self, video_emb, token_seq_ids, attention_mask):
        """
        # max_seq_len = max_video_len + num_special_tokens

        :param video_emb: [batch, max_video_len, video_emb_dim]
        :param token_seq_ids: [batch, max_seq_len]
        :param attention_mask: [batch, max_seq_len] <br>
        """
        # Project video embedding to token embedding space (hidden dim)
        video_emb = self.projection_layer(video_emb)

        # Encode video with positional embeddings
        video_emb = self.embedding_fn(inputs_embeds=video_emb,
                                      position_ids=torch.arange(1, self.max_len + 1, device=self.device))

        # Encode token sequence ([CLS] [UNK].. [SEP] [PAD]..)
        embeddings_input = self.embedding_fn(input_ids=token_seq_ids)

        # Replace [UNK] embeddings with video embeddings
        embeddings_input[:, 1: self.max_len+1, :] = video_emb

        # Extract the sequence embeddings from the final layer of the transformer
        last_hidden_states = self.transformer(inputs_embeds=embeddings_input,           # [batch, max_len, emb_dim]
                                              attention_mask=attention_mask)[0]

        # Obtain the CLS token embedding from the last hidden layer
        cls_output = last_hidden_states[:, 0, :]                                        # [batch, emb_dim]

        logits = self.fc(cls_output)

        return logits

    def get_hidden_size(self):
        return self.transformer.config.hidden_size
