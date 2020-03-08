"""
Action Recognition Models
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, model_params):
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
