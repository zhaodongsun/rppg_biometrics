import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class classification_head(nn.Module):
    def __init__(self, num_classes=100):
        super(classification_head, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ppg_transformer(nn.Module):
    def __init__(self, num_classes=100, d_model=64, nhead=4, num_layers=2, dim_feedforward=1024):
        super(ppg_transformer, self).__init__()

        self.conv1 = nn.Conv1d(1, int(d_model/4), kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(int(d_model/4), int(d_model/2), kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(int(d_model/2), d_model, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Create the Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Create the output classifier
        self.fc_input_dim = 5*d_model
        self.fc1 = nn.Linear(self.fc_input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        

    def forward(self, x):

        x = self.pool3(torch.tanh(self.conv1(x)))
        x = self.pool3(torch.tanh(self.conv2(x)))
        x = self.pool2(torch.tanh(self.conv3(x)))
        # Transpose the input for the Transformer encoder
        x = x.permute((2, 0, 1))

        # Add positional encoding to the input
        x = self.pos_encoder(x)

        # Apply the Transformer encoder
        x = self.transformer_encoder(x)

        x = x.permute((1,2,0))
        x = x.reshape(-1, self.fc_input_dim) ## TODO

        # Apply the output classifier
        ppg_embed = self.fc1(x)
        cls_output = self.fc2(ppg_embed)
        return cls_output, ppg_embed


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


