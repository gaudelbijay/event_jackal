import numpy as np
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    this is the encoding module
    """
    def __init__(self,
                 input_dim,
                 num_layers=2,
                 hidden_size=512,
                 history_length=1,
                 concat_action=False,
                 dropout=0.0):
        """
        state_dim: the state dimension
        stacked_frames: #timesteps considered in history
        hidden_size: hidden layer size
        num_layers: how many layers

        the input state should be of size [batch, stacked_frames, state_dim]
        the output should be of size [batch, hidden_size]
        """
        super().__init__()
        self.hidden_size = self.feature_dim = hidden_size
    
    def forward(self, states, actions=None):
        return None


class MLPEncoder(Encoder):
    def __init__(self,
                 input_dim,
                 num_layers=2,
                 hidden_size=512,
                 history_length=1,
                 concat_action=False,
                 dropout=0.0):
        super().__init__(input_dim=input_dim,
                         num_layers=num_layers,
                         hidden_size=hidden_size,
                         history_length=history_length,
                         concat_action=concat_action,
                         dropout=dropout)

        layers = []
        for i in range(num_layers):
            input_dim = hidden_size if i > 0 else input_dim * history_length
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)  # mlp will flatten time sequence
        return self.net(x)


class CNNEncoder(Encoder):
    def __init__(self,
                 input_dim,
                 num_layers=2,
                 hidden_size=512,
                 history_length=1,
                 concat_action=False,
                 dropout=0.0):
        super().__init__(input_dim=input_dim,
                         num_layers=num_layers,
                         hidden_size=hidden_size,
                         history_length=history_length,
                         concat_action=concat_action,
                         dropout=dropout)

        layers = []
        if num_layers > 1:
            for i in range(num_layers-1):
                input_channel = hidden_size if i > 0 else input_dim
                layers.append(nn.Conv1d(in_channels=input_channel,
                                        out_channels=hidden_size,
                                        kernel_size=3, padding=1))
                layers.append(nn.ReLU())

            layers.extend([
                nn.Conv1d(in_channels=hidden_size,
                          out_channels=hidden_size,
                          kernel_size=history_length, padding=0),
                nn.ReLU()])
        else:
            layers.extend([
                nn.Conv1d(in_channels=input_dim,
                          out_channels=hidden_size,
                          kernel_size=history_length, padding=0),
                nn.ReLU()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x [batch, seq_len, state_dim]
        x = x.permute(0, 2, 1) # [batch, state_dim, seq_len]
        x = self.net(x) # [batch, hidden_dim, 1]
        return x.squeeze(-1)


class RNNEncoder(Encoder):
    def __init__(self,
                 input_dim,
                 num_layers=2,
                 hidden_size=512,
                 history_length=1,
                 concat_action=False,
                 dropout=0.0):
        super().__init__(input_dim=input_dim,
                         num_layers=num_layers,
                         hidden_size=hidden_size,
                         history_length=history_length,
                         concat_action=concat_action,
                         dropout=dropout)

        self.num_layers = num_layers
        self.net = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

    def forward(self, x):
        """
        always start with h0 = 0
        """
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        output, hn = self.net(x, h0)
        y = output[:,-1]
        return y


class TransformerEncoder(Encoder):
    def __init__(self,
                 input_dim,
                 num_layers=2,
                 hidden_size=512,
                 history_length=1,
                 concat_action=False,
                 dropout=0.0):
        super().__init__(input_dim=input_dim,
                         num_layers=num_layers,
                         hidden_size=hidden_size,
                         history_length=history_length,
                         concat_action=concat_action,
                         dropout=dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=num_layers
        )
        self.embed_state = nn.Linear(input_dim, hidden_size)
        self.embed_timestep = nn.Embedding(history_length, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

    def forward(self, states, actions=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        state_embeddings = self.embed_state(states)
        timesteps = torch.arange(seq_length, device=states.device).unsqueeze(0).expand(batch_size, seq_length)
        time_embeddings = self.embed_timestep(timesteps)
        inputs = state_embeddings + time_embeddings
        inputs = self.embed_ln(inputs)
        transformer_output = self.transformer(inputs)
        x = transformer_output[:, -1, :]
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, num_layers=2, hidden_layer_size=512):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = hidden_layer_size

        layers = []
        for i in range(num_layers):
            input_dim = hidden_layer_size if i > 0 else self.input_dim
            layers.append(nn.Linear(input_dim, hidden_layer_size))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)