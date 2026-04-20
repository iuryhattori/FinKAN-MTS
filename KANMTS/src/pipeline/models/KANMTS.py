import math

from layers.Embed import DataEmbedding_inverted
from layers.KANLinear import KANLinear
from layers.Transformer_EncDec import Encoder, EncoderLayer

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.signal import savgol_filter
from torch import flatten, dropout
from torch.nn.functional import gelu

# from src.kan import KAN

acv = nn.GELU()


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT
    repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper:
    https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
                0.5
                * x
                * (
                        1.0
                        + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (x + 0.044715 * torch.pow(x, 3.0))
                )
                )
        )



class TokenMixingKAN(nn.Module):
    def __init__(self, n_tokens, n_channel, n_hidden, n_hidden1=20):
        super().__init__()
        self.layer_norm = nn.LayerNorm([n_tokens, n_channel])
        self.kan1 = KANLinear(n_tokens, n_hidden)
        self.kan2 = KANLinear(n_hidden, n_tokens)
        self.activations = None

   
    def forward(self, X): 
        z = X.permute(0, 2, 1)
        z = self.layer_norm(z)  # LayerNorm normalizes along the last dimension (channel dimension), processing each channel independently. This keeps the shape of z as (32, 96, 11).
        z = z.permute(0, 2, 1) 
        z = self.kan1(z)
        z = self.kan2(z)

        U = X + z
        self.activations = U
        return U


class ChannelMixingKAN(nn.Module): 
    def __init__(self, n_tokens, n_channel, n_hidden, n_hidden1=20):
        super().__init__()
        self.layer_norm = nn.LayerNorm([n_tokens, n_channel])
        self.kan1 = KANLinear(n_channel, n_hidden)
        self.kan2 = KANLinear(n_hidden, n_channel)
        self.activations = None

    def forward(self, U): 
        z = U.permute(0, 2, 1)
        z = self.layer_norm(z) 
        z = self.kan1(z)
        z = self.kan2(z)
        z = z.permute(0, 2, 1)

        Y = U + z  # 32,11,512
        self.activations = Y
        return Y


class Mixer2dTriUKAN(nn.Module):
    def __init__(self, time_steps, channels, d_core, grid_size, hidden_dim, dropout): 
        super(Mixer2dTriUKAN, self).__init__()

        self.net1 = nn.Sequential(
            KANLinear(time_steps, time_steps, grid_size=grid_size),
            NewGELU(),
            KANLinear(time_steps, d_core, grid_size=grid_size),

        )
        self.net2 = nn.Sequential(
            KANLinear(d_core, time_steps, grid_size=grid_size),
            NewGELU(),
            KANLinear(time_steps, time_steps, grid_size=grid_size),
        )

        self.TokenMixingKAN = TokenMixingKAN(d_core, channels + 5, hidden_dim)
        self.ChannelMixingKAN = ChannelMixingKAN(d_core, channels + 5, hidden_dim)

       

    def forward(self, inputs, *args, **kwargs): 
        batch_size, channels, d_series = inputs.shape

        combined_mean = self.net1(inputs)  # Fully connected layer

        # Time dimension processing
        TokenMixing = self.TokenMixingKAN(combined_mean) 

        x = TokenMixing + combined_mean

       
        ChannelMixing = self.ChannelMixingKAN(x)

        y = ChannelMixing + x  # (32,11,512)

        z = self.net2(y)

        return z, None 

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.batch_size = configs.batch_size
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.grid_size)
        self.use_norm = configs.use_norm
        # self.stock = StockMixer(configs.batch_size, configs.d_model, market=20)

        # Encoder
        self.encoder = Encoder(
            [  # Multiple EncoderLayers
                EncoderLayer(
                    Mixer2dTriUKAN(configs.d_model, configs.enc_in, configs.d_core, configs.grid_size,
                                configs.hidden_dim,
                                configs.dropout),
                    # Models relationships between channels by fusing multi-channel sequence representations and core representations of the entire multivariate sequence.
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
        )

        # Decoder  Mapping output
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Filter data for each batch and channel

        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape

     
        enc_out = self.enc_embedding(x_enc, x_mark_enc) 

        enc_out, attns = self.encoder(enc_out, attn_mask=None)  

        # Restore to original input format
        # batch_size, d_series, channels = dec_out.shape
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N] 
        # dec_out = self.kan(enc_out).permute(0, 2, 1)[:, :, :N]  

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)  # （32,96,7）
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]-self.pred_len: Selects self.pred_len elements starting from the last element.








