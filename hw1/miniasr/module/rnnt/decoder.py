import torch
import torch.nn as nn

from miniasr import module


class BaseDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, output_size, n_layers, dropout=0.2, share_weight=False, module='LSTM'):
        super(BaseDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.lstm = getattr(nn, module)(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.output_proj = nn.Linear(hidden_size, output_size)

        if share_weight:
            self.embedding.weight = self.output_proj.weight

    def forward(self, inputs, length=None, hidden=None):

        embed_inputs = self.embedding(inputs)

        if length is not None:
            sorted_seq_lengths, indices = torch.sort(length, descending=True)
            embed_inputs = embed_inputs[indices]
            embed_inputs = nn.utils.rnn.pack_padded_sequence(
                embed_inputs, sorted_seq_lengths.cpu(), batch_first=True)

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(embed_inputs, hidden)

        if length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        outputs = self.output_proj(outputs)

        return outputs, hidden

class TransDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, output_size, n_layers, dropout=0.2, n_head=4):
        super(TransDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.declayer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=n_head, dim_feedforward=output_size, dropout=dropout)
        self.lstm = nn.TransformerDecoder(self.declayer, n_layers)
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self.nhead = n_head
        self.pe = PositionalEncoding(hidden_size)
    def padding_mask(self, seq, pad_idx):
        return (seq == pad_idx)  # [B, 1, L]

    def sequence_mask(self, seq):
        batch_size, seq_len = seq.size()
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, inputs, enc_len=None, length=None, gt=None, pidx=0):
        pm = self.padding_mask(gt, pidx)
        sm = self.sequence_mask(gt).cuda()
        # print('decode', pm.shape, sm.shape)
        # sm = (sm & pm.unsqueeze(-2).expand())
        embed_inputs = self.embedding(gt) + self.pe(gt.shape[-1])
        # print('decode 69', gt, inputs.shape)
        '''if length is not None:
            sorted_seq_lengths, indices = torch.sort(length, descending=True)
            # print('decode', indices.shape, length.shape, sorted_seq_lengths.shape)
            embed_inputs = embed_inputs[indices]'''
            

        # print('decode', pm.shape)
        # print('dec80', embed_inputs[0], inputs[0], gt, embed_inputs.shape, gt.shape, inputs.shape, enc_len, sm, pm)
        outputs = self.lstm(embed_inputs.transpose(0, 1), inputs.transpose(0, 1), 
                            tgt_mask=(sm if length != None else None), tgt_key_padding_mask=(pm if length != None else None))
        # print('dec82', outputs[0])
        '''
        if length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]
        '''

        outputs = self.output_proj(outputs)

        return outputs.transpose(0, 1)
    
    
def build_decoder(vocab_size, config):
    if config.name != 'las':
        return BaseDecoder(
            hidden_size=config.decoder.hidden_size,
            vocab_size=vocab_size,
            output_size=config.decoder.output_size,
            n_layers=config.decoder.n_layers,
            dropout=config.dropout,
            share_weight=config.share_weight,
            module=config.decoder.module
        )
    else:
        return TransDecoder(
            hidden_size=config.decoder.hidden_size, 
            vocab_size=vocab_size, 
            output_size=config.decoder.ff_dim,
            n_layers=config.decoder.n_layers,
            dropout=config.dropout,
            n_head=config.decoder.n_head,
        )

import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 1000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int):
        return self.pe[:, :length]