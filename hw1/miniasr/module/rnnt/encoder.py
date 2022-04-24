import torch
import torch.nn as nn
from conformer.encoder import ConformerEncoder

class BaseEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout=0.2, bidirectional=True, module='GRU'):
        super(BaseEncoder, self).__init__()

        self.lstm = getattr(nn, module)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.output_proj = nn.Linear(2 * hidden_size if bidirectional else hidden_size,
                                     output_size,
                                     bias=True)

    def forward(self, inputs, input_lengths):
        assert inputs.dim() == 3

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs = inputs[indices]
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths.cpu(), batch_first=True)

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(inputs)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        logits = self.output_proj(outputs)

        return logits, hidden


class ConfEnc(nn.Module):
    def __init__(self, input_dim, encoder_dim, num_layers):
        super(ConfEnc, self).__init__()
        self.lstm = ConformerEncoder(input_dim=input_dim, encoder_dim=encoder_dim, num_layers=num_layers, num_attention_heads=4)

    def forward(self, inputs, input_lengths):
        out, outlen = self.lstm(inputs, input_lengths)
        return out, outlen


def build_encoder(in_dim, config):
    if(config.encoder.module in ['GRU', 'LSTM']):
        return BaseEncoder(
            input_size=in_dim,
            hidden_size=config.encoder.hidden_size,
            output_size=config.encoder.output_size,
            n_layers=config.encoder.n_layers,
            dropout=config.dropout,
            bidirectional=config.encoder.bidirectional,
            module=config.encoder.module
        )
    else:
        return ConfEnc(input_dim=in_dim, encoder_dim=config.encoder.output_size, num_layers=config.encoder.n_layers)
    
