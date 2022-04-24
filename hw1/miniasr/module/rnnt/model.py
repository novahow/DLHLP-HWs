import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import build_encoder
from .decoder import build_decoder
from warprnnt_pytorch import RNNTLoss


class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()

        self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)

        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size, bias=True)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        outputs = self.forward_layer(concat_state)

        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)

        return outputs


class Transducer(nn.Module):
    def __init__(self, in_dim, vocab_size, config, tokenizer):
        super(Transducer, self).__init__()
        # define encoder
        self.config = config
        self.encoder = build_encoder(in_dim, config)
        # define decoder
        self.decoder = build_decoder(tokenizer.vocab_size, config)
        # define JointNet
        self.joint = JointNet(
            input_size=config.encoder.output_size + config.decoder.output_size,
            inner_dim=config.joint.inner_size,
            vocab_size=tokenizer.vocab_size
        )
        self.vocab_size = tokenizer.vocab_size
        '''
        if config.share_embedding:
            assert self.decoder.embedding.weight.size() == self.joint.project_layer.weight.size(), '%d != %d' % (self.decoder.embedding.weight.size(1),  self.joint.project_layer.weight.size(1))
            self.joint.project_layer.weight = self.decoder.embedding.weight
        '''
        self.crit = RNNTLoss()
        self.fbankdim = 240
        self.kersize = 3
        self.stride = 2
        self.tokenizer = tokenizer
        self.downsample = nn.Conv1d(self.fbankdim, self.fbankdim, kernel_size=self.kersize, stride=self.stride)
        
    def forward(self, inputs, inputs_length, targets, targets_length):
        '''inputs_length = ((inputs_length + 2 * self.downsample.padding[0] - 1
                             - self.downsample.dilation[0] * 
                                    (self.kersize - 1)) / self.stride + 1).int()
        
        inputs = self.downsample(inputs.transpose(-1, -2)).transpose(-1, -2)'''
        enc_state, el = self.encoder(inputs, inputs_length)
        # print('model', el, enc_state.shape, inputs.shape, inputs_length)
  
        concat_targets = F.pad(targets, pad=(1, 0), value=self.tokenizer.sos_idx)
        # concat_targets = F.pad(concat_targets, pad=(0, 1), value=self.tokenizer.eos_idx)
        dec_state, dh = self.decoder(concat_targets, targets_length + 1)
        # if self.config.encoder.module not in ['GRU', 'LSTM']:
            # enc_state = torch.transpose(enc_state, -1, -2)
        logits = self.joint(enc_state, dec_state)

        # loss = self.crit(logits, targets.int(), inputs_length.int(), targets_length.int())
        
        return logits, (inputs_length if self.config.encoder.module in ['GRU', 'LSTM'] \
                else el + enc_state.shape[-2] - max(el))

    def recognize(self, inputs, inputs_length):

        batch_size = inputs.size(0)

        enc_states, _ = self.encoder(inputs, inputs_length)

        zero_token = torch.LongTensor([[3]])
        if inputs.is_cuda:
            zero_token = zero_token.cuda()

        def decode(enc_state, lengths):
            token_list = []

            dec_state, hidden = self.decoder(zero_token)

            for t in range(lengths):
                logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
                out = F.softmax(logits, dim=0).detach()
                pred = torch.argmax(out, dim=0)
                pred = int(pred.item())

                if pred != 0:
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])

                    if enc_state.is_cuda:
                        token = token.cuda()

                    dec_state, hidden = self.decoder(token, hidden=hidden)

            return token_list

        results = []
        for i in range(batch_size):
            decoded_seq = decode(enc_states[i], inputs_length[i])
            results.append(decoded_seq)

        return results


class LAS(nn.Module):
    def __init__(self, in_dim, vocab_size, config, tokenizer):
        super(LAS, self).__init__()
        # define encoder
        self.config = config
        self.encoder = build_encoder(in_dim, config)
        # define decoder
        self.decoder = build_decoder(tokenizer.vocab_size, config)
        # define JointNet
        
        self.vocab_size = tokenizer.vocab_size
        '''
        if config.share_embedding:
            assert self.decoder.embedding.weight.size() == self.joint.project_layer.weight.size(), '%d != %d' % (self.decoder.embedding.weight.size(1),  self.joint.project_layer.weight.size(1))
            self.joint.project_layer.weight = self.decoder.embedding.weight
        '''
        self.fbankdim = 240
        self.kersize = 3
        self.stride = 2
        self.tokenizer = tokenizer
        self.downsample = nn.Conv1d(self.fbankdim, self.fbankdim, kernel_size=self.kersize, stride=self.stride)
        
    def forward(self, inputs, inputs_length, targets, targets_length):
        enc_state, el = self.encoder(inputs, inputs_length)
        # print('model', el, enc_state.shape, inputs.shape, inputs_length)
  
        concat_targets = F.pad(targets, pad=(1, 0, 0, 0), value=self.tokenizer.sos_idx)

        dec_state = self.decoder(enc_state, enc_len=el, length=targets_length, gt=concat_targets, pidx=self.tokenizer.pad_idx)
        # if self.config.encoder.module not in ['GRU', 'LSTM']:
            # enc_state = torch.transpose(enc_state, -1, -2)
        logits = dec_state
        # loss = self.crit(logits, targets.int(), inputs_length.int(), targets_length.int())
        # print('model', logits.shape)
        return logits, (inputs_length if self.config.encoder.module in ['GRU', 'LSTM'] \
                else el + enc_state.shape[-2] - max(el))
