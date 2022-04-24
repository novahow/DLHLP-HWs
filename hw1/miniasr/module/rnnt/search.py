import torch
import torch.nn.functional as F


def GreedyDecode(model, inputs, input_lengths):

    assert inputs.dim() == 3
    # f = [batch_size, time_step, feature_dim]
    f, _ = model.encoder(inputs, input_lengths)

    zero_token = torch.LongTensor([[model.tokenizer.sos_idx]]).cuda()
    results = []
    batch_size = inputs.size(0)
    # print(model.tokenizer.null_idx, model.tokenizer.sos_idx, model.tokenizer.vocab_size)
    def decode(inputs, lengths):
        pred = torch.LongTensor(model.tokenizer.sos_idx)
        token_list = []
        hidden = None
        token = zero_token
        dec_state, hidden = model.decoder(zero_token)
        t = 0
        while t < inputs.shape[0] and len(token_list) < inputs.shape[0]:
            logits = model.joint(inputs[t].view(-1), dec_state.view(-1))
            out = F.softmax(logits, dim=0).detach()
            pred = torch.argmax(out, dim=0)
            top = int(pred.item())
            if top != model.tokenizer.null_idx:
                token_list.append(top)
                token = token.new_tensor([[pred]], dtype=torch.long)
                dec_state, hidden = model.decoder(token, hidden=hidden)
                
            else:
                t += 1
            
            if top == model.tokenizer.eos_idx:
                break
        
        '''
        for t in range(inputs.shape[0]):
            logits = model.joint(inputs[t].view(-1), dec_state.view(-1))
            out = F.softmax(logits, dim=0).detach()
            pred = torch.argmax(out, dim=0)
            top = int(pred.item())
            # print('search', top)
            if top != model.tokenizer.null_idx:
                token_list.append(top)
                token = token.new_tensor([[pred]], dtype=torch.long)
                dec_state, hidden = model.decoder(token, hidden=hidden)
        '''

        return token_list
            
    for i in range(batch_size):
        decoded_seq = decode(f[i], input_lengths[i])
        results.append(decoded_seq)

    return results




@torch.no_grad()
def sooftdecode(model, inputs, input_lengths):
    """
    Recognize input speech. This method consists of the forward of the encoder and the decode() of the decoder.
    Args:
        inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
            `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
    Returns:
        * predictions (torch.FloatTensor): Result of model predictions.
    """
    outputs = list()

    encoder_outputs, output_lengths = model.encoder(inputs, input_lengths)
    max_length = encoder_outputs.size(1)
    def decode(encoder_output, max_length):
        """
        Decode `encoder_outputs`.
        Args:
            encoder_output (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(seq_length, dimension)``
            max_length (int): max decoding time step
        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        pred_tokens, hidden_state = list(), None
        decoder_input = encoder_output.new_tensor([[model.tokenizer.sos_idx]], dtype=torch.long)

        for t in range(max_length):
            decoder_output, hidden_state = model.decoder(decoder_input, hidden=hidden_state)
            step_output = model.joint(encoder_output[t].view(-1), decoder_output.view(-1))
            step_output = step_output.softmax(dim=0)
            pred_token = step_output.argmax(dim=0)
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)
            decoder_input = step_output.new_tensor([[pred_token]], dtype=torch.long)

        return torch.LongTensor(pred_tokens)
    for encoder_output in encoder_outputs:
        decoded_seq = decode(encoder_output, max_length)
        outputs.append(decoded_seq)

    outputs = torch.stack(outputs, dim=1).transpose(0, 1)

    return outputs


def lasdecode(model, inputs, input_lengths):
    encoder_outputs, output_lengths = model.encoder(inputs, input_lengths)
    max_length = encoder_outputs.size(1)
    def decode(encoder_output, max_length):
        """
        Decode `encoder_outputs`.
        Args:
            encoder_output (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(seq_length, dimension)``
            max_length (int): max decoding time step
        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        pred_tokens, hidden_state = list(), None
        decoder_input = torch.LongTensor([[model.tokenizer.sos_idx]]).cuda()


        for t in range(max_length):
            step_output = model.decoder(decoder_input, encoder_output[t])
            step_output = step_output.softmax(dim=0)
            pred_token = step_output.argmax(dim=0)
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)
            decoder_input = step_output.new_tensor([[pred_token]], dtype=torch.long)

        return torch.LongTensor(pred_tokens)
    
    for encoder_output in encoder_outputs:
        decoded_seq = decode(encoder_output, max_length)
        outputs.append(decoded_seq)

    outputs = torch.stack(outputs, dim=1).transpose(0, 1)

    return outputs


def recognize(model, inputs, inputs_length):
    
    batch_size = inputs.size(0)

    enc_states, _el = model.encoder(inputs, inputs_length)

    zero_token = torch.LongTensor([[model.tokenizer.sos_idx]])
    if inputs.is_cuda:
        zero_token = zero_token.cuda()

    def decode(enc_state, lengths):
        token_list = []
        token = zero_token
        dec_state, hidden = model.decoder(token)

        # print(lengths)
        t = 0
        while t < lengths and len(token_list) < lengths:
            logits = model.joint(enc_state[t].view(-1), dec_state.view(-1))
            pred = torch.argmax(logits, dim=0)
            top = int(pred.item())

            if top != model.tokenizer.null_idx:
                token_list.append(top)
                token = token.new_tensor([[pred]], dtype=torch.long)
                dec_state, hidden = model.decoder(token, hidden=hidden)
            else:
                t += 1
                
            if top == model.tokenizer.eos_idx:
                break
            
        return token_list

    results = []
    for i in range(batch_size):
        decoded_seq = decode(enc_states[i], _el[i])
        results.append(decoded_seq)

    return results



def LasDecode(model, inputs, input_lengths):
    
    assert inputs.dim() == 3
    # f = [batch_size, time_step, feature_dim]
    f, el = model.encoder(inputs, input_lengths)
    zero_token = torch.LongTensor([[model.tokenizer.sos_idx]]).cuda()
    results = []
    batch_size = inputs.size(0)
    # print(model.tokenizer.null_idx, model.tokenizer.sos_idx, model.tokenizer.vocab_size)
    def decode(inputs, lengths):
        # print('search', inputs.shape, lengths)
        pred = torch.LongTensor(model.tokenizer.sos_idx)
        token_list = []
        hidden = None
        token = zero_token
        enc_len = torch.LongTensor(1).cuda()
        enc_state = inputs.unsqueeze(0)

        t = 0
        while t < lengths and len(token_list) < lengths:
            logits = model.decoder(enc_state, enc_len=enc_len, gt=token)
            out = F.softmax(logits, dim=-1).detach()
            pred = torch.argmax(out, dim=-1)
            # print('search', token)
            top = int(pred[0][-1].item())
            token_list.append(top)
            pred = pred.new_tensor([[top]], dtype=torch.long)
            token = torch.cat([token, pred], dim=-1)
            t += 1
            
            if top == model.tokenizer.eos_idx:
                break
        return token_list
            
    for i in range(batch_size):
        decoded_seq = decode(f[i], el[i])
        results.append(decoded_seq)

    return results