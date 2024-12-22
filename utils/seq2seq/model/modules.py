import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from utils.seq2seq.model.layers import FFTBlock
import utils.seq2seq.Constants as Constants
from utils.seq2seq.tools import get_mask_from_lengths
from utils.seq2seq.model.optimizer import ScheduledOptim

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)

def get_model(args, config, device, train=False):

    model = StethoSpeech(config).to(device)
    print(f"Initialized StethoSpeech Model: {model}")
    if args.restore_step:
        ckpt_path = os.path.join(
            config["path"]["root_path"],"ckpt",
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f"Loaded StethoSpeech checkpoint from {ckpt_path}")
        model.load_state_dict(ckpt["model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

class StethoSpeech(nn.Module):
    """ StethoSpeech """

    def __init__(self, config):
        super(StethoSpeech, self).__init__()
        self.hubert_dim = config["preprocessing"]["hubert_dim"]
        self._vocab_size = config["preprocessing"]["ctc_vocab_size"]

        # Encoder
        self.enclin = nn.Linear(self.hubert_dim,config["transformer"]["encoder_hidden"])
        self.encoder = Encoder(config)

        # CTC 
        self.fc1 = nn.Linear(config["transformer"]["decoder_hidden"], self.hubert_dim)
        self.lm_head = nn.Linear(self.hubert_dim, self._vocab_size) # hidden_size,vocab_size

        # Decoder
        self.decoder = Decoder(config)
        self.fc2 = nn.Linear(config["transformer"]["decoder_hidden"], self.hubert_dim)

        self.infer = False
        self.dropout = nn.Dropout(config["transformer"]["decoder_dropout"])

    def forward(
        self,
        texts,
        src_lens,
        max_src_len,
        huberts = None,
        huberts_lens = None,
        max_trg_len = None,
        ctc_labels = None,
        ctc_labels_lens = None,
        max_ctc_labels_lens = None,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        trg_masks = (
            get_mask_from_lengths(huberts_lens, max_trg_len)
            if huberts_lens is not None
            else None
        )
        
        texts = self.enclin(texts)
        texts = F.relu(texts)
        encoder_out = self.encoder(texts, src_masks)

        mel_masks = src_masks
        output = encoder_out
            
        output_forctc = self.fc1(output) 

        # for CTC  
        if self.infer == False:
            hidden_states = output_forctc  #[0]
            hidden_states = self.dropout(hidden_states)

            # each element is True if the corresponding element in ctc_labels is greater than zero
            labels_mask = ctc_labels > 0  #>= 0
            target_lengths = labels_mask.sum(-1)

            # masked_select selects the elements from ctc_labels where labels_mask is True
            # a 1-dimensional tensor containing the non-padding target labels.
            flattened_targets = ctc_labels.masked_select(labels_mask)
            
            # map the intermediate hidden states to CTC logits, which will be used to calculate the CTC loss.
            ctc_logits = self.lm_head(hidden_states)

            # converts the logits into log probabilities.
            log_probs = nn.functional.log_softmax(ctc_logits, dim=-1, dtype=torch.float32)

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.fc2(output) 

        if self.infer == True:
            output = output.squeeze(0)
            return output

        return (
                output,
                src_masks,
                trg_masks,
                src_lens,
                target_lengths,
                flattened_targets,
                log_probs
            )
    
    def test(self):
        self.eval()
        self.infer = True
        
class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Encoder, self).__init__()

        n_position = config["transformer"]["max_seq_len"] + 1
        n_src_vocab = 100+1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["transformer"]["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            # src_seq = self.src_word_emb(src_seq)
            enc_output = src_seq + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["transformer"]["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["transformer"]["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask, return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask