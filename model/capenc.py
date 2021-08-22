# this code is developed based on https://github.com/jayleicn/TVQA

import torch
from torch import nn
from .rnn import RNNEncoder
from .bidaf2 import BidafAttn
from .gcn import GCN
from .PosEmbed import positionalencoding1d
import pickle


class TextEnc(nn.Module):
    def __init__(self, opt):
        super(TextEnc, self).__init__()
        hsize1 = opt.hsize1
        embed_size = opt.embed_size

        self.bert_fc = nn.Sequential(
            nn.Linear(opt.embed_size, hsize1*2),
            nn.Dropout(0.5),
            #nn.Linear(opt.embed_size, hsize1*2 if self.attn_fusion else embed_size),
            nn.Tanh()
        )
        self.lstm_raw_text = RNNEncoder(opt.embed_size, hsize1, bidirectional=True, dropout_p=0, n_layers=1, rnn_type='lstm')

    def forward(self, text_input):
        text_hidden, text_lens = text_input
        text_projected = self.bert_fc(text_hidden)
        # text_encoded = self.bert_fc(text_hidden)
        text_encoded, _ = self.lstm_raw_text(text_projected, text_lens) #600

        return text_encoded, text_lens


class ClassificationTask(nn.Module):
    def __init__(self, opt, textenc, videnc=None, audenc=None, objenc=None):
        super(ClassificationTask, self).__init__()
        hsize1 = opt.hsize1
        hsize2 = opt.hsize2
        embed_size = opt.embed_size
        n_labels = opt.n_labels
        
        self.opt = opt
        self.attn_fusion = not opt.disable_attn_fusion
        ctx_multi = 3 if self.attn_fusion else 1 #[x, u*x, u] or [x]
        final_multi = len(opt.input_streams)
        self.input_streams = opt.input_streams
        
        self.textenc = textenc

        if 'cap' in self.input_streams:
            self.new_cap_ctx_rnn = RNNEncoder(hsize1 * 2 * 1, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        self.final_fc = nn.Sequential(
            nn.Linear(hsize1*2, hsize2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hsize2, n_labels),
            # nn.Sigmoid()
        )

    def forward(self, vid_input, sub_input, aud_input, obj_input, state_input):
    # def forward(self, vid_encoded, vid_lens, sub_encoded, sub_lens, aud_encoded, aud_lens, obj_encoded, obj_lens, state_input, generate_vectors, test=False):
        final_vectors = []

        if 'cap' in self.input_streams:
            state_encoded, state_lens = self.textenc(state_input)
            state_vec = self.new_cap_ctx_rnn(state_encoded, state_lens)[1]
            final_vectors.append(state_vec)
        else:
            state_vec = state_lens = torch.ones(5)
        
        outputs = self.final_fc(torch.cat(final_vectors, dim=1)).squeeze()

        return outputs, state_vec


