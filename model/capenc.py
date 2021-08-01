# this code is developed based on https://github.com/jayleicn/TVQA

import torch
from torch import nn
from .rnn import RNNEncoder
from .bidaf2 import BidafAttn
from .gcn import GCN
from .PosEmbed import positionalencoding1d
import pickle

class VideoEnc(nn.Module):
    def __init__(self, opt):
        super(VideoEnc, self).__init__()
        hsize1 = opt.hsize1
        embed_size = opt.embed_size

        self.video_fc = nn.Sequential(
            nn.Linear(opt.vid_feat_size, embed_size),
            nn.Dropout(0.5),
            nn.Tanh()
        )
        self.lstm_raw = RNNEncoder(opt.embed_size, hsize1, bidirectional=True, dropout_p=0, n_layers=1, rnn_type='lstm')

    def forward(self, vid_input):
        vid_feat, vid_lens = vid_input
        vid_projected = self.video_fc(vid_feat) #768
        vid_encoded, _ = self.lstm_raw(vid_projected, vid_lens) #600

        return vid_encoded, vid_lens

class AudioEnc(nn.Module):
    def __init__(self, opt):
        super(AudioEnc, self).__init__()
        hsize1 = opt.hsize1
        embed_size = opt.embed_size

        self.audio_fc = nn.Sequential(
                nn.Linear(opt.aud_feat_size, embed_size),
                nn.Dropout(0.5),
                nn.Tanh()
            )
        self.lstm_raw_aud = RNNEncoder(opt.embed_size, hsize1, bidirectional=True, dropout_p=0, n_layers=1, rnn_type='lstm')

    def forward(self, aud_input):
        aud_feat, aud_lens = aud_input
        aud_projected = self.audio_fc(aud_feat) #768
        aud_encoded, _ = self.lstm_raw_aud(aud_projected, aud_lens) #600

        return aud_encoded, aud_lens

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

        # self.lstm_mature_vid = RNNEncoder(hsize2 * 2 * 5, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        
        self.textenc = textenc
        # if 'vid' in self.input_streams:
        #     self.videnc = videnc
        #     self.new_vid_ctx_rnn = RNNEncoder(hsize1 * 2 * ctx_multi, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        
        # if 'sub' in self.input_streams:
        #     # self.subenc = subenc
        #     self.new_sub_ctx_rnn = RNNEncoder(hsize1 * 2 * ctx_multi, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        # if 'aud' in self.input_streams:
        #     self.audenc = audenc
        #     self.new_aud_ctx_rnn = RNNEncoder(hsize1 * 2 * ctx_multi, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        # if 'obj' in self.input_streams:
        #     self.objenc = objenc
        #     self.new_obj_ctx_rnn = RNNEncoder(hsize1 * 2 * ctx_multi, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        if 'cap' in self.input_streams:
            self.new_cap_ctx_rnn = RNNEncoder(hsize1 * 2 * 1, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        # if 'cap' in self.input_streams and not self.attn_fusion:
        #     self.cap_ctx_rnn = RNNEncoder(hsize1 * 2 * ctx_multi, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        
        # if len(self.input_streams) > 1:
        #     self.bidaf = BidafAttn(hsize1 * 2, method="dot")  

        #     self.final_fc = nn.Sequential(
        #         nn.Linear(hsize2*2*(final_multi-1), hsize2),
        #         nn.ReLU(),
        #         nn.Dropout(0.5),
        #         nn.Linear(hsize2, n_labels),
        #         # nn.Sigmoid()
        #     )
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
                
        # if 'vid' in self.input_streams:
        #     vid_encoded, vid_lens = self.videnc(vid_input) #600
        #     if self.attn_fusion:
        #         u_va, _ = self.bidaf(new_generate_vectors, generate_lens, vid_encoded, vid_lens)
        #         concat_vid = torch.cat([new_generate_vectors, u_va, new_generate_vectors*u_va], dim=-1)
        #         concat_lens = generate_lens
        #     else:
        #         concat_vid = vid_encoded
        #         concat_lens = vid_lens

        #     new_vec_vid = self.new_vid_ctx_rnn(concat_vid, concat_lens)[1]
        #     final_vectors.append(new_vec_vid)
        # else:
        #     vid_encoded = vid_lens = torch.ones(5)

        # if 'sub' in self.input_streams:
        #     sub_encoded, sub_lens = self.textenc(sub_input)
        #     if self.attn_fusion:
        #         u_sa, _ = self.bidaf(new_generate_vectors, generate_lens, sub_encoded, sub_lens)
        #         concat_sub = torch.cat([new_generate_vectors, u_sa, new_generate_vectors*u_sa], dim=-1)
        #         concat_lens = generate_lens
        #     else:
        #         concat_sub = sub_encoded
        #         concat_lens = sub_lens

        #     new_vec_sub = self.new_sub_ctx_rnn(concat_sub, concat_lens)[1]
        #     final_vectors.append(new_vec_sub)
        # else:
        #     sub_encoded = sub_lens = torch.ones(5)

        # if 'aud' in self.input_streams:
        #     aud_encoded, aud_lens = self.audenc(aud_input) #768
        #     if self.attn_fusion:
        #         u_va, _ = self.bidaf(new_generate_vectors, generate_lens, aud_encoded, aud_lens)
        #         concat_aud = torch.cat([new_generate_vectors, u_va, new_generate_vectors*u_va], dim=-1)
        #         concat_lens = generate_lens
        #     else:
        #         concat_aud = aud_encoded
        #         concat_lens = aud_lens

        #     new_vec_aud = self.new_aud_ctx_rnn(concat_aud, concat_lens)[1]
        #     final_vectors.append(new_vec_aud)
        # else:
        #     aud_encoded = aud_lens = torch.ones(5)

        # if 'obj' in self.input_streams:
        #     obj_encoded, obj_lens = self.objenc(obj_input)
        #     if self.attn_fusion:
        #         u_oa, _ = self.bidaf(new_generate_vectors, generate_lens, obj_encoded, obj_lens)
        #         concat_obj = torch.cat([new_generate_vectors, u_oa, new_generate_vectors*u_oa], dim=-1)
        #         concat_lens = generate_lens
        #     else:
        #         concat_obj = obj_encoded
        #         concat_lens = obj_lens
            
        #     new_vec_obj = self.new_obj_ctx_rnn(concat_obj, concat_lens)[1]
        #     final_vectors.append(new_vec_obj)
        # else:
        #     obj_encoded = obj_lens = torch.ones(5)
        
        outputs = self.final_fc(torch.cat(final_vectors, dim=1)).squeeze()

        return outputs, state_vec


