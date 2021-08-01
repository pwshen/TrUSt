# this code is developed based on https://github.com/jayleicn/TVQA

import torch
from torch import nn
import torch.nn.functional as F
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
        # self.lstm_raw_text = RNNEncoder(opt.embed_size, hsize1, bidirectional=True, dropout_p=0, n_layers=1, rnn_type='lstm')

    def forward(self, text_input):
        text_hidden, text_lens = text_input
        # text_projected = self.bert_fc(text_hidden)
        text_encoded = self.bert_fc(text_hidden)
        # text_encoded, _ = self.lstm_raw_text(text_projected, text_lens) #600

        return text_encoded, text_lens

class ObjectEnc(nn.Module):
    def __init__(self, opt):
        super(ObjectEnc, self).__init__()
        hsize1 = opt.hsize1
        embed_size = opt.embed_size

        self.obj_num = opt.obj_num
        self.use_obj_pos = not opt.no_obj_pos
        self.use_frame_pos = not opt.no_frame_pos
        self.use_gcn = not opt.no_gcn
        self.use_gcn_fc = opt.use_gcn_fc

        node_dim = opt.obj_feat_size
        node_dim += opt.pos_size if self.use_obj_pos else 0
        # node_dim += opt.pos_size if self.use_frame_pos else 0
        self.obj_fc = nn.Sequential(
            nn.Linear(node_dim, embed_size),
            nn.ELU(inplace=True)
        )

        if self.use_obj_pos:
            self.obj_pos_fc = nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                # nn.Dropout(0.5),
                nn.Conv2d(64, opt.pos_size, kernel_size=1),
                nn.BatchNorm2d(opt.pos_size),
                nn.ReLU(),
                # nn.Dropout(0.5)
            )

        # if self.use_frame_pos:
        #     self.frame_pos = positionalencoding1d(opt.pos_size, self.num_frames)
        #     self.frame_pos = self.frame_pos.unsqueeze(1).expand(-1, opt.obj_num, -1).cuda()
        #     logger.info("Using framePos")

        if self.use_gcn:
            self.gcn = GCN(
                embed_size,
                embed_size,
                embed_size,
                dropout=0.5,
                mode=['GCN_sim'],
                skip=True,
                num_layers=3,
                ST_n_next=3
            )

        if self.use_gcn_fc:
            self.gcn_fc = nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.ELU(inplace=True),
            )
        self.lstm_raw_obj = RNNEncoder(opt.embed_size, hsize1, bidirectional=True, dropout_p=0, n_layers=1, rnn_type='lstm')

    def forward(self, obj_input):
        obj_feat, obj_mask = obj_input['obj']
        loc, loc_mask = obj_input['loc']
        bsz = obj_feat.shape[0]
        frame_num = obj_feat.shape[1]
        obj_lens = torch.tensor([frame_num * (self.obj_num)] * bsz, dtype=torch.long)

        if self.use_obj_pos:
            loc_feat = self.obj_pos_fc(loc.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            obj_feat = torch.cat([obj_feat, loc_feat], dim=-1)
        # if self.use_frame_pos:
        #     frame_pos = self.frame_pos.unsqueeze(0).expand(bsz, -1, -1, -1)
        #     obj_feat = torch.cat([obj_feat, frame_pos], dim=-1)
        obj_projected = self.obj_fc(obj_feat)
        obj_projected = obj_projected.view(bsz, -1, obj_projected.shape[-1])

        if self.use_gcn:
            obj_mask = obj_mask.view(bsz, -1)
            obj_projected = self.gcn(obj_projected, obj_mask)
        if self.use_gcn_fc:
            obj_projected = self.gcn_fc(obj_projected)

        obj_encoded, _ = self.lstm_raw_obj(obj_projected, obj_lens)

        return obj_encoded, obj_lens

class SimilarityTask(nn.Module):
    def __init__(self, opt, textenc, videnc=None, audenc=None, objenc=None):
        super(SimilarityTask, self).__init__()
        hsize1 = opt.hsize1
        hsize2 = opt.hsize2
        embed_size = opt.embed_size
        
        ctx_multi = 1
        final_multi = len(opt.input_streams)-1
        
        self.input_streams = opt.input_streams

        self.textenc = textenc
        if 'vid' in self.input_streams:
            self.videnc = videnc
            self.vid_ctx_rnn = RNNEncoder(hsize1 * 2 * ctx_multi, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        
        if 'sub' in self.input_streams:
            self.sub_ctx_rnn = RNNEncoder(hsize1 * 2 * ctx_multi, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        if 'aud' in self.input_streams:
            self.audenc = audenc
            self.aud_ctx_rnn = RNNEncoder(hsize1 * 2 * ctx_multi, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        if 'obj' in self.input_streams:
            self.objenc = objenc
            self.obj_ctx_rnn = RNNEncoder(hsize1 * 2 * ctx_multi, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        if 'cap' in self.input_streams:
            self.cap_ctx_rnn = RNNEncoder(hsize1 * 2 * ctx_multi, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        self.generate_fc = nn.Sequential(
                nn.Linear(hsize2*2*final_multi, embed_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(embed_size, embed_size)
            )
    
    def max_along_time(self, outputs, lengths):
        max_outputs = [outputs[i, :int(lengths[i]), :].max(dim=0)[0] for i in range(len(lengths))]
        ret = torch.stack(max_outputs, dim=0)
        assert ret.size() == torch.Size([outputs.size()[0], outputs.size()[2]])
        return ret

    def forward(self, vid_input, sub_input, aud_input, obj_input, state_input):
        encoded_vectors = []

        # if 'cap' in self.input_streams:
        #     state_encoded, state_lens = self.textenc(state_input)
        #     state_vec = self.cap_ctx_rnn(state_encoded, state_lens)[1]
        # else:
        #     state_encoded = state_lens = torch.ones(5)
                
        if 'vid' in self.input_streams:
            vid_encoded, vid_lens = self.videnc(vid_input) #600
            vec_vid = self.vid_ctx_rnn(vid_encoded, vid_lens)[1]

            encoded_vectors.append(vec_vid)
        else:
            vid_encoded = vid_lens = torch.ones(5)

        if 'sub' in self.input_streams:
            sub_encoded, sub_lens = self.textenc(sub_input)
            vec_sub = self.sub_ctx_rnn(sub_encoded, sub_lens)[1]
            
            encoded_vectors.append(vec_sub)
        else:
            sub_encoded = sub_lens = torch.ones(5)

        if 'aud' in self.input_streams:
            aud_encoded, aud_lens = self.audenc(aud_input) #768
            vec_aud = self.aud_ctx_rnn(aud_encoded, aud_lens)[1]
            
            encoded_vectors.append(vec_aud)
        else:
            aud_encoded = aud_lens = torch.ones(5)

        if 'obj' in self.input_streams:
            obj_encoded, obj_lens = self.objenc(obj_input)
            vec_obj = self.obj_ctx_rnn(obj_encoded, obj_lens)[1]

            encoded_vectors.append(vec_obj)
        else:
            obj_encoded = obj_lens = torch.ones(5)

        generate_vectors = self.generate_fc(torch.cat(encoded_vectors, dim=1))

        return generate_vectors

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
        if 'vid' in self.input_streams:
            self.videnc = videnc
            self.new_vid_ctx_rnn = RNNEncoder(hsize1 * 2 * ctx_multi, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        
        if 'sub' in self.input_streams:
            # self.subenc = subenc
            self.new_sub_ctx_rnn = RNNEncoder(hsize1 * 2 * ctx_multi, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        if 'aud' in self.input_streams:
            self.audenc = audenc
            self.new_aud_ctx_rnn = RNNEncoder(hsize1 * 2 * ctx_multi, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        if 'obj' in self.input_streams:
            self.objenc = objenc
            self.new_obj_ctx_rnn = RNNEncoder(hsize1 * 2 * ctx_multi, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        # if 'cap' in self.input_streams:
        #     self.new_cap_ctx_rnn = RNNEncoder(hsize1 * 2 * 1, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        # if 'cap' in self.input_streams and not self.attn_fusion:
        #     self.cap_ctx_rnn = RNNEncoder(hsize1 * 2 * ctx_multi, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        

        if self.attn_fusion:
            # self.weight = nn.Sequential(nn.Linear(hsize2*2*(final_multi-1), 1), nn.Softmax(dim=1))

            self.bidaf = BidafAttn(hsize1 * 2, method="dot")  

            self.final_fc = nn.Sequential(
                nn.Linear(hsize2*2*(final_multi-1), hsize2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hsize2, n_labels)
            )
        else:
            self.final_fc = nn.Linear(hsize2*2*(final_multi-1), n_labels)
            
    
    def max_along_time(self, outputs, lengths):
        max_outputs = [outputs[i, :int(lengths[i]), :].max(dim=0)[0] for i in range(len(lengths))]
        ret = torch.stack(max_outputs, dim=0)
        assert ret.size() == torch.Size([outputs.size()[0], outputs.size()[2]])
        return ret

    def forward(self, vid_input, sub_input, aud_input, obj_input, generate_vectors=None, test=False):
        final_vectors = []


        if self.attn_fusion:
            vf, ve = vid_input
            new_generate_vectors = generate_vectors.unsqueeze(1)
            generate_lens = torch.ones(vf.shape[0], dtype=torch.int)
        else:
            new_generate_vectors = generate_vectors
            final_vectors.append(new_generate_vectors)
                
        if 'vid' in self.input_streams:
            vid_encoded, vid_lens = self.videnc(vid_input) #600
            if self.attn_fusion:
                u_va, _ = self.bidaf(new_generate_vectors, generate_lens, vid_encoded, vid_lens)
                concat_vid = torch.cat([new_generate_vectors, u_va, new_generate_vectors*u_va], dim=-1)
                concat_lens = generate_lens
            else:
                concat_vid = vid_encoded
                concat_lens = vid_lens

            new_vec_vid = self.new_vid_ctx_rnn(concat_vid, concat_lens)[1]
            final_vectors.append(new_vec_vid)

        if 'sub' in self.input_streams:
            sub_encoded, sub_lens = self.textenc(sub_input)
            if self.attn_fusion:
                u_sa, _ = self.bidaf(new_generate_vectors, generate_lens, sub_encoded, sub_lens)
                concat_sub = torch.cat([new_generate_vectors, u_sa, new_generate_vectors*u_sa], dim=-1)
                concat_lens = generate_lens
            else:
                concat_sub = sub_encoded
                concat_lens = sub_lens

            new_vec_sub = self.new_sub_ctx_rnn(concat_sub, concat_lens)[1]
            final_vectors.append(new_vec_sub)

        if 'aud' in self.input_streams:
            aud_encoded, aud_lens = self.audenc(aud_input) #768
            if self.attn_fusion:
                u_va, _ = self.bidaf(new_generate_vectors, generate_lens, aud_encoded, aud_lens)
                concat_aud = torch.cat([new_generate_vectors, u_va, new_generate_vectors*u_va], dim=-1)
                concat_lens = generate_lens
            else:
                concat_aud = aud_encoded
                concat_lens = aud_lens

            new_vec_aud = self.new_aud_ctx_rnn(concat_aud, concat_lens)[1]
            final_vectors.append(new_vec_aud) 

        if 'obj' in self.input_streams:
            obj_encoded, obj_lens = self.objenc(obj_input)
            if self.attn_fusion:
                u_oa, _ = self.bidaf(new_generate_vectors, generate_lens, obj_encoded, obj_lens)
                concat_obj = torch.cat([new_generate_vectors, u_oa, new_generate_vectors*u_oa], dim=-1)
                concat_lens = generate_lens
            else:
                concat_obj = obj_encoded
                concat_lens = obj_lens
            
            new_vec_obj = self.new_obj_ctx_rnn(concat_obj, concat_lens)[1]
            final_vectors.append(new_vec_obj)
        
        outputs = self.final_fc(torch.cat(final_vectors, dim=-1)).squeeze()

        return outputs

class MultiTaskLoss(nn.Module):
    def __init__(self, similarity_loss='Cosine'):
        super(MultiTaskLoss, self).__init__()
        self.loss_fn1 = nn.CrossEntropyLoss(size_average=False)
        self.loss_fn2 = nn.CosineSimilarity(dim=1)
        # self.loss_fn2 = nn.MSELoss(size_average=False)

        self.vars1 = nn.Parameter(torch.zeros(1))
        self.vars2 = nn.Parameter(torch.zeros(1))

    def forward(self, gen_cap, tru_cap, outputs, labels):

        precision1 = torch.exp(-self.vars1)
        loss1 = self.loss_fn1(outputs, labels) * precision1 + self.vars1
        print(loss1.data.tolist())

        precision2 = torch.exp(-self.vars2)
        loss2 = torch.sum((1.-self.loss_fn2(gen_cap, tru_cap)) * precision2 + self.vars2, -1)
        # loss2 = self.loss_fn2(gen_cap, tru_cap) * precision2 + self.vars2
        # loss2 = torch.mean(loss2)
        print(loss2.data.tolist())

        print(self.vars1.data.tolist())
        print(self.vars2.data.tolist())
        total_loss = loss1 + loss2
        return total_loss, loss1, loss2

