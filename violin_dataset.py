# this code is developed based on https://github.com/jayleicn/TVQA

import numpy as np
import h5py
import os
import json
import re
import torch
import pickle
from collections import Counter
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from config import get_argparse
from transformers import *

def clean_str(string):
    """ Tokenization/string cleaning for strings.
    Taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?:.\'`]", " ", string)  # <> are added after the cleaning process
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)  # split as two words
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\.\.\.", " . ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

class ViolinDataset(Dataset):
    def __init__(self, opt, bert_tokenizer, mode='train'):
        print('='*20)
        super(ViolinDataset, self).__init__()

        self.mode = mode
        self.vid_feat = {}
        self.aud_feat = {}
        self.embed_dim = 300
        self.bert_tokenizer = bert_tokenizer
        self.max_sub_l = opt.max_sub_l
        self.input_streams = opt.input_streams
        self.no_normalize_v = opt.no_normalize_v
        self.trope2idx = json.load(open(os.path.join(opt.feat_dir, 'trope2idx.json'),'r'))
        
        # entire_clip_info = json.load(open(os.path.join(opt.feat_dir, 'violin_annotation.json'),'r'))
        # self.clip_info = []
        
        # for clip_id, clip in entire_clip_info.items():
        #     if clip['split'] == self.mode or self.mode == 'all':
        #         self.clip_info.append(clip)
        trope_file = json.load(open(os.path.join(opt.feat_dir, opt.trope_file),'r'))
        if self.mode == 'train':
            self.clip_info = trope_file['train']
        elif self.mode == 'validate':
            self.clip_info = trope_file['val']
        else:
            self.clip_info = trope_file['test']

        # print('dataset mode', self.mode, '\tdata size', len(self.clip_info))
        self.clip_set = set([c['data-video-name'][:-4] for c in self.clip_info])
        # removed problem video
        # r_list = ['436.mp4', '1559.mp4', '5367.mp4', '9285.mp4', '11843.mp4', '22700.mp4'] 
        # r_list += ['17115.mp4', '1848.mp4', '26698.mp4', '33266.mp4', '33316.mp4', '26694.mp4']
        #r_list_h5 = set([i.split(".")[0] for i in r_list])
        
        # for clip in self.clip_info:
        #     if clip['data-video-name'] in r_list:
        #         self.clip_info.remove(clip)
        # for r in r_list:
        #     if r in self.clip_set:
        #         self.clip_set.remove(r)
        # assert len(clip_set) == len(self.clip_info)

        if 'vid' in self.input_streams:
            assert opt.cnn_feat or opt.c3d_feat
            print('loading video {} features'.format(opt.visual_feat))
            #with h5py.File(os.path.join(opt.feat_dir, 'TVtrope_resnet101.hdf5' if opt.feat=='resnet' else 'all_c3d_fc6_features.h5'), 'r') as fin:
            if opt.cnn_feat and opt.visual_feat in ['cnn', 'both']:
                with h5py.File(opt.cnn_feat, 'r') as fin:
                    fin_keys = fin.keys() #[i for i in fin.keys() if i not in r_list_h5]
                    for clip_id in tqdm(fin_keys):
                        # if int(clip_id) == 26694:#26694 for broken feat
                        #     continue
                        if clip_id in self.clip_set:
                            if opt.frame == '':
                                self.vid_feat[clip_id] = torch.Tensor(np.array(fin[clip_id]))
                            # else:
                            #     tt = torch.Tensor(np.array(fin[clip_id]))
                            #     frame_num = 0
                            #     if opt.frame == 'last':
                            #         frame_num = len(tt)-1
                            #     elif opt.frame == 'mid':
                            #         frame_num = int(len(tt)/2)
                            #     self.vid_feat[clip_id] = tt[frame_num].unsqueeze(0)
                            
            if opt.c3d_feat and opt.visual_feat in ['c3d', 'both']:
                with h5py.File(opt.c3d_feat, 'r') as fin:
                    fin_keys = fin.keys() #[i for i in fin.keys() if i not in r_list_h5]
                    for clip_id in tqdm(fin_keys):
                        # if int(clip_id) == 26694:#26694 for broken feat
                        #     continue
                        if clip_id in self.clip_set:
                            if opt.frame == '':
                                cur_feat = torch.Tensor(np.array(fin[clip_id]))
                            # else:
                            #     tt = torch.Tensor(np.array(fin[clip_id]))
                            #     frame_num = 0
                            #     if opt.frame == 'last':
                            #         frame_num = len(tt)-1
                            #     elif opt.frame == 'mid':
                            #         frame_num = int(len(tt)/2)
                            #     cur_feat = tt[frame_num].unsqueeze(0)
                            self.vid_feat[clip_id] = self.concat_workaround(self.vid_feat[clip_id], cur_feat) if opt.cnn_feat and opt.visual_feat in ['cnn', 'both'] else cur_feat
            # assert len(self.vid_feat) == len(self.clip_info)
        if 'aud' in opt.input_streams:
            print('loading audio {} features'.format(opt.audio_feat))
            if opt.audio_feat == 'sn':
                with h5py.File(opt.sn_feat, 'r') as fin:
                    fin_keys = fin.keys()
                    for clip_id in tqdm(fin_keys):
                        c_id = clip_id[:-4]
                        if c_id in self.clip_set:
                            self.aud_feat[c_id] = torch.Tensor(np.array(fin[clip_id]))

        if 'obj' in opt.input_streams:
            print('loading object {} features'.format(opt.object_feat))
            self.obj_num = opt.obj_num
            self.vcpt_dir = opt.vcpt_dir


        # print('loading subtitles and statements')
        print('loading description')
        for clip in tqdm(self.clip_info):
            clip['padded_statement'] = self.tokenize_and_pad(clean_str(clip['data-video-descrip']).lower())
            # if 'sub' in self.input_streams:
            clip['padded_sub'] = self.tokenize_and_pad(clean_str(clip['data-video-sub']).lower())
            # get statement
            # clip['padded_statement'] = [[self.tokenize_and_pad(clean_str(pair[i]).lower()) for i in range(2)] for pair in clip['statement']]
            
    def concat_workaround(self, cnn, c3d): #a workaround for stupid a hero bug
        if cnn.shape[0]+1 == c3d.shape[0]:
            cnn_new = torch.zeros(c3d.shape[0], cnn.shape[1])
            cnn_new[0] = cnn[0]
            cnn_new[1:] = cnn
        else:
            cnn_new = cnn
        return torch.cat([cnn_new, c3d], dim=1)

    def tokenize_and_pad(self, sent):
        tokens = self.bert_tokenizer.tokenize(sent)
        if len(tokens) > self.max_sub_l-2:
            tokens = tokens[:self.max_sub_l-2]
        tokens = ['[CLS]']+tokens+['[SEP]']
        sent_len = len(tokens)
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(tokens)
        padding = [0]*(self.max_sub_l-len(tokens))
        input_ids += padding
        input_mask += padding
        return (torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(sent_len, dtype=torch.int))
    
    def __len__(self):
        return len(self.clip_info)
    
    def __getitem__(self, idx):
        clip = self.clip_info[idx]
        #print(clip['file'])
        
        # visual feat
        clip_name = clip['data-video-name'][:-4]
        if 'vid' in self.input_streams:
            vid_feat = self.vid_feat[clip_name]
            if not self.no_normalize_v:
                vid_feat = nn.functional.normalize(vid_feat, p=2, dim=1)
        else:
            vid_feat = None

        if 'aud' in self.input_streams:
            aud_feat = self.aud_feat[clip_name]
        else:
            aud_feat = None

        # subtitles
        sub_input = clip['padded_sub']

        # objects
        obj_feat = {'obj': '', 'loc': '', 'obj_num': 0}
        if 'obj' in self.input_streams:
            vcpt_path = os.path.join(self.vcpt_dir, clip_name + '.pickle')
            with open(vcpt_path, "rb") as f:
                vcpt = pickle.load(f)
            obj_feat["obj"], obj_feat["loc"] = self.get_vcpt(vcpt, self.obj_num)
            obj_feat['obj_num'] = self.obj_num
        else:
            obj_feat = None

        # statements
        state_input = clip['padded_statement']

        label = torch.tensor(self.trope2idx[clip['data-video-tropename']])
                       
        return clip_name, vid_feat, sub_input, aud_feat, obj_feat, state_input, label

    def get_vcpt(self, vcpt, obj_num):
        # vcpt frames * box_num {box:[4], name:'', cfd:[], roi:[2048]}
        obj = []
        loc = []
        for frame in vcpt:
            obj_sort = sorted(frame, key=lambda i:i['cfd'], reverse=True)[:obj_num]
            o = []
            l = []
            for i in obj_sort:
                o.append(i['roi'])
                # w = i['name'].lower()
                # w = self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
                # o.append(w)
                l.append(i['box'])
            obj.append(o)
            loc.append(l)
        return obj, loc

    # def get_state_pair(self, idx):
    #     clip = self.clip_info[int(idx/3)]
    #     return clip['statement'][idx%3]

def pad_collate(batch):
    def pad_video_seq(vid_seq):
        lengths = torch.LongTensor([len(seq) for seq in vid_seq])
        v_dim = vid_seq[0].size()[1]
        padded_seqs = torch.zeros(len(vid_seq), max(lengths), v_dim).float()
        for idx, seq in enumerate(vid_seq):
            padded_seqs[idx, :lengths[idx]] = seq
        return padded_seqs, lengths
    
    def pad_word_seq(word_list):
        word_seq = [torch.LongTensor(s) for s in word_list]
        lengths = torch.LongTensor([len(seq) for seq in word_seq])
        padded_seqs = torch.zeros(len(word_seq), max(lengths)).long()
        for idx, seq in enumerate(word_seq):
            padded_seqs[idx, :lengths[idx]] = seq
        return padded_seqs, lengths

    def pad_obj_seq(obj_seq, obj_num):
        """sequences is a list of torch float tensors (created from numpy)"""
        # (batch, frame, obj_num, feat_size)
        lengths = torch.LongTensor([len(seq) for seq in obj_seq])
        v_dim = len(obj_seq[0][0][0]) # 2048 or 4
        padded_seqs = torch.zeros(len(obj_seq), max(lengths), obj_num, v_dim).float()
        mask = torch.zeros(len(obj_seq), max(lengths), obj_num, dtype=int)
        for idx, seq in enumerate(obj_seq):
            f_num = lengths[idx]  # frame num
            for i in range(f_num):
                padded_seqs[idx, i, :len(seq[i])] = torch.tensor(seq[i])
                mask[idx, i, :len(seq[i])] = 1
        return padded_seqs, mask
    
    clip_ids, vid_feat, sub_input, aud_feat, obj_feat, state_input, labels = [[x[i] for x in batch] for i in range(7)]
    padded_vid_feat = pad_video_seq(vid_feat) if type(vid_feat[0]) != type(None) else None
    padded_aud_feat = pad_video_seq(aud_feat) if type(aud_feat[0]) != type(None) else None

    padded_obj_feat = {'obj': '', 'loc': '', 'obj_num': 0}
    if type(obj_feat[0]) != type(None):
        padded_obj = pad_obj_seq([obj['obj'] for obj in obj_feat], obj_feat[0]['obj_num'])
        padded_loc = pad_obj_seq([obj['loc'] for obj in obj_feat], obj_feat[0]['obj_num'])
        padded_obj_feat['obj'] = padded_obj
        padded_obj_feat['loc'] = padded_loc
    else:
        padded_obj_feat = None

    return clip_ids, padded_vid_feat, sub_input, padded_aud_feat, padded_obj_feat, state_input, labels

def preprocess_batch(batch, bert, opt):
    def clip_seq(seq, lens, max_len):
        if seq.size()[1] > max_len:
            seq = seq[:,:max_len]
            lens = lens.clamp(min=1, max=max_len)
        return seq.to(opt.device), lens.to(opt.device)

    def clip_obj_seq(seq, masks, max_len):
        if seq.size()[1] > max_len:
            seq = seq[:,:max_len]
            masks = masks[:,:max_len]
        return seq.to(opt.device), masks.to(opt.device)
    
    def extract_bert_feat(bert_input):
        input_ids = torch.stack([x[0] for x in bert_input]).to(opt.device)
        input_mask = torch.stack([x[1] for x in bert_input]).to(opt.device)
        input_lens = torch.stack([x[2] for x in bert_input]).to(opt.device)
        with torch.no_grad():
            output = bert(input_ids, input_mask)
        return output[0], input_lens
    
    clip_ids, padded_vid_feat, sub_input, padded_aud_feat, padded_obj_feat, state_input, labels = batch
    ret = []
    ret.append(clip_ids)
    if 'vid' in opt.input_streams:
        ret.append(clip_seq(padded_vid_feat[0], padded_vid_feat[1], opt.max_vid_l))
    else:
        ret.append(None)
    if 'sub' in opt.input_streams:
        ret.append(extract_bert_feat(sub_input))
    else:
        ret.append(None)
    if 'aud' in opt.input_streams:
        ret.append(clip_seq(padded_aud_feat[0], padded_aud_feat[1], opt.max_aud_l))
    else:
        ret.append(None)
    if 'obj' in opt.input_streams:
        obj = clip_obj_seq(padded_obj_feat['obj'][0], padded_obj_feat['obj'][1], opt.max_vid_l)
        loc = clip_obj_seq(padded_obj_feat['loc'][0], padded_obj_feat['loc'][1], opt.max_vid_l)
        ret.append({'obj':obj, 'loc':loc})
    else:
        ret.append(None)
    ret.append(extract_bert_feat(state_input))
    ret.append(torch.tensor(labels).to(opt.device))
    return ret
