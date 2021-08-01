import os
import time
import torch
import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir_base", type=str, default="results/results")
    parser.add_argument("--trope_file", type=str, default="trope_split1.json")
    parser.add_argument("--feat_dir", type=str, default="/tmp2/r08922016/video_tropes/data/")
    parser.add_argument("--bert_dir", type=str, default="/tmp2/r08922016/video_tropes/data/bert")
    parser.add_argument("--vcpt_dir", type=str, default="/work/pwshen1214/tvtrope_vcpts/", help="visual concepts feature path")
    
    parser.add_argument("--model", type=str, default="ViolinBase", choices=['ViolinBase'])
    parser.add_argument("--data", type=str, default="ViolinDataset", choices=['ViolinDataset'])
    #parser.add_argument("--log_freq", type=int, default=10, help="print, save training info")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=5e-6, help="weight decay")
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs to run")
    parser.add_argument("--losssm_w", type=float, default=10, help="number of epochs to run")
    parser.add_argument("--losscls_w", type=float, default=0.01, help="number of epochs to run")
    #parser.add_argument("--grad_clip", type=float, default=0.01, help="gradient clip value")
    #parser.add_argument("--init_train_epoch", type=int, default=15, help="number of epochs for initial train (without early stopping)")
    #parser.add_argument("--max_es_cnt", type=int, default=200, help="number of epochs to early stop")
    parser.add_argument("--batch_size", type=int, default=128, help="mini-batch size")
    parser.add_argument("--test_batch_size", type=int, default=256, help="mini-batch size for testing")
    parser.add_argument("--device", type=int, default=0, help="gpu ordinal, -1 indicates cpu")

    parser.add_argument("--vid_feat_size", type=int, default=4096, help="visual feature dimension")
    parser.add_argument("--aud_feat_size", type=int, default=1024, help="audio feature dimension")
    parser.add_argument("--obj_feat_size", type=int, default=2048, help="object feature dimension")
    parser.add_argument("--input_streams", type=str, nargs="+", choices=["vid", "sub", "none", "cap", 'aud', 'obj'], default=[], help="input streams for the model, or use a single option 'none'")

    parser.add_argument("--hsize1", type=int, default=384, help="hidden size for the video lstm")
    parser.add_argument("--hsize2", type=int, default=384, help="hidden size for the fusion lstm")
    parser.add_argument("--n_labels", type=int, default=132, help="class of tropes")
    parser.add_argument("--embed_size", type=int, default=768, help="word embedding dim")
    parser.add_argument("--pos_size", type=int, default=128, help="position embedding dim")
    parser.add_argument("--max_sub_l", type=int, default=256, help="max length for subtitle")
    parser.add_argument("--max_vid_l", type=int, default=100, help="max length for video feature")
    parser.add_argument("--max_aud_l", type=int, default=256, help="max length for video feature")
    # parser.add_argument("--max_obj_l", type=int, default=256, help="max length for video feature")
    parser.add_argument("--obj_num", type=int, default=5, help="max object num")
    parser.add_argument("--no_normalize_v", action="store_true", help="do not normalize video featrue")
    parser.add_argument("--no_obj_pos", action="store_true", help="do not use object position")
    parser.add_argument("--no_frame_pos", action="store_true", help="do not normalize video featrue")
    parser.add_argument("--no_gcn", action="store_true", help="do not normalize video featrue")
    parser.add_argument("--use_gcn_fc", action="store_true", help="do not normalize video featrue")

    
    parser.add_argument("--visual_feat", type=str, default="", choices=['', 'cnn', 's3d', 'both_cs'])
    parser.add_argument("--speech_feat", type=str, default="", choices=['', 'sub', 'speech'])
    parser.add_argument("--audio_feat", type=str, default="", choices=['', 'sn'])
    parser.add_argument("--object_feat", type=str, default="", choices=['', 'frcnn'])
    parser.add_argument("--cnn_feat", type=str, default="")
    parser.add_argument("--s3d_feat", type=str, default="")
    parser.add_argument("--sn_feat", type=str, default="")
    
    
    parser.add_argument("--disable_attn_fusion", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--model_path", type=str, default="/tmp2/r08922016/video_tropes/violin-master/results/results_2020_11_17_02_38_24_ViolinBase_vid-resnet/best_valid.pth")
    parser.add_argument("--frame", type=str, default="", choices=['first', 'last', 'mid', ''], help="testing with only one frame")
    
    parser.add_argument("--ori_resultdir", type=str, default="")
    
    opt = parser.parse_args()
    if opt.device >= 0:
        opt.device = torch.device('cuda:0')
    #opt.results_dir = opt.results_dir_base + time.strftime("_%Y_%m_%d_%H_%M_%S")+'_'+opt.model
    opt.results_dir = "{}{}_{}_{}_{}_{}_{}".format(opt.results_dir_base, time.strftime("_%Y_%m_%d_%H_%M_%S"), '+'.join(opt.input_streams), opt.visual_feat, opt.audio_feat, '' if opt.disable_attn_fusion else "attn", opt.trope_file.split(".")[0])
    if opt.frame != '':
        opt.results_dir+='_frame-'+opt.frame
    #opt.results_dir += '_'+'-'.join(opt.input_streams+[opt.visual_feat])
    if 'none' in opt.input_streams:
        assert len(opt.input_streams) == 1
        opt.input_streams = []
    v_feat_size = {
        '': 0,
        'cnn': 2048,
        'c3d': 2304,
        's3d': 1024, 
        'both_cs':4352
    }
    opt.vid_feat_size = v_feat_size[opt.visual_feat]#2048 if opt.visual_feat == 'resnet' else 4096
    if 'cap' not in opt.input_streams:
        if not opt.disable_attn_fusion:
            print("WARNING: disable_attn_fusion should be set to TRUE")
            opt.disable_attn_fusion = True

    return opt
