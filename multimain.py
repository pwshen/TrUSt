# this code is developed based on https://github.com/jayleicn/TVQA

import os
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from violin_dataset import ViolinDataset, pad_collate, preprocess_batch
from configmulti import get_argparse
from model.ViolinBaseMulti import VideoEnc, AudioEnc, TextEnc, ObjectEnc, SimilarityTask, ClassificationTask, MultiTaskLoss
from model.capenc import ClassificationTask as capenc
from model.capenc import TextEnc as capte

from transformers import *


def check_param(model):
    grad_lst = []
    for name, param in model.named_parameters():
        grad_lst.append(torch.norm(param.grad.data.view(-1)).item())
    return grad_lst

def get_data_loader(opt, dset, batch_size, if_shuffle):
    return DataLoader(dset, batch_size=batch_size, shuffle=if_shuffle, num_workers=0, collate_fn=pad_collate)

def train_epoch(opt, trn_dset, val_dset, tst_dset, smtask, clstask, optimizer, criterionsm, criterioncls, criterion, epoch, previous_best_acc, cap_enc):
    smtask.train()
    clstask.train()
    # criterion.train()
    train_loader = get_data_loader(opt, trn_dset, opt.batch_size, True)

    #check_param(model)
    train_losssm = []
    train_losscls = []
    train_loss = []
    valid_acc_log = ["epoch\ttrn acc\tval acc"]
    train_corrects = []
    print('epoch', epoch, '='*20)
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        with torch.autograd.set_detect_anomaly(True):
            cur_clip_ids, padded_vid_feat, sub_feat, padded_aud_feat, padded_obj_feat, state_feat, labels = preprocess_batch(batch, bert, opt)

            _, tru_cap = cap_enc(padded_vid_feat, sub_feat, padded_aud_feat, padded_obj_feat, state_feat)

            gen_cap = smtask(padded_vid_feat, sub_feat, padded_aud_feat, padded_obj_feat, state_feat)
            outputs = clstask(padded_vid_feat, sub_feat, padded_aud_feat, padded_obj_feat, gen_cap)
            
            losssm = torch.sum(1. - criterionsm(gen_cap, tru_cap.detach()), -1)
            losscls = criterioncls(outputs, labels)

            loss = losscls * opt.losscls_w + losssm * opt.losssm_w

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        train_losssm.append(losssm.item())
        train_losscls.append(losscls.item())
        train_loss.append(loss.item())
        pred_ids = outputs.data.max(1)[1]
        train_corrects += pred_ids.eq(labels.data).cpu().numpy().tolist()

    train_acc = sum(train_corrects) / float(len(train_corrects))
    train_losssm = sum(train_losssm) / float(len(train_corrects))
    train_losscls = sum(train_losscls) / float(len(train_corrects))
    train_loss = sum(train_loss) / float(len(train_corrects))
    #print(check_param(model))
    
    # validate
    valid_loader = get_data_loader(opt, val_dset, opt.test_batch_size, False)
    valid_acc, valid_losssm, valid_losscls, valid_loss = validate(smtask, clstask, valid_loader, criterionsm, criterioncls, criterion, cap_enc, opt=opt)

    valid_log_str = "%02d\t%.4f\t%.4f" % (epoch, train_acc, valid_acc)
    valid_acc_log.append(valid_log_str)
    print("\n Epoch %d losssm %.4f losscls %.4f\n" % (epoch, valid_losssm, valid_losscls))

    print("\n Epoch %d TRAIN loss %.4f acc %.4f VAL loss %.4f acc %.4f\n"
            % (epoch, train_loss, train_acc, valid_loss, valid_acc))
    with open(os.path.join(opt.results_dir, "valid_acc.log"), "a") as f:
        f.write("Epoch %d TRAIN loss %.4f acc %.4f VAL loss %.4f acc %.4f\n"
            % (epoch, train_losscls, train_acc, valid_losscls, valid_acc))
    with open(os.path.join(opt.results_dir, "valid_cls.log"), "a") as f:
        f.write("Epoch %d TRAIN cls %.4f acc %.4f VAL cls %.4f acc %.4f\n"
            % (epoch, train_losscls, train_acc, valid_losscls, valid_acc))
    with open(os.path.join(opt.results_dir, "valid_sm.log"), "a") as f:
        f.write("Epoch %d TRAIN sm %.4f acc %.4f VAL sm %.4f acc %.4f\n"
            % (epoch, train_losssm, train_acc, valid_losssm, valid_acc))

    # torch.save(model.state_dict(), os.path.join(opt.results_dir, "model_epoch_{}.pth".format(epoch)))
    if valid_acc > previous_best_acc:
        previous_best_acc = valid_acc
        torch.save(smtask.state_dict(), os.path.join(opt.results_dir, "sm_valid.pth"))
        torch.save(clstask.state_dict(), os.path.join(opt.results_dir, "cls_valid.pth"))
        # torch.save(criterion.state_dict(), os.path.join(opt.results_dir, "criterion.pth"))

        test_loader = get_data_loader(opt, tst_dset, opt.test_batch_size, False)
        test_acc, test_losssm, test_losscls, test_loss = validate(smtask, clstask, test_loader, criterionsm, criterioncls, criterion, cap_enc, dump_result=True, opt=opt)
        print("\n Epoch %d TEST loss %.4f acc %.4f\n"
            % (epoch, test_loss, test_acc))
        with open(os.path.join(opt.results_dir, "valid_acc.log"), "a") as f:
            f.write("Epoch %d TEST loss %.4f acc %.4f\n"
            % (epoch, test_loss, test_acc))
    
    return previous_best_acc

def validate(smtask, clstask, valid_loader, criterionsm, criterioncls, criterion, cap_enc, dump_result=False, opt=None):
    smtask.eval()
    clstask.eval()
    # criterion.eval()
    with torch.no_grad():
        valid_corrects = []
        clip_ids = []
        true_labels = []
        pred_labels = []
        valid_losssm = []
        valid_losscls = []
        valid_loss = []
        generate_caps = []
        true_caps = []
        for _, batch in enumerate(tqdm(valid_loader)):
            cur_clip_ids, padded_vid_feat, sub_feat, padded_aud_feat, padded_obj_feat, state_feat, labels = preprocess_batch(batch, bert, opt)

            _, tru_cap = cap_enc(padded_vid_feat, sub_feat, padded_aud_feat, padded_obj_feat, state_feat)
            gen_cap = smtask(padded_vid_feat, sub_feat, padded_aud_feat, padded_obj_feat, state_feat)
            outputs = clstask(padded_vid_feat, sub_feat, padded_aud_feat, padded_obj_feat, gen_cap)

            valid_sm = torch.sum(1. - criterionsm(gen_cap, tru_cap.detach()), -1)
            valid_cls = criterioncls(outputs, labels)
            loss = valid_cls * opt.losscls_w + valid_sm * opt.losssm_w

            # loss, valid_cls, valid_sm = criterion(gen_cap, tru_cap, outputs, labels)

            # measure accuracy and record loss
            clip_ids += cur_clip_ids
            valid_losssm.append(valid_sm.item())
            valid_losscls.append(valid_cls.item())
            valid_loss.append(loss.item())
            pred_ids = outputs.data.max(1)[1]
            valid_corrects += pred_ids.eq(labels.data).cpu().numpy().tolist()

            true_labels += labels.cpu().numpy().tolist()
            pred_labels += pred_ids.cpu().numpy().tolist()
            true_caps += tru_cap.cpu().numpy().tolist()
            generate_caps += gen_cap.cpu().numpy().tolist()
            
        valid_acc = sum(valid_corrects) / float(len(valid_corrects))
        valid_losssm = sum(valid_losssm) / float(len(valid_corrects))
        valid_losscls = sum(valid_losscls) / float(len(valid_corrects))
        valid_loss = sum(valid_loss) / float(len(valid_corrects))

        trope2idx = json.load(open(os.path.join(opt.feat_dir, 'trope2idx.json'),'r'))
        idx2trope = dict((v,k) for k,v in trope2idx.items())
        outputs = []
        for i in range(len(clip_ids)):
            out = {}
            out['video_name'] = clip_ids[i]
            out['true'] = idx2trope[true_labels[i]]
            out['pred'] = idx2trope[pred_labels[i]]
            outputs.append(out)
        if dump_result:
            output_file = "{}/outputs.json".format(opt.results_dir)
            with open(output_file, 'w') as f:
                json.dump(outputs, f)

        pca_input = {}
        pca_input['true_cap'] = true_caps
        pca_input['gen_cap'] = generate_caps
        pca_input['labels'] = true_labels
        out_file = "{}/pca_input.json".format(opt.results_dir)
        with open(out_file, 'w') as f:
            json.dump(pca_input, f)


    return valid_acc, valid_losssm, valid_losscls, valid_loss

if __name__ == '__main__':
    random_seed = 219373241
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

    opt = get_argparse()
    print(opt)
    
    bert = BertModel.from_pretrained(opt.bert_dir)
    bert_tokenizer = BertTokenizer.from_pretrained(opt.bert_dir)
    bert.to(opt.device)
    bert.eval()
    
    DSET = eval(opt.data)
    
    if not opt.test:
        os.makedirs(opt.results_dir)
        trn_dset = DSET(opt, bert_tokenizer, 'train')
        val_dset = DSET(opt, bert_tokenizer, 'validate')
        tst_dset = DSET(opt, bert_tokenizer, 'test')
    else:
        tst_dset = DSET(opt, bert_tokenizer, 'test')

    textenc = TextEnc(opt)
    if 'vid' in opt.input_streams:
        videnc = VideoEnc(opt)
    else:
        videnc = None
    if 'aud' in opt.input_streams:
        audenc = AudioEnc(opt)
    else:
        audenc = None
    if 'obj' in opt.input_streams:
        objenc = ObjectEnc(opt)
    else:
        objenc = None
    smtask = SimilarityTask(opt, textenc, videnc, audenc, objenc)
    clstask = ClassificationTask(opt, textenc, videnc, audenc, objenc)
    # criterion = MultiTaskLoss(similarity_loss='Cosine')
    print(smtask)
    print(clstask)

    cap_te = capte(opt)
    cap_enc = capenc(opt, cap_te)
    cap_enc.load_state_dict(torch.load('./data/capenc.pth'))
    cap_enc.to(opt.device)
    cap_enc.eval()

    if opt.test:
        # model.load_state_dict(torch.load(opt.model_path))
        smtask.load_state_dict(torch.load(os.path.join(opt.model_dir, "sm_valid.pth")))
        clstask.load_state_dict(torch.load(os.path.join(opt.model_dir, "cls_valid.pth")))
        # criterion.load_state_dict(torch.load(os.path.join(opt.model_dir, "criterion.pth")))

    smtask.to(opt.device)
    clstask.to(opt.device)
    # criterion.to(opt.device)
    
    if opt.test:
        # criterion = nn.CrossEntropyLoss(size_average=False).to(opt.device)
        test_loader = get_data_loader(opt, tst_dset, opt.test_batch_size, False)
        test_loss, test_acc, test_loss1, test_loss2 = validate(smtask, clstask, test_loader, criterionsm, criterioncls, None, cap_enc, opt=opt)
        print("Test loss %.4f acc %.4f\n" % (test_loss, test_acc))
        with open(opt.model_path+'_test.res','w') as ftst:
            ftst.write("Test loss %.4f acc %.4f\n" % (test_loss, test_acc))
    else:
        criterionsm = nn.CosineSimilarity(dim=1).to(opt.device)
        criterioncls = nn.CrossEntropyLoss(size_average=False).to(opt.device)

        params = list(smtask.parameters()) + list(clstask.parameters())
        optimizer = torch.optim.Adam(params, lr=opt.lr, weight_decay=opt.wd)

        best_acc = 0.
        for epoch in range(opt.n_epoch):
            best_acc = train_epoch(opt, trn_dset, val_dset, tst_dset, smtask, clstask, optimizer, criterionsm, criterioncls, None, epoch, best_acc, cap_enc)
