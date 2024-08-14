import os    
import tqdm
import jsonlines
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import argparse

def sort_index(lst, std, top, rev=True):
    lst_concat = np.array(lst[0])
    for i in range(len(lst) - 1):
        lst_concat = np.concatenate((lst_concat, lst[i+1]))
    
    if top == 0:
        cci_threshold = np.mean(lst_concat)
    elif top < 0:
        # max - n%*(max-min)
        cci_threshold = (1+top/100) * np.max(lst_concat) - top/100 * np.min(lst_concat)
    else:
        cci_threshold = np.sort(lst_concat)[-top]
    
    lst = [np.array(l) for l in lst]
    zero_idx = [l < cci_threshold for l in lst]

    for i in range(len(lst)):
        lst[i][zero_idx[i]] = 0
    #print(lst)
    for i in range(len(lst)):
        lst[i] = np.sum(lst[i])
    return np.array(lst)

parser = argparse.ArgumentParser()
parser.add_argument("--val", action= "store_true", help = "Using the CTI threshold calibrated on validation set")
parser.add_argument("--CCI", type=int, default=-5, help = "CCI threshold. Using Top k strategy if k > 0; otherwise Top (-k)% if k < 0, default -5.")

args = parser.parse_args()

if args.val:
    val = True
else:
    val = False

cci_topk = args.CCI
#cci_topk = 3
#cci_topk = -5

dir_path_input = './data/data-match/'
dir_path_attribute = './internals-match/'


files = os.listdir(dir_path_input)
langs = ['te', 'bn', 'ja', 'fi', 'ru']

# calibration thresholds for CTI
threshold = {'te': 6.152940643304108, 'bn': 1.8217959971786004, 'ja': 2.4492197151360466, 'fi': 2.3185973456030258, 'ru': 6.349929943943213}

res_thres = dict()
for fname in files:
    if 'pos' not in fname: continue
    if 'val' in fname or 'train' in fname: continue

    lang = fname.split('.')[0].split('-')[1]
    print('*****************************')
    print(lang)
    idx = 0
    
    pos_ins_cci = []
    neg_ins_cci = []

    # for CCI
    accuracy_cci = 0
    accuracy_tot_cci = 0
    
    correlation_nli = 0
    correlation_nli_tot = 0

    human_nli = 0
    human_nli_tot = 0

    human_nli_inlang = 0
    human_nli_inlang_tot = 0

    dir_path_nli = './nli-predictions/'
    nli_dir = dir_path_nli + lang + '-nli-translate-en.txt'
    nli_inlang_dir = dir_path_nli + lang + '-nli-in-lang.txt'
    with open(nli_dir) as f:
        nli_pred = f.read()
        nli_pred = eval(nli_pred)
    
    with open(nli_inlang_dir) as f:
        nli_pred_inlang = f.read()
        nli_pred_inlang = eval(nli_pred_inlang)
    
    with open(dir_path_input + fname) as f:
        for item in jsonlines.Reader(f):
            #"<Q>: " + item['query'] + " <P>:" + passage
            # Get weighted average CCI for each context
            attr_file = dir_path_attribute + lang + '-' + str(idx) + '.json'
            with open(attr_file) as r:
                res_pecora = json.load(r)

            if val:
                threshold_CTI = threshold[item['query_language']]
            else:
                threshold_CTI = np.mean(res_pecora["cti_scores"]) + 1 * np.std(res_pecora["cti_scores"])
            
            mark_attri = False
            for i in res_pecora["cti_scores"]:
                if i >= threshold_CTI:
                    mark_attri = True

            # Evaluate CCI correlation
            if not mark_attri:
                for i, n in enumerate(item['idxs']):
                    mark_nli = True if nli_pred[n] == 1 else False
                    mark_human = True if item['ais'][i] else False
                    mark_nli_inlang = True if nli_pred_inlang[n] == 1 else False
                    
                    neg_ins_cci.append(0)
                    if not mark_human:
                        accuracy_cci += 1
                    accuracy_tot_cci += 1

                    if mark_nli == mark_human:
                        human_nli += 1
                    human_nli_tot += 1

                    if mark_nli_inlang == mark_human:
                        human_nli_inlang += 1
                    human_nli_inlang_tot += 1
            else:
                num_avg = np.sum(np.array(res_pecora['cti_scores']) >= threshold_CTI)
                cci_list = res_pecora['cci_scores'][:num_avg]

                # Get 0/1 mask to skip index in the template
                # 0 means useless tokens <0>, <1> ...
                passage_id = 0
                mask = np.ones(len(res_pecora['input_context_tokens']), dtype='float64')
                for i in range(len(res_pecora['input_context_tokens'])):
                    if str(passage_id) in res_pecora['input_context_tokens'][i] and ('<' in res_pecora['input_context_tokens'][max(0, i-1)] or '<' in res_pecora['input_context_tokens'][i]) and ('>' in res_pecora['input_context_tokens'][min(len(res_pecora['input_context_tokens'])-1, i+1)] or '>' in res_pecora['input_context_tokens'][i]):
                        mask[i] = 0
                        if '<' in res_pecora['input_context_tokens'][max(0, i-1)]:
                            mask[max(0, i-1)] = 0
                        if '>' in res_pecora['input_context_tokens'][min(len(res_pecora['input_context_tokens'])-1, i+1)]:
                            mask[min(len(res_pecora['input_context_tokens'])-1, i+1)] = 0
                        passage_id += 1

                if sum(mask) == 0: 
                    idx += 1
                    continue

                cci_res = np.array([0 for _ in range(len(item['ais']))], dtype="float64")
                for i in cci_list:
                    # Split concat passages 
                    split_value_list = []
                    save_flag = False
                    tmp = []
                
                    for j in range(len(mask)):
                        if mask[j] == 0 and save_flag:
                            split_value_list.append(tmp)
                            tmp = []
                            save_flag = False
                        if mask[j] == 1:
                            tmp.append(i['input_context_scores'][j])
                            save_flag = True
                    if save_flag:
                        split_value_list.append(tmp)

                    if len(item['ais']) != len(split_value_list):
                        idx += 1
                        continue

                    cci_res += sort_index(split_value_list, 0, cci_topk)
                
                for i, n in enumerate(item['idxs']):
                    mark_nli = True if nli_pred[n] == 1 else False
                    mark_human = True if item['ais'][i] else False
                    mark_cci = True if cci_res[i] else False
                    mark_nli_inlang = True if nli_pred_inlang[n] == 1 else False

                    if item['ais'][i]:
                        pos_ins_cci.append(1 if mark_cci else 0)
                    else:
                        neg_ins_cci.append(1 if mark_cci else 0) 

                    if mark_nli == mark_cci:
                        correlation_nli += 1
                    correlation_nli_tot += 1
                    
                    if mark_nli == mark_human:
                        human_nli += 1
                    human_nli_tot += 1

                    if mark_human == mark_cci:
                        accuracy_cci += 1
                    accuracy_tot_cci += 1

                    if mark_nli_inlang == mark_human:
                        human_nli_inlang += 1
                    human_nli_inlang_tot += 1

            idx += 1

    print("=============")

    # Agreement results
    label_cci = np.array([1 for _ in range(len(pos_ins_cci))] + [0 for _ in range(len(neg_ins_cci))])
    predict_cci = np.array(pos_ins_cci + neg_ins_cci)
    roc_auc_cci = roc_auc_score(label_cci, predict_cci)

    print("pos_num:     {}".format(len(pos_ins_cci)))
    print("neg_num:     {}".format(len(neg_ins_cci)))
    print()
    print()
    print("ROC_AUC:     {}".format(roc_auc_cci))
    print("MIRAGE agreement with human:    {}/{}={}".format(accuracy_cci, accuracy_tot_cci, accuracy_cci/accuracy_tot_cci))
    #print("MIRAGE agreemtn with NLI_trans: {}/{}={}".format(correlation_nli, correlation_nli_tot, correlation_nli/correlation_nli_tot))
    print("NLI_origin agreement with human: {}/{}={}".format(human_nli_inlang, human_nli_inlang_tot, human_nli_inlang/human_nli_inlang_tot))
    print("NLI_trans agreement with human: {}/{}={}".format(human_nli, human_nli_tot, human_nli/human_nli_tot))
    print()

    #print("pos_mean:    {}".format(np.mean(pos_ins_cci)))
    #print("neg_mean:    {}".format(np.mean(neg_ins_cci)))
    #print("pos_median:  {}".format(np.median(pos_ins_cci)))
    #print("neg_median:  {}".format(np.median(neg_ins_cci)))

    #print("all_mean:    {}".format(np.mean(pos_ins_cci + neg_ins_cci)))
    #print("all_median:  {}".format(np.median(pos_ins_cci + neg_ins_cci)))

