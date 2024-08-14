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

parser = argparse.ArgumentParser()
parser.add_argument("--val", action= "store_true", help = "Using the CTI threshold calibrated on validation set")

args = parser.parse_args()

if args.val:
    val = True
else:
    val = False

dir_path_input = './data/data-full/'
dir_path_attribute = './internals-full/'

# calibration thresholds for CTI
threshold = {'te': 8.509128727876837, 'bn': 6.088339890774511, 'ja': 8.90982038209497, 'fi': 12.430429453398368, 'ru': 11.73047786754601}

files = os.listdir(dir_path_input)
for fname in files:
    if 'pos' not in fname: continue
    if 'val' in fname or 'train' in fname: continue

    lang = fname.split('.')[0].split('-')[1]
    print('*****************************')
    print(lang)
    idx = 0

    pos_ins = []
    neg_ins = []

    accuracy = 0
    accuracy_tot = 0
    
    with open(dir_path_input + fname) as f:
        for item in jsonlines.Reader(f):
            #"<Q>: " + item['query'] + " <P>:" + passage
            # Get the highest CTI score for each instance
            save_path = dir_path_attribute + lang + '-' + str(idx) + '.json'
            with open(save_path) as r:
                res_pecora = json.load(r)
            
            if val:
                threshold_CTI = threshold[item['query_language']]
            else:
                threshold_CTI = 1.01 * np.mean(res_pecora["cti_scores"]) + 2.8 * np.std(res_pecora["cti_scores"])
            mark_attri = False
            for i in res_pecora["cti_scores"]:
                if i >= threshold_CTI:
                    mark_attri = True

            score = np.max(res_pecora["cti_scores"])

            if item['ais']:
                pos_ins.append(score)
            else:
                neg_ins.append(score)
                
            if mark_attri == item['ais']:
                accuracy += 1
            accuracy_tot += 1

            idx += 1

    label = np.array([1 for _ in range(len(pos_ins))] + [0 for _ in range(len(neg_ins))])
    predict = np.array(pos_ins+neg_ins)
    roc_auc = roc_auc_score(label, predict)

    print()
    print("Accuracy:    {}/{}={}".format(accuracy, accuracy_tot, accuracy/accuracy_tot))
    print("ROC AUC:     {}".format(roc_auc))
    print()
    print("=============")
