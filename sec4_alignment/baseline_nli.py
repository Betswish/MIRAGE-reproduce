import argparse
import collections
import json
import re
import string
import torch
import copy
import os
import tqdm
import jsonlines
from nltk import sent_tokenize
import numpy as np
from tqdm import tqdm
import sys
from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        pipeline
        )

def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    print(max_memory)
    return max_memory


AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"

autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

print('vector for 0: {}'.format(autoais_tokenizer.encode('0')))
print('vector for 1: {}'.format(autoais_tokenizer.encode('1')))


dir_path_input = './data/data-full/'
dir_path_output = './nli-predictions/'

files = os.listdir(dir_path_input)

for fname in files:
    if 'pos' not in fname: continue
    print('*****************************')
    lang = fname.split('.')[0].split('-')[1]
    print(lang)
    
    save_path = dir_path_output + lang + '-nli-translate-en.txt'
    predictions = []
    idx = 0
    with open(dir_path_input + fname) as f:
        for item in jsonlines.Reader(f):
            print("Current: {} - {}".format(lang, idx))
            #"<Q>: " + item['query'] + " <P>:" + passage
            pred = _run_nli_autoais(item['passage_en'], item['prediction_translated_en'])
            predictions.append(pred)
            
            idx += 1

    with open(save_path, 'w') as f:
        f.write(str(predictions))

#example:
#passage = 'I have an apple.'
#claim = 'I have an orange.'
#_run_nli_autoais(passage, claim)

