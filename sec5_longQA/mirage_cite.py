import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from nltk import sent_tokenize
import re
import numpy as np
import string
import torch
from searcher import SearcherWithinDocs

import pandas as pd
from transformers import AutoTokenizer
from utils import *
import inseq
from inseq.commands.attribute_context.attribute_context import AttributeContextArgs, attribute_context, attribute_context_with_model


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def mirage_cite(res_mirage, cti_threshold, start_pos_sent, end_pos_sent, topk_CCI, doc_seps):
    res = []

    sum_weight = 0
    sum_value = np.zeros(len(res_mirage['input_context_tokens']))
    
    for i in res_mirage['cci_scores']:
        # CTI Filtering
        if not (i["cti_idx"] >= start_pos_sent and i["cti_idx"] < end_pos_sent): continue
        if i['cti_score'] >= cti_threshold:
            # CCI Focus
            CCI_value = np.array(i['input_context_scores'])
            if topk_CCI == 0:
                cci_threshold = np.mean(CCI_value)
            elif topk_CCI < 0:
                cci_threshold = (1+topk_CCI/100) * np.max(CCI_value) - topk_CCI/100 * np.min(CCI_value)
            else:
                cci_threshold = np.sort(CCI_value)[-topk_CCI]
            zero_idx = CCI_value < cci_threshold
            CCI_value[zero_idx] = 0

            sum_value += CCI_value

        if i['cti_score'] < cti_threshold: break

    sum_tmp = 0
    for i, v in enumerate(sum_value):
        sum_tmp += v
        if doc_seps[i] or (i == len(sum_value)-1): # meet '\n'
            res.append(sum_tmp)
            sum_tmp = 0
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, help="Output data file")
    
    parser.add_argument("--CTI", type=int, default=1, help="CTI filtering strategy: How many standard deviations over average")
    parser.add_argument("--CCI", type=int, default=-5, help="CCI filtering strategy: Top k if k > 0; Top (-k)% if k < 0")

    # CTI and CCI strategies
    #topk_CTI = 1 # 1 means over average+1SD
    #topk_CTI = 0 # 0 means over average

    #topk_CCI = -5 # -5 means range top5%
    #topk_CCI = 0
    #topk_CCI = 3 # 3 means top 3

    cite_idx_acs = False
    
    args = parser.parse_args()
    data = json.load(open(args.f))

    topk_CTI = args.CTI
    topk_CCI = args.CCI

    if not data["args"]["standard"]:
        load_dir = "./internal_selfcitation/"
    else:
        load_dir = "./internal_standard/"

    new_data = []

    prefix = load_dir + data["args"]["model"].lower().replace('/','_')+'-shot'+str(data["args"]["shot"])+'-seed'+str(data["args"]["seed"]) 

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data["args"]["model"], use_fast=False)

    # Fix OPT bos token problem in HF
    if "opt" in data["args"]["model"]:
        tokenizer.bos_token = "<s>"
    tokenizer.padding_side = "left"
    
    num_empty = 0
    for idx, item in enumerate(tqdm(data['data'])):
        if item["output"] == "": 
            new_data.append(item)
            num_empty += 1
            continue
    
        item["output"] = item["output"].strip()
        for i in range(10):
            r_tmp = "\n" * (10-i)
            item["output"] = item["output"].replace(r_tmp, " ")

        output = remove_citations(item["output"])

        # read MIRAGE attribute results
        read_path = prefix + '-'+str(idx)+'.json'
        with open(read_path) as r:
            res_mirage = json.load(r)

        if topk_CTI >= 0:
            cti_threshold = np.mean(res_mirage["cti_scores"]) + topk_CTI * np.std(res_mirage["cti_scores"])
        else:
            raise ValueError('CTI filtering parameter should be equal or larger than 0.')
        
        if "qampari" in args.f:
            sents = [item['question'] + ' ' + x.strip() for x in item['output'].rstrip(".").split(",")]
        else:
            sents = sent_tokenize(output)
        # check num and index of '\n' (i.e. <0x0A> in Llama, zephyr, mistral)
        # num should constantly be 5
        doc_seps = np.array(res_mirage["input_context_tokens"])
        doc_seps = doc_seps == '<0x0A>'
        #num_doc = pd.value_counts(res_mirage["input_context_tokens"])["<0x0A>"]
        
        new_output = ""
        start_pos_sent = 0
        end_pos_sent = 0
        print("\n\n")
        print("="*5)
        print(item['prompt'])
        print(item['output'])
        for sent in sents:
            # e.g. [1,3,4]
            original_ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] 
            end_pos_sent = start_pos_sent + len(tokenizer.tokenize(sent))
            
            # e.g. [0, 0, 20, 3, 0]; always length == 5
            cite_result_mirage = mirage_cite(res_mirage, cti_threshold, start_pos_sent, end_pos_sent, topk_CCI, doc_seps)
            #print(cite_result_mirage)
            #print()
            start_pos_sent = end_pos_sent

            if len(cite_result_mirage) >= 0:
                print("\n-----")
                print("Original sentence:", sent)
                print("Original ref:", original_ref)
                sent = remove_citations(sent)
               
                best_doc_id_tmp = {i: v for i, v in enumerate(cite_result_mirage) if v}
                best_doc_id = list(dict(sorted(best_doc_id_tmp.items(), key=lambda item: item[1], reverse=True)).keys())

                if cite_idx_acs:
                    best_doc_id = sorted(best_doc_id)

                print("New ref:", best_doc_id)
                best_doc_id_str = ""
                for i in best_doc_id:
                    best_doc_id_str += "[" + str(i+1) + "]"
                sent = best_doc_id_str + " " + sent
                print("New sentence:", sent)
            
            if "qampari" in args.f:
                new_output += sent.replace(item['question'], '').strip() + ", "
            else:
                new_output += sent + " "

        item['output'] = new_output.rstrip().rstrip(",")
        print("\n-----")
        print("Final output: " + item['output'])
        new_data.append(item)

    print("num_empty:")
    print(num_empty)
    print()
    data['data'] = new_data 
    
    tag = f".mirage_cite"     
    tag += "_CTI_" + str(topk_CTI)
    tag += "_CCI_" + str(topk_CCI)

    if cite_idx_acs:
        tag += '_acs'

    json.dump(data, open(args.f + f"{tag}", 'w'), indent=4)

if __name__ == "__main__":
    main()
