import argparse
import ast
import logging
import os
import sys
import json
import jsonlines

import pandas as pd
import torch
from tqdm import tqdm

from transformers import BartForConditionalGeneration, MT5ForConditionalGeneration, AutoTokenizer
from transformers import logging as transformers_logging


sys.path.append(os.path.join(os.getcwd()))  # noqa: E402 # isort:skip

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

transformers_logging.set_verbosity_info()

def main():
    model_class = MT5ForConditionalGeneration

    checkpoint = "gsarti/cora_mgen"
    logger.info("Evaluate the following checkpoints: %s", checkpoint)

    model = model_class.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
   
    
    #dir_path = './xor_attriqa/in-language/'
    dir_path = './xor_attriqa/concat-in-language/'

    files = os.listdir(dir_path)
    
    for fname in files:
        print('*****************************')
        print(fname)

        with open(dir_path + fname) as f:
            count = 0
            
            num = 0
            num_match = 0
            pos_case = []
            neg_case = []
            comp_case = []
            pos_idx = []
            neg_idx = []

            for item in tqdm(jsonlines.Reader(f)):
                if item['passage_retrieved_language'] == 'en':
                    passage = item['passage_en']
                else:
                    passage = item['passage_in_language']
                
                # Template0: "<Q>: {0} <P>:{1}"
                data = "<Q>: " + item['query'] + " <P>:" + passage
                data_list = [data]

                count += 1
                with torch.no_grad():
                    inputs_dict = tokenizer.batch_encode_plus(data_list, return_tensors="pt", padding=True, truncation=True)
                    input_ids = inputs_dict.input_ids.to(device)
                    attention_mask = inputs_dict.attention_mask.to(device)
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        num_beams=4,
                        min_length=1,
                        max_length=20,
                        early_stopping=False,
                        num_return_sequences=1,
                        output_scores=False,
                        return_dict_in_generate=False
                    )
        
                    answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    
                    if answers[0] == item['prediction']: 
                        num += 1
                        pos_case.append(item)
                        pos_idx.append(count)
                    else:
                        neg_case.append(item)
                        neg_idx.append(count)
                        comp_case.append("Answer: " + answers[i] + ', XOR-AttriQA Answer: ' + item['prediction'])

                    if item['prediction'] in answers[i]:
                        num_match += 1
            print(num)
            print(num_match)
            print(count)

            with jsonlines.open('./record/pos-' + fname, 'a') as f:
                for i in pos_case:
                    f.write(i)
            with jsonlines.open('./record/neg-' + fname, 'w') as f:
                for i in neg_case:
                    f.write(i)

if __name__ == "__main__":
    main()
