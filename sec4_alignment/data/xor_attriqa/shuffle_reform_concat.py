
import os
import json
import jsonlines
import tqdm
import random

dir_path_input = './in-language/'
dir_path_output = './concat-in-language/'

files = os.listdir(dir_path_input)

for fname in files:
    print('*****************************')
    print(fname)
    
    shuffled_idxs = []
    with open('../record/neg-' + fname) as f:
        for item in jsonlines.Reader(f):
            idxs = item['idxs'].copy()
            while idxs == item['idxs'] and len(idxs) > 1:
                random.shuffle(idxs)
            shuffled_idxs.append(idxs)

    print("log: idxs shuffled!")
    items = []
    with open(dir_path_input + fname) as f:
        for item in jsonlines.Reader(f):
            items.append(item)

    print("log: data files read!")
    
    records = []
    avg_len = 0
    for idxs in shuffled_idxs:
        record = {
                'query': items[idxs[0]]['query'],
                'query_language': items[idxs[0]]['query_language'],
                'answers': items[idxs[0]]['answers'],
                'prediction': items[idxs[0]]['prediction'],
                'prediction_correct': items[idxs[0]]['prediction_correct'],
                'passage_in_language': "", 
                'passage_retrieved_language': 'multi',
                'split_passages_en': [],
                'intrepetability_vote': [],
                'ais_vote': [],
                'intrepetability': [],
                'ais': [],
                'idxs': idxs
                }


        for idx in idxs:
            if items[idx]['passage_retrieved_language'] == 'en':
                passage = items[idx]['passage_en']
            else:
                passage = items[idx]['passage_in_language']
            
            if len(record['passage_in_language']) > 0 and record['passage_in_language'][-1] != ' ':
                record['passage_in_language'] += ' '

            num_passages = len(record['ais'])
            record['passage_in_language'] += '<' + str(num_passages) + '> ' + passage
            record['split_passages_en'].append(items[idx]['passage_en'])
            record['intrepetability_vote'].append(items[idx]['intrepetability_vote'])
            record['ais_vote'].append(items[idx]['ais_vote'])
            record['intrepetability'].append(items[idx]['intrepetability'])
            record['ais'].append(items[idx]['ais'])

        records.append(record)
        avg_len += len(idxs)
    
    avg_len /= len(shuffled_idxs)

    print("log: new data proceeded! length: " + str(len(records)))
    print("log: avg index length" + str(avg_len))

    with jsonlines.open(dir_path_output + fname, 'w') as f:
        for v in records:
            f.write(v)

    print("log: file written!")
