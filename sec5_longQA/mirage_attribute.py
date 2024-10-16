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
import os

from utils import *
import inseq
from inseq.commands.attribute_context.attribute_context import AttributeContextArgs, attribute_context, attribute_context_with_model


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, help="Output data file")
    
    args = parser.parse_args()
    data = json.load(open(args.f))
    if not data["args"]["standard"]:
        save_dir = "./internal_selfcitation/"
    else:
        save_dir = "./internal_standard/"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load prompt
    prompt_data = json.load(open(data["args"]["prompt_file"]))

    # Load model
    model, tokenizer = load_model(data["args"]["model"])
    model_mirage = inseq.load_model(
            model,
            "saliency",
            model_kwargs={"device_map": 'cuda:0', "torch_dtype": torch.float16},
            tokenizer_kwargs={"use_fast": False},
    )

    stop = []
    stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
    if "llama-3" in data["args"]["model"].lower():
            stop_token_ids = list(set([tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [model.config.eos_token_id]))
    else:
            stop_token_ids = list(set([tokenizer._convert_token_to_id(stop_token) for stop_token in stop] + [model.config.eos_token_id]))
    if "llama" in data["args"]["model"].lower() or "zephyr" in data["args"]["model"].lower() or "mistral" in data["args"]["model"].lower():
        stop_token_ids.remove(tokenizer.unk_token_id)

    special_tokens_to_keep = []

    if "zephyr" in data["args"]["model"].lower():
        decoder_input_output_separator = '\n '
        special_tokens_to_keep = ["</s>"]
    elif "llama-2" in data["args"]["model"].lower():
        decoder_input_output_separator = ' '
    elif "mistral" in data["args"]["model"].lower():
        decoder_input_output_separator = ' '
    else:
        raise ValueError("model not supported yet")

    cache_dir = os.getenv("TMPDIR")

    num_empty = 0
    for idx, item in enumerate(tqdm(data['data'])):
        if item["output"] == "": 
            num_empty += 1
            continue

        item["output"] = item["output"].strip()
        for i in range(10):
            r_tmp = "\n" * (10-i)
            item["output"] = item["output"].replace(r_tmp, " ")
        doc_list = item['docs']

        input_context_text = "".join([make_doc_prompt(doc, doc_id, prompt_data["doc_prompt"], use_shorter=None) for doc_id, doc in enumerate(doc_list)])
        input_current_text = item['question']
        input_template = prompt_data["demo_prompt"].replace("{INST}", prompt_data["instruction"]).replace("{Q}", "{current}").replace("{A}</s>", "").replace("{A}", "").replace("{D}", "{context}").rstrip()
        contextless_input_current_text = input_template.replace("{context}", "")
        output_current_text = remove_citations(item["output"])

        if idx == 0:
            print("***********")
            print("input_context_text")
            print(input_context_text)
            print("***********")
            print("input_current_text")
            print(input_current_text)
            print("***********")
            print("input_template")
            print(input_template)
            print("***********")
            print("contextless_input_current_text")
            print(contextless_input_current_text)
            print("***********")
            print("output_current_text")
            print(output_current_text)
            print("***********")
            print("decoder_input_output_separator")
            print(decoder_input_output_separator)
            
        save_path = save_dir + data["args"]["model"].lower().replace('/','_') + '-shot' + str(data["args"]["shot"]) + '-seed' + str(data["args"]["seed"]) + '-' + str(idx) + '.json'
        lm_rag_prompting_example = AttributeContextArgs(
                model_name_or_path=data["args"]["model"],
                input_context_text=input_context_text,
                input_current_text=input_current_text,
                output_template="{current}",
                input_template=input_template,
                contextless_input_current_text=contextless_input_current_text,
                show_intermediate_outputs=False,
                attributed_fn="contrast_prob_diff",
                context_sensitivity_std_threshold=0,
                output_current_text=output_current_text,
                attribution_method="saliency",
                attribution_kwargs={"logprob": True},
                save_path=save_path,
                tokenizer_kwargs={"use_fast": False},
                model_kwargs={
                    "device_map": 'auto',
                    "torch_dtype": torch.float16,
                    "max_memory": get_max_memory(),
                    "load_in_8bit": False,
                    "cache_dir": cache_dir,
                    },
                generation_kwargs={
                    "do_sample": True,
                    "temperature": data["args"]["temperature"],
                    "top_p": data["args"]["top_p"],
                    "max_new_tokens": data["args"]["max_new_tokens"],
                    "num_return_sequences": 1,
                    "eos_token_id": stop_token_ids
                    },
                decoder_input_output_separator=decoder_input_output_separator,
                special_tokens_to_keep=special_tokens_to_keep,
                show_viz=False,
                )

        gen = attribute_context_with_model(lm_rag_prompting_example, model_mirage)
        #print(gen)
    print()

if __name__ == "__main__":
    main()
