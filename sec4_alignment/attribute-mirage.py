from inseq.commands.attribute_context.attribute_context import AttributeContextArgs, attribute_context
import os
import tqdm
import jsonlines
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--full", action= "store_true", help = "Attribute original XOR-AttriQA")
args = parser.parse_args()

if not args.full:
    dir_path_input = './data/data-match/'
    dir_path_output = './internals-match/'
else:
    dir_path_input = './data/data-full/'
    dir_path_output = './internals-full/'


model_path = 'gsarti/cora_mgen'


if not os.path.exists(dir_path_output):
    os.makedirs(dir_path_output)

files = os.listdir(dir_path_input)

for fname in files:
    if 'pos' not in fname: continue
    print('*****************************')
    print(fname)
    idx = 0
    with open(dir_path_input + fname) as f:
        for item in jsonlines.Reader(f):
            print("Current: {} - {}".format(fname, idx))
            #"<Q>: " + item['query'] + " <P>:" + passage
            save_path = dir_path_output + fname.split('.')[0].split('-')[1] + '-' + str(idx) + '.json'
            lm_rag_prompting_example = AttributeContextArgs(
                    model_name_or_path=model_path,
                    input_context_text=item['passage_in_language'],
                    input_current_text=f"<Q>: {item['query']}",
                    output_template="{current}",
                    input_template="{current} <P>:{context}",
                    show_intermediate_outputs=False,
                    attributed_fn="contrast_prob_diff",
                    output_current_text=item['prediction'],
                    context_sensitivity_std_threshold=0,
                    save_path=save_path,
                    generation_kwargs={"num_beams": 4, "min_length": 1, "max_length": 20, "early_stopping": False}
                    )

            attribute_context(lm_rag_prompting_example)

            idx += 1
