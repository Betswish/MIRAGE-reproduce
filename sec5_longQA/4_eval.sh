
# Experiments in main paper with self-citation prompts
# Also, here we evaluate the fluency and correctness of the answers through --mauve and --claims_nli respectively
# These two scores are reported in Appendix Table 8

## Self-citation results
python eval.py --f result/selfcitation/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-42.json --citations --claims_nli --mauve
python eval.py --f result/selfcitation/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-43.json --citations --claims_nli --mauve
python eval.py --f result/selfcitation/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-44.json --citations --claims_nli --mauve

python eval.py --f result/selfcitation/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-42.json --citations --claims_nli --mauve
python eval.py --f result/selfcitation/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-43.json --citations --claims_nli --mauve
python eval.py --f result/selfcitation/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-44.json --citations --claims_nli --mauve

## MIRAGE restuls
python eval.py --f result/selfcitation/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-42.json.mirage_cite_CTI_1_CCI_-5 --citations --claims_nli --mauve
python eval.py --f result/selfcitation/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-43.json.mirage_cite_CTI_1_CCI_-5 --citations --claims_nli --mauve
python eval.py --f result/selfcitation/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-44.json.mirage_cite_CTI_1_CCI_-5 --citations --claims_nli --mauve

python eval.py --f result/selfcitation/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-42.json.mirage_cite_CTI_1_CCI_-5 --citations --claims_nli --mauve
python eval.py --f result/selfcitation/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-43.json.mirage_cite_CTI_1_CCI_-5 --citations --claims_nli --mauve
python eval.py --f result/selfcitation/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-44.json.mirage_cite_CTI_1_CCI_-5 --citations --claims_nli --mauve

# Extended experiments with standard prompt in appendix
python eval.py --f result/standard/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-42.json.mirage_cite_CTI_1_CCI_-5 --citations --claims_nli --mauve
python eval.py --f result/standard/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-43.json.mirage_cite_CTI_1_CCI_-5 --citations --claims_nli --mauve
python eval.py --f result/standard/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-44.json.mirage_cite_CTI_1_CCI_-5 --citations --claims_nli --mauve

python eval.py --f result/standard/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-42.json.mirage_cite_CTI_1_CCI_-5 --citations --claims_nli --mauve
python eval.py --f result/standard/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-43.json.mirage_cite_CTI_1_CCI_-5 --citations --claims_nli --mauve
python eval.py --f result/standard/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-44.json.mirage_cite_CTI_1_CCI_-5 --citations --claims_nli --mauve
