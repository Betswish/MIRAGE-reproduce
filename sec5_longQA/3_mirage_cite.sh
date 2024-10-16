# Experiments in main paper with self-citation prompts
python mirage_cite.py --f result/selfcitation/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-42.json --CTI 1 --CCI -5
python mirage_cite.py --f result/selfcitation/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-43.json --CTI 1 --CCI -5
python mirage_cite.py --f result/selfcitation/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-44.json --CTI 1 --CCI -5

python mirage_cite.py --f result/selfcitation/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-42.json --CTI 1 --CCI -5
python mirage_cite.py --f result/selfcitation/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-43.json --CTI 1 --CCI -5
python mirage_cite.py --f result/selfcitation/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-44.json --CTI 1 --CCI -5

# Extended experiments with standard prompt in appendix 
python mirage_cite.py --f result/standard/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-42.json --CTI 1 --CCI -5
python mirage_cite.py --f result/standard/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-43.json --CTI 1 --CCI -5
python mirage_cite.py --f result/standard/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-44.json --CTI 1 --CCI -5

python mirage_cite.py --f result/standard/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-42.json --CTI 1 --CCI -5
python mirage_cite.py --f result/standard/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-43.json --CTI 1 --CCI -5
python mirage_cite.py --f result/standard/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-44.json --CTI 1 --CCI -5
