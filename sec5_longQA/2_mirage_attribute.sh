# Experiments in main paper with self-citation prompts
python mirage_attribute.py --f result/selfcitation/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-42.json
python mirage_attribute.py --f result/selfcitation/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-43.json
python mirage_attribute.py --f result/selfcitation/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-44.json

python mirage_attribute.py --f result/selfcitation/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-42.json
python mirage_attribute.py --f result/selfcitation/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-43.json
python mirage_attribute.py --f result/selfcitation/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-44.json

# Extended experiments with standard prompt in appendix
python mirage_attribute.py --f result/standard/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-42.json 
python mirage_attribute.py --f result/standard/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-43.json 
python mirage_attribute.py --f result/standard/eli5-zephyr-7b-beta-bm25-shot0-ndoc5-44.json 

python mirage_attribute.py --f result/standard/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-42.json 
python mirage_attribute.py --f result/standard/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-43.json 
python mirage_attribute.py --f result/standard/eli5-Llama-2-7b-chat-hf-bm25-shot0-ndoc5-44.json 
