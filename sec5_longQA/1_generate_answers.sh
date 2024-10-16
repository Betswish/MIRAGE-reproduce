# Experiments in main paper with self-citation prompts
python run.py --config configs/eli5_zephyr_shot0_ndoc5_bm25_selfcitation.yaml --seed 42
python run.py --config configs/eli5_zephyr_shot0_ndoc5_bm25_selfcitation.yaml --seed 43
python run.py --config configs/eli5_zephyr_shot0_ndoc5_bm25_selfcitation.yaml --seed 44

python run.py --config configs/eli5_llama2_7b_shot0_ndoc5_bm25_selfcitation.yaml --seed 42
python run.py --config configs/eli5_llama2_7b_shot0_ndoc5_bm25_selfcitation.yaml --seed 43
python run.py --config configs/eli5_llama2_7b_shot0_ndoc5_bm25_selfcitation.yaml --seed 44

# Extended experiments with standard prompt in appendix
python run.py --config configs/eli5_zephyr_shot0_ndoc5_bm25_standard.yaml --seed 42
python run.py --config configs/eli5_zephyr_shot0_ndoc5_bm25_standard.yaml --seed 43
python run.py --config configs/eli5_zephyr_shot0_ndoc5_bm25_standard.yaml --seed 44

python run.py --config configs/eli5_llama2_7b_shot0_ndoc5_bm25_standard.yaml --seed 42
python run.py --config configs/eli5_llama2_7b_shot0_ndoc5_bm25_standard.yaml --seed 43
python run.py --config configs/eli5_llama2_7b_shot0_ndoc5_bm25_standard.yaml --seed 44
