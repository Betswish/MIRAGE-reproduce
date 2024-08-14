0. ``cd data/`` ``bash get_data.sh`` to get the filtered XOR-AttriQA-Match
1. run ``python attribute-mirage.py`` to analysis model internals with MIRAGE for XOR-AttriQA-Full and XOR-AttriQA-Match.
2. run ``python baseline_nli.py`` to get NLI predictions on XOR-AttriQA-Full
3. run ``python MIRAGE_match.py --CCI k`` to get MIRAGE answer attribution on XOR-AttriQA-Match. 
python MIRAGE_match.py --val --CCI k

val: Using calibrated threshold for CTI filtering.
CCI: CCI threshold. Using Top k strategy if k > 0; otherwise Top (-k)% if k < 0, default -5.

4. run ``python MIRAGE_full.py`` to get MIRAGE answer attribution on XOR-AttriQA-Full.
python MIRAGE_full.py --val
val: Using calibrated threshold for CTI filtering.


