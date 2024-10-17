0. ``cd data/`` ``bash get_data.sh`` to get the filtered XOR-AttriQA-Match
1. run ``bash attribute-mirage.sh`` to analysis model internals with MIRAGE for XOR-AttriQA-Full and XOR-AttriQA-Match.
2. run ``python baseline_nli.py`` to get NLI predictions on XOR-AttriQA-Full
3. run ``python MIRAGE_match.py --CCI k`` to get MIRAGE answer attribution on XOR-AttriQA-Match. 

- CCI: CCI threshold. Using Top k strategy if k > 0; otherwise Top (-k)% if k < 0, default -5.

- use ``python MIRAGE_match.py --val --CCI k`` to use calibrated threshold for CTI filtering.

4. run ``python MIRAGE_full.py`` to get MIRAGE answer attribution on XOR-AttriQA-Full.
- similarly, use ``python MIRAGE_full.py --val`` to use calibrated threshold for CTI filtering.


