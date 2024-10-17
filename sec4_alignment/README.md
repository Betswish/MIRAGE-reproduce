## Steps for reproducing the human alignment experiments in Section 4

The results can be easily reproduced through **TWO** steps:

**Step 1** 

Run 
```
bash attribute-mirage.sh
```
to analysis model internals with MIRAGE for XOR-AttriQA-Full and XOR-AttriQA-Match. The results will be saved in ``internals-full/`` and ``internals-match/`` respectively.

**Step 2** 

Run 
```
python MIRAGE_match.py --CCI k
```
to get MIRAGE<sub>ex</sub> answer attribution on XOR-AttriQA-Match. Also, the scores of MIRAGE and NLI methods will be in the outputs.

- ``CCI``: CCI threshold. Using Top k strategy if k > 0; otherwise Top (-k)% if k < 0. Default ``-5``.

- ``val``: Specify ``--val`` to use use calibrated threshold for CTI filtering. ``python MIRAGE_match.py --val --CCI k`` enables you to get the results with MIRAGE<sub>cal</sub>.

**Extra 0**

In this repository, we have already provided the filtered dataset XOR-AttriQA-Match, which is stored in ``data/data-match/``. But we still provided the code for the filtering process:
```
cd data/
bash get_data.sh
```

**Extra 1**

For NLI prediction, we already provided the predictions in ``nli-predictions/``. But you could try it with the code
```
python baseline_nli.py
```
The result will be saved in the ``nli-predictions/`` folder.

**Extra 2**

For the experiments on the full dataset mentioned in the Appendix, run 
```
python MIRAGE_full.py --CCI k
```
to get MIRAGE answer attribution on XOR-AttriQA-Full. 

- similarly with **Step 2**, the ``CCI`` parameter is used for defining the CCI threshold, and ``--val`` parameter is for calibrating the CTI threshold with the validation set (i.e. ``python MIRAGE_full.py --val --CCI k``).






