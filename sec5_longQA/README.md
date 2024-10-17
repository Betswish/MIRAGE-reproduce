## Steps for reproducing the citation experiments on long-form QA dataset ELI5 in Section 5

The results can be easily reproduced through **FIVE** steps:

**Step 1** 

Run 
```
bash 0_download_data.sh
```
for downloading the datasets, which will be saved in the ``data/`` folder.

**Step 2** 

Run 
```
bash 1_generate_answers.sh
```
to get the LLMs' original responses to the questions in the dataset. 
The generations with the self-citation instruction (Sec5) and standard instruction (Appendix)
will be saved in ``result/selfcitation`` and ``result/standard`` respectively.

**Step3**

Run
```
bash 2_mirage_attribute.sh
```
to get the model internals analyzed by MIRAGE. These intermediate results will be saved in ``./internal_selfcitation/`` or ``./internal_standard/`` according to the adopted instructions.

**Step 4**

Run
```
bash 3_mirage_cite.sh
```
to generate citations with the model internals obtained by MIRAGE. Two parameters can be specified here:

- ``CTI``: The parameter for the threshold of CTI selection. It means how many standard deviations are added to the mean CTI scores. Default ``1``. 
- ``CCI``: The parameter for the threshold of CCI selection. It means Top k selection strategy if k > 0; otherwise Top (-k)% strategy if k < 0. Default ``-5``.

**Step 5**

Run
```
bash 4_eval.sh
```
to evaluate the performance of both MIRAGE and self-citation. Besides citation quality, we also evaluate the fluency and correctness, as shown in Table 8 in the appendix.
