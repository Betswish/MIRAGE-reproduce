<div align="center">
  <img src="fig/mirage_logo.png" width="400"/> 
  <h4> Toward faithful answer attribution with model internals 🌴 </h4> 
</div>
<br/>
<div align="center">

Authors (_* Equal contribution_): [Jirui Qi*](https://betswish.github.io/) • [Gabriele Sarti*](https://gsarti.com/) • [Raquel Fernández](https://staff.fnwi.uva.nl/r.fernandezrovira/) • [Arianna Bisazza](https://www.cs.rug.nl/~bisazza/)  
</div>

>[!TIP]
> This is the repository for reproducing all experimental results in our [MIRAGE paper](https://arxiv.org/abs/2406.13663), accepted by the [EMNLP 2024](https://2024.emnlp.org/) Main Conference.
>If you find the paper helpful and use the content, we kindly suggest you cite through:
```bibtex
@inproceedings{Qi2024ModelIA,
  title={Model Internals-based Answer Attribution for Trustworthy Retrieval-Augmented Generation},
  author={Jirui Qi and Gabriele Sarti and Raquel Fern'andez and Arianna Bisazza},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:270619780}
}
```


## Environment: 
For a quick start, you may load our environment easily with Conda:
```
conda env create -f MIRAGE.yaml
```

Alternatively, you can install all packages by yourself:

Python: `3.9.19`

Packages: `pip install -r requirements.txt`

## Reproduction of the alignment with human annotations (Experiments in Section 4)
The code is in the folder ``sec4_alignment``. See more detailed instructions in the README.MD there.

## Reproduction of citation generation and comparison with self-citation on long-form QA dataset ELI5 (Experiments in Section 5)
The code is in the folder ``sec5_longQA``. See more detailed instructions in the README.MD there.

  
