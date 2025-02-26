<div align="center">
  <img src="fig/mirage_logo.png" width="400"/> 
  <h4> Toward faithful answer attribution with model internals 🌴 </h4> 
</div>
<br/>
<div align="center">

Authors (_* Equal contribution_): [Jirui Qi*](https://betswish.github.io/) • [Gabriele Sarti*](https://gsarti.com/) • [Raquel Fernández](https://staff.fnwi.uva.nl/r.fernandezrovira/) • [Arianna Bisazza](https://www.cs.rug.nl/~bisazza/)  
</div>

>[!TIP]
> This is the repository for reproducing all experimental results in our [MIRAGE paper](https://aclanthology.org/2024.emnlp-main.347/), accepted by the [EMNLP 2024](https://2024.emnlp.org/) Main Conference.
> Also, check our demo [here](https://huggingface.co/spaces/gsarti/mirage)!

If you find the paper helpful and use the content, we kindly suggest you cite through:
```bibtex
@inproceedings{qi-etal-2024-model,
    title = "Model Internals-based Answer Attribution for Trustworthy Retrieval-Augmented Generation",
    author = "Qi, Jirui  and
      Sarti, Gabriele  and
      Fern{\'a}ndez, Raquel  and
      Bisazza, Arianna",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.347/",
    doi = "10.18653/v1/2024.emnlp-main.347",
    pages = "6037--6053",
    abstract = "Ensuring the verifiability of model answers is a fundamental challenge for retrieval-augmented generation (RAG) in the question answering (QA) domain. Recently, self-citation prompting was proposed to make large language models (LLMs) generate citations to supporting documents along with their answers. However, self-citing LLMs often struggle to match the required format, refer to non-existent sources, and fail to faithfully reflect LLMs' context usage throughout the generation. In this work, we present MIRAGE {--} Model Internals-based RAG Explanations {--} a plug-and-play approach using model internals for faithful answer attribution in RAG applications. MIRAGE detects context-sensitive answer tokens and pairs them with retrieved documents contributing to their prediction via saliency methods. We evaluate our proposed approach on a multilingual extractive QA dataset, finding high agreement with human answer attribution. On open-ended QA, MIRAGE achieves citation quality and efficiency comparable to self-citation while also allowing for a finer-grained control of attribution parameters. Our qualitative evaluation highlights the faithfulness of MIRAGE`s attributions and underscores the promising application of model internals for RAG answer attribution. Code and data released at https://github.com/Betswish/MIRAGE."
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

  
