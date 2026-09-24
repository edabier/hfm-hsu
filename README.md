<h1>Benchmarking Hyperspectral Foundation Models for Hyperspectral Unmixing</h1>

[Dabier Edgard](https://github.com/edabier)<sup>* </sup>, [Tupin Florence](https://scholar.google.com/citations?user=j9x2t78AAAAJ&hl=fr&oi=ao)<sup>* </sup>, [Kervazo Christophe](https://scholar.google.com/citations?hl=fr&user=MePhLXIAAAAJ)<sup>*</sup>, [Gori Pietro](https://scholar.google.com/citations?hl=fr&user=id9wCjsAAAAJ)<sup> *</sup>

<sup>*</sup> LTCI, Télécom Paris, Institut Polytechnique de Paris, Palaiseau, France

<div align="center">
<!-- <a href="link"><img alt="Pape" src="https://img.shields.io/badge/GRSL-Paper-0068F0?style=flat-square&logo=ieee" /></a> -->
<a href="link"><img alt="Pape" src="https://img.shields.io/badge/ArXiv-preprint-AA142D?style=flat-square&logo=arxiv" /></a>
</div>

# Overview

This is the official library accompanying the article "**Benchmarking Hyperspectral Foundation Models for Hyperspectral Unmixing**".  
It introduces a model-agnostic framework to perform blind Hyperspectral Unmixing (HSU) on any Hyperspectral Foundation Model (HFM).  
The framework is illustrated bellow. 

<div align="center">
<img src=Fig/hfm-hsu.png width="80%">

Figure 1. Unmixing from HFM features framework
</div>

This library currently supports **DOFA**<sup>1</sup>, **HyperFree**<sup>2</sup>, **HyperSIGMA**<sup>3</sup>, **HyperSL**<sup>4</sup>, **Panopticon**<sup>5</sup>,  **SpecAware**<sup>6</sup>, **SpecViT**<sup>7</sup> and **UniverSat**<sup>8</sup>

# Requirements

Install the following dependencies by running:

```console
conda create --name hfm-hsu-env python
conda activate hfm-hsu-env
pip install -r requirements.txt
```

# Getting started

To start working with HFMs, clone the desired model repository, download the [pretrained weights](pretrained_models) and place it in the corresponding [folder](pretrained_models).  
Each model might require additional dependencies that have to be installed following their documentation for proper operation.  
The `example.ipynb` file can be used straight away to perform unmixing on one of the available dataset (Samson, Jasper, Apex and Urban), using the selected HFM.

# Citation

If you find our benchmarking tool helpful, please cite it as follows:

```
@misc{dabier_2026,
    title={Benchmarking Hyperspectral Foundation Models for Hyperspectral Unmixing}, 
    author={Edgard Dabier and Christophe Kervazo and Pietro Gori and Florence Tupin},
    year={2026},
    eprint={2609.28283},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2609.28283}, 
}
```

# Aknowledgments
This project is based on [CNNAEU](https://ieeexplore.ieee.org/document/9096565) and [DeepTrans](https://github.com/preetam22n/DeepTrans-HSU), and it uses [DOFA](https://github.com/zhu-xlab/DOFA)<sup>1</sup>, [HyperFree](https://github.com/Jingtao-Li-CVer/HyperFree)<sup>2</sup>, [HyperSIGMA](https://github.com/WHU-Sigma/HyperSIGMA/tree/main)<sup>3</sup>, [HyperSL](https://github.com/kkweil/HyperSL)<sup>4</sup>, [Panopticon](https://github.com/Panopticon-FM/panopticon)<sup>5</sup>, [SpecAware](https://github.com/busbyjrj/SpecAware)<sup>6</sup>, [SpecViT](https://github.com/AABNassim/spectral_earth)<sup>7</sup> and [UniverSat](https://github.com/gastruc/UniverSat)<sup>8</sup>.<br>