# MMTAGE
MMTAGE is a gene-level multimodal representation learning framework dedicated to the integrated analysis of single-cell RNA-seq (scRNA-seq) and T cell receptor-seq (scTCR-seq) data, featuring GEX unimodal, TCR unimodal and multimodal fusion modules to capture gene-TCR associations while preserving the inherent biological properties of each modality. It generates two tailored multimodal embeddings (mmMean and mmCLS): mmMean balances the capture of antigen specificity and gene expression functional signals, while mmCLS maximizes the retention of high-fidelity transcriptomic profiles, flexibly adapting to diverse downstream analytical needs.

# Dependencies
MMTAGE is writen in Python based on Pytorch. The required software dependencies are listed below:
```
torch=2.0.0
cuda=11.8.0
python=3.10
einops=0.8.1
local_attention=1.10.0
timm=1.0.24
iopath=0.1.10
omegaconf=2.3.0
transformers=4.46.3
scanpy=1.9.8
```

# Installation
Setup conda environment:
```
# Create environment
conda create -n MMTAGE python=3.10 -y
conda activate MMTAGE

# Instaill requirements
conda install pytorch==2.0.0 -c pytorch -y
pip install einops==0.8.1
pip install local_attention==1.10.0
pip install timm==1.0.24
pip install iopath==0.1.10
pip install omegaconf==2.3.0
pip install transformers==4.46.3
pip install scanpy==1.9.8

# Clone MMTAGE
git clone https://github.com/Xiaobei-debug/MMTAGE.git
cd MMTAGE
```

# Data
All the data used in the paper were collected from public databases: 10X, Gene Expression Omnibus (GEO), VDJdb, IEDB, McPAS-TCR, and PIRD.

# Usage of MMTAGE
Data Preparation:
Prepare the train dataset in <BASE_FOLDER>/data/.

Training MMTAGE with XX dataset:
```
python train_model.py --device 'cuda' --epochs 20 --batch_size 2 --lr 0.0001 --save_dir "./checkpoints/" --dataset "./data/XX.h5ad"
```
Extract features from MMTAGE:
```
python train_model.py --extract_feat True --load_model "./checkpoints/checkpoint19.pt"
```

Corresponding embeddings are saved in multimodal_emb_cls.h5ad and multimodal_emb_mean.h5ad
