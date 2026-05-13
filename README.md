# MHG-SSL
# Motif-Aware Hierarchical Graph Pretraining for Molecular Property Prediction

## 🚀Requirements
```
python                    3.8
torch                     2.3.0
torch-geometric           2.6.1
rdkit                     2020.09.1
numpy                     1.24.4
pandas                    2.0.3
scikit-learn              1.3.2
tqdm                      4.67.1
scipy                     1.10.1
```

## 📌Dataset
Download the downstream data from https://moleculenet.org/datasets-1, and save the .csv files in the ./finetune/dataset/[dataset_name]/raw/, where [dataset_name] is replaced by the downstream dataset name. For example, bace.csv is saved in './finetune/dataset/bace/raw/bace.csv'.

## 🔥Training
You can pretrain the model by
```
python pretrain.py
```

## 🌈Evaluation
You can evaluate the pretrained model by finetuning on downstream tasks
```
cd finetune
mkdir model_checkpoints
python finetune.py
```
