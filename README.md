# PriFold: 

## Introduction
This is the official Pytorch implement of PriFold, a deep-learning-based RNA secondary structure prediction method.
Predicting RNA secondary structures is crucial for understanding RNA function, designing RNA-based therapeutics, and studying molecular interactions within cells. Existing deep-learning-based methods for RNA secondary structure prediction have mainly focused on local structural properties, often overlooking the global characteristics and evolutionary features of RNA sequences. Guided by biological priors, we propose PriFold, incorporating two key innovations: 1) improving attention mechanism with pairing probabilities to utilize global pairing characteristics, and 2) implementing data augmentation based on RNA covariation to leverage evolutionary information. Our structured enhanced pretraining and finetuning strategy significantly optimizes model performance. Extensive experiments demonstrate that PriFold achieves state-of-the-art results in RNA secondary structure prediction on benchmark datasets such as bpRNA, RNAStrAlign and ArchiveII. These results not only validate our prediction approach but also highlight the potential of integrating biological priors, such as global characteristics and evolutionary information, into RNA structure prediction tasks, opening new avenues for research in RNA biology and bioinformatics.

This repo contain code for pretrained model in `/tiny_llama2`, and code for secondary structure prediction module in `/utils`

## Trained models
Our pretrained RNA language model and secondary structure prediction model can be downloaded from the following link: [https://huggingface.co/yfish/PriFold](https://huggingface.co/yfish/PriFold)

### Dataset
The datasets used in our work (bpRNA, RNAStrAlign, and ArchiveII) can be acquired through the following link: [https://github.com/mxfold/mxfold2/releases/tag/v0.1.0](https://github.com/mxfold/mxfold2/releases/tag/v0.1.0)

## Inference
To run the inference script, run:

    ./inference.sh

## Train
To run the training script, run:

    ./run_train.sh
