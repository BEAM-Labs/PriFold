# PriFold: Biological Priors Improve RNA Secondary Structure Predictions

## 🔬 Introduction
This is the official PyTorch implementation of PriFold, a deep-learning-based RNA secondary structure prediction method.

Predicting RNA secondary structures is crucial for understanding RNA function, designing RNA-based therapeutics, and studying molecular interactions within cells. Existing deep-learning-based methods for RNA secondary structure prediction have mainly focused on local structural properties, often overlooking the global characteristics and evolutionary features of RNA sequences.

Guided by biological priors, we propose PriFold, incorporating two key innovations:
1. Improving attention mechanism with pairing probabilities to utilize global pairing characteristics
2. Implementing data augmentation based on RNA covariation to leverage evolutionary information

Our structured enhanced pretraining and finetuning strategy significantly optimizes model performance. Extensive experiments demonstrate that PriFold achieves state-of-the-art results in RNA secondary structure prediction on benchmark datasets such as bpRNA, RNAStrAlign and ArchiveII.

## 🛠️ Installation

```bash
git clone https://github.com/BEAM-Labs/PriFold.git
cd PriFold

conda create -n prifold python=3.10
conda activate prifold
pip install -r requirements.txt
```

## 🔧 Resources

### Trained Models
Our pretrained RNA language model and secondary structure prediction model are available at:
[https://huggingface.co/yfish/PriFold](https://huggingface.co/yfish/PriFold)

### Dataset
The datasets used in our work (bpRNA, RNAStrAlign, and ArchiveII) are available at:
[https://huggingface.co/yfish/PriFold](https://huggingface.co/yfish/PriFold)

## 🚀 Usage

### Inference
Before running training or inference, you need to download the model and data files. For instructions, refer to [https://huggingface.co/yfish/PriFold](https://huggingface.co/yfish/PriFold).

To run the inference script:
```bash
./inference.sh
```

### Training
To run the training script:
```bash
./train.sh
```

## 📚 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{yang2025prifold,
  title={PriFold: Biological Priors Improve RNA Secondary Structure Predictions},
  author={Yang, Chenchen and Wu, Hao and Shen, Tao and Zou, Kai and Sun, Siqi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={1},
  pages={950--958},
  year={2025}
}
```


## 🎯 Impact
Our experimental results not only validate our prediction approach but also highlight the potential of integrating biological priors, such as global characteristics and evolutionary information, into RNA structure prediction tasks, opening new avenues for research in RNA biology and bioinformatics.
