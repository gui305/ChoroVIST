# **ChoroVIST**
**ChoroVIST** is an interface to analyze choroid metrics such as Vascular Index and Thickness and performs choroid region segmentations using a backend code called **Choroidalyzer**, which is also available on GitHub on https://github.com/justinengelmann/Choroidalyzer . The interface is implemented with a simple database and includes several useful features that make it easy to use and highly relevant for medical professionals.o use

## ChoroVIST
![App Screenshot](https://github.com/user-attachments/assets/ce603e6a-8b02-4c33-8899-e9592a425a91)

## 🔧 Installation

It is necessary to first create a conda environment. If you haven't installed it yet, we recommend using the lightweight [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main). Then, create an environment for ChoroVIST and activate it:


```bash
conda create -n chorovist python=3.10
conda activate chorovist
pip install torch torchvision torchaudio matplotlib tqdm pandas scikit-image scipy
