# Reproducing and Extending "Understanding Transformers via N-gram Statistics"
This repository contains code and analysis for our project based on the paper
"Understanding Transformers via N-gram Statistics" by Timothy Nguyen
(arXiv:2407.12034).

# Project Description
As part of our Data Mining course, we reproduce selected experiments from the paper and extend them with our own investigations. Our work includes:

- Finetuning of a 124 M gpt2 based model, as our replication of the small TinyStories Model 160 M presented during the paper

- Reproducing Figure 5: N-gram matching vs model predictions

- Reproducing Table 13 & 14: Top-1 accuracy and average distance between model predictions and N-gram rules on TinyStories
  
- Designing new experiments to explore linguistic properties of N-gram rules

- Writing a final report summarizing our results



# Technologies
* Python 3.10+

* PyTorch

* Transformers (Hugging Face)

* NumPy, pandas

* Matplotlib, Seaborn

* Dask 

* Logger

* Collections

# Usage
Install the dependencies:

**pip install -r requirements.txt**

Then run one of the experiment scripts

The scripts generate outputs such as heatmaps, accuracy plots, token-level comparisons,and models with checkpoints reports for learning curve analysis

# Structure
Our Github repository is divided into two main parts, and one miscellaneus folder

- Model_Training_Scripts: this folders contains python scripts for the model training(finetuning), which save the output model in safetensors format for compatibility and security. For our project, the models trained were stored in (https://huggingface.co/dadosbon) and are open to public access.

- Replicates_and_Experiments: This folder contains 5 jupyter notebooks describing the experiments done during our project.
- eE_Kontexte.ipynb
- eE_Modellvergleich
- eERegeln_aus_fremden_datensätzen.ipynb
- figure5_replicate.ipynb
- replica_top1acc_distance.ipynb.

Each notebook presents a new experiment along with explanations of the methodology and results.


# Contributors
* Daniel Felipe Rivera-Cerquera
* Larissa Roth

# Reference
**Nguyen, Timothy. Understanding Transformers via N-gram Statistics. arXiv preprint arXiv:2407.12034, 2024.
https://doi.org/10.48550/arXiv.2407.12034**
