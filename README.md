# Semantic Similarity Analysis Using BERT

This repository hosts the files used in a project comparing AI tools used for academic research. Specifically, semantic similarity analysis was performed on search results from different citation-mapping literature tools. The contents of the files are as follows:
- README.md: Instructions for setting up and running BERT
- Python scripts for running the semantic similarity analysis

## Project Overview

BERT (Bidirectional Encoder Representations from Transformers) is a type of Large Language Model (LLM). BERT excels at National Language Processing (NLP) tasks such as sentiment analysis, question answering, semantic similarity.

## System Requirements
Pytorch is an open source deep learning library. One of its capabilities is enabling LLMs, such as BERT, to be run on a local computer. Review the specified hardware and software requirements to see if your computer is able to run Pytorch.

Recommended hardware requirements:
- Computer with Nvidia GPU that supports CUDA or AMD GPU
- Intel Core i5 and higher or AMD Ryzen 7 and higher
- 8 GB RAM
- 256 GB storage

**Note**: The requirements were taken from https://www.geeksforgeeks.org/python/pytorch-system-requirements. Although the PyTorch website does not list specific hardware requirements, running LLMS requires a nontrivial amount of computing power.

Software requirements:
For a list of supported operating systems, see the Pytorch website (https://pytorch.org/get-started/locally/) and select your OS (Linux, Mac, Windows) listed on the chart.

## Part 1: Set up and run BERT

To set up and run BERT, perform the following tasks in order:
1. Install Python</li>
2. Set up a Python virtual environment
3. 

### Install Python
Install Python from the official site (https://www.python.org/downloads/). Select the **Add python.exe to PATH** option on the install screen. Although Pytorch supports Python versions 3.10-3.14, it is recommended to install 3.10, as dependency issues in libraries can occur with later versions. 

**Note**: pip, a package manager for Python, is automatically installed when using the official Python installer.

### Set up a Python virtual environment
To set up a Python virtual environment:
1. Create a project folder, e.g. `C:\projects\bert-local`
2. , run:

python -m venv .venv</li>
<li>Activate the environment:</li>
  
</ol>


Install Pytorch

This project uses Pytorch version 2.11.0

To install the PyTorch binaries, you will need to use the supported package manager: pip.

To install PyTorch via pip, and do have a CUDA-capable system, in the above selector, choose OS: Windows, Package: Pip and the CUDA version suited to your machine. Often, the latest CUDA version is better. Then, run the command that is presented to you.

Install Hugging Face Transformers and dependencies
In your activated virtual environment, run pip install transformers

Cache the model locally for offline use

In Python, after loading the model and tokenizer once, run:

python
save_dir = "./models/bert-base-uncased"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
