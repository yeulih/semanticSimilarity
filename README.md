# Semantic Similarity Analysis Using BERT

This repository hosts the files used in a project comparing AI tools used for academic research. Specifically, semantic similarity analysis was performed on search results from different citation-mapping literature tools. The contents of the files are as follows:
- README.md: Instructions for setting up and running BERT
  The instructions assume that you are familiar with installing developer software and working in a command-line environment. 
- Python scripts for running the semantic similarity analysis

## Project Overview

BERT (Bidirectional Encoder Representations from Transformers) is a type of Large Language Model (LLM). BERT excels at National Language Processing (NLP) tasks such as sentiment analysis, question answering, semantic similarity.

## System Requirements
Pytorch is an open source deep learning library. One of its capabilities is enabling LLMs, such as BERT, to be run on a local computer. Before proceeding with this project, review the specified hardware and software requirements to see if your computer is able to run Pytorch.

Recommended hardware requirements:
- Computer with Nvidia GPU that supports CUDA or AMD GPU (optional if you will not be using heavy computing)
- Intel Core i5 and higher or AMD Ryzen 7 and higher
- 8 GB RAM
- 256 GB storage

**Note**: The requirements were taken from https://www.geeksforgeeks.org/python/pytorch-system-requirements. Although the PyTorch website does not list specific hardware requirements, running LLMs requires a nontrivial amount of computing power.

Software requirements:
For a list of supported operating systems, see the Pytorch website (https://pytorch.org/get-started/locally/) and select your OS (Linux, Mac, Windows) listed on the interactive chart.

## Part 1: Set up and run BERT

To set up and run BERT, perform the following tasks in order:
1. Install Python</li>
2. Set up a Python virtual environment
3. 

### Install Python

Install Python from the official site (https://www.python.org/downloads/). Make sure to select the **Add python.exe to PATH** option on the install screen. pip, a package manager for Python, is automatically installed when using the official Python installer.

**Important!** Although Pytorch supports Python versions 3.10-3.14, it is recommended to install 3.10, as dependency issues in libraries can occur with later versions. 

### Set up a Python virtual environment

To set up a Python virtual environment:
1. In a command-line, create a project folder. E.g. `C:\projects\bert-local`
2. In the folder, run `python -m venv *<env_name>*` to create a virtual environment. E.g. `python -m venv .venv`
3. In the same folder, run `.\*<env_name>*\scripts\activate.bat` to activate the environment. E.g. `.\.venv\scripts\activate.bat`
  
### Install Pytorch

The package manager: pip is used to install the PyTorch binaries. The command to install Pytorch in the virtual environment differs based on the following factors:
- OS
- computing using a GPU or CPU
- GPU model (if applicable).

Use one of the following methods to find the command syntax:
- If your GPU supports the latest CUDA SDKs and you want to install the latest stable Pytorch version, select your configuration using the interactive chart on https://pytorch.org/get-started/locally/ and the command will be generated for you.
- If your GPU supports earlier CUDA SDKs or you want to install an earlier Pytorch version, find the relevant command at https://pytorch.org/get-started/previous-versions/.

*Important!* Even if your GPU supports the latest CUDA SDKs, compatability issues still might occur when attemping to install one of the latest Pytorch CUDA wheels. You can use online LLMs such as ChatGPT to help you troubleshoot which CUDA wheel and Pytorch version to install, or you can choose to only use the CPU to compute and skip installing the CUDA wheel.

This project ran on a computer that has the Nivdieo Quadro T2000. After much trial and error, the final working configuration was:
Python version: 3.11.9
Pytorch version: 2.7.1
CUDA wheel 11.8

Install Hugging Face Transformers and dependencies
In your activated virtual environment, run pip install transformers

Cache the model locally for offline use

In Python, after loading the model and tokenizer once, run:

python
save_dir = "./models/bert-base-uncased"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
