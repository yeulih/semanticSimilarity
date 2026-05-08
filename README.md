# Semantic Similarity Analysis Using BERT

This repository hosts the files used in a project comparing AI tools used for academic research. Specifically, semantic similarity analysis was performed on search results from different citation-mapping literature tools. The contents of the files are as follows:
<ul>
<li>README.md: Instructions for setting up and running BERT</li>
<li>Python scripts for running the semantic similarity analysis</li>
</ul>

Project Overview

BERT (Bidirectional Encoder Representations from Transformers) is a type of LLM. BERT excels at National Language Processing (NLP) tasks such as sentiment analysis, question answering, semantic similarity.

## System Requirements
Pytorch is an open source deep learning library. One of its capabilities is allowing Large Language Models (LLMs), including BERT, to be run on a local computer. Check the specified hardware and software requirements to see if you can run PyTorch on your local computer.

Recommended Hardware requirements:
*Note*: Although the PyTorch website does not list specific hardware requirements, the requirements posted at https://www.geeksforgeeks.org/python/pytorch-system-requirements seemed reasonable.
<ul>
<li>Computer with Nvidia GPU that supports CUDA or AMD GPU</li>
<li>Intel Core i5 and higher or AMD Ryzen 7 and higher</li>
<li>8 GB RAM</li>
<li>256 GB Storage</li>
</ul>

Software requirements:
For a list of supported operating systems, see the Pytorch website (https://pytorch.org/get-started/locally/) and select your OS (Linux, Mac, Windows) listed on the chart.

Set up and run BERT

To set up and run BERT, perform the following tasks in order:


Install Python
Install Python 3.9–3.12 from the official site (check “Add python.exe to PATH” during install)

https://www.python.org/downloads/windows/

Note: pip, package manager for Python, is alo automatically installed.


Set up a Python virtual environment

<ol>
<li>Create a project folder</li>
<li>In that folder, run:

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
