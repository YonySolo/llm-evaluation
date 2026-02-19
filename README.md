# LLM Evaluation Benchmark

A Python tool that evaluates and compares the performance of open-source language models using the Hugging Face Inference API.

<img width="2085" height="924" alt="image" src="https://github.com/user-attachments/assets/3e8726f2-e070-4d90-bbc4-fa00e4bdbb73" />


## What It Does

- Sends a dataset of questions to multiple LLM models
- Measures accuracy and response latency for each model
- Generates a visual comparison chart

## Models Tested

- HuggingFaceTB/SmolLM3-3B
- mistralai/Mistral-7B-Instruct-v0.3
- google/gemma-2-2b-it

## Key Findings

- SmolLM3-3B achieved 90% accuracy but had the highest latency (1.38s)
- Mistral-7B was the fastest (0.29s) but returned errors on the free API tier
- Gemma-2-2b also returned errors, highlighting model availability constraints

## Technologies

- Python
- Hugging Face Inference API
- pandas
- matplotlib

## Setup

1. Clone the repo
2. Create a virtual environment and install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Hugging Face token: `HF_TOKEN=your_token`
4. Run `python evaluate.py` to generate results
5. Run `python analyse.py` to analyze and create charts
