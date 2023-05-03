Custom Trainer and Training Loop for Code-to-Summary and Summary-to-Code Models

This repository contains a custom trainer and training loop for end-to-end fine-tuning of code-to-summary and summary-to-code models. The custom trainer is designed for use with Hugging Face's Transformers library and PyTorch.
Features

    End-to-end fine-tuning of code-to-summary and summary-to-code models
    Custom loss function with a focus on generating concise, human-readable, and accurate summaries that can effectively recreate the original code
    BERTScore metric for summary quality evaluation
    Easy integration with Hugging Face Transformers library and pre-trained models

Dependencies

    Python 3.6 or higher
    PyTorch 1.9.0 or higher
    Transformers 4.0.0 or higher
    bert-score 0.3.10 or higher

Installation

Clone this repository and install the required dependencies:

bash

git clone https://github.com/your-username/custom-trainer.git
cd custom-trainer
pip install -r requirements.txt
