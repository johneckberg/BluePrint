# BluePrint

BluePrint is a Python module that utilizes transformers and reinforcement learning techniques to improve the performance of code summarization and code generation models. The module trains two models: a code-to-summary model and a summary-to-code model. It combines Proximal Policy Optimization (PPO) with supervised learning to train these models.

## Key Components

1. **RewardModel**: A class that computes rewards for the PPO algorithm by comparing the generated summary with the actual code and a reference model. The reward function is based on cross-entropy loss, length penalty, and reference model penalty.
2. **StreamingCodeDataset**: A custom dataset class that efficiently handles large datasets using the Hugging Face datasets library, making it suitable for training on large code datasets like the GitHub Code dataset.
3. **BluePrint**: The main class responsible for training and saving the models. It initializes the models, loads the GitHub Code dataset, configures the PPO algorithm, and sets up the learning rate scheduler and optimizer. The `train()` method handles the training loop, while `save_models()` saves the trained models.

## Usage

1. Instantiate the BluePrint class with the appropriate paths to the pre-trained code-to-summary and summary-to-code models.
  Note that due to the peft process, starting wieghts should not have to be saved as two seperate copies if using the same model.
4. Train the models using the `train()` method with the desired number of epochs, PPO steps, and supervised steps.
5. Save the trained models using the `save_models()` method with the output paths.

```python
blueprint = BluePrint("path/to/decapoda-research/llama-7b-hf", "path/to/decapoda-research/llama-7b-hf")
blueprint.train(num_epochs=3, ppo_steps=10, supervised_steps=10)
blueprint.save_models("path/to/save/code_to_summary/model", "path/to/save/summary_to_code/model")
```

## Dependencies

- torch
- transformers
- trl
- peft
- huggingface/datasets

Make sure to install the required packages before running the code.
