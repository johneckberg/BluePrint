import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# Load pre-trained models and tokenizers
from BlueTrainer.CustomTrainer import CustomTrainer

# link: https://huggingface.co/baseten/alpaca-30b
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-30b-hf")
code_to_summary_model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-30b-hf")
summary_to_code_model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-30b-hf")

# Stream GitHub Code dataset and filter for Python code
# link: https://huggingface.co/datasets/codeparrot/github-code
ds = load_dataset("codeparrot/github-code", streaming=True, split="train", languages=["Python"])


# Custom dataset for prompting alpaca #1
class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, num_samples=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples if self.num_samples else len(self.dataset)

    def __getitem__(self, idx):
        item = next(iter(self.dataset.skip(idx).take(1)))
        input_text = "summarize the following code: " + item['code']
        input_code = self.tokenizer.encode(input_text, return_tensors='pt')
        return {'input_code': input_code[0]}


# Instantiate custom trainer
training_args = TrainingArguments(
    output_dir="./training_output",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
    # Add other training arguments as needed
)

# Prepare the models for PEFT fine-tuning
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

code_to_summary_model = get_peft_model(code_to_summary_model, peft_config)
summary_to_code_model = get_peft_model(summary_to_code_model, peft_config)

# Train models on data
num_samples = 1000  # Adjust the number of samples as needed
train_dataset = CustomDataset(ds, tokenizer, num_samples=num_samples)
trainer = CustomTrainer(
    code_to_summary=code_to_summary_model,
    summary_to_code=summary_to_code_model,
    training_args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
