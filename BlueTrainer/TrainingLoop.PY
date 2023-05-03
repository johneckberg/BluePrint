from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader

# Load pre-trained models and tokenizers
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
code_to_summary_model = GPT2LMHeadModel.from_pretrained(model_name)
summary_to_code_model = GPT2LMHeadModel.from_pretrained(model_name)

# Prepare data (assuming you have a list of input codes)
input_codes = ["input code 1", "input code 2"]

def tokenize_data(tokenizer, data):
    return tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True)

input_code_encodings = tokenize_data(tokenizer, input_codes)

# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, input_code_encodings):
        self.input_code_encodings = input_code_encodings

    def __len__(self):
        return len(self.input_code_encodings['input_ids'])

    def __getitem__(self, idx):
        return {'input_code': torch.tensor(self.input_code_encodings['input_ids'][idx])}

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

# Train model on data
train_dataset = CustomDataset(input_code_encodings)
trainer = CustomTrainer(
    model=None,  # No need to pass model here since we handle it inside the compute_loss method
    args=training_args,
    train_dataset=train_dataset,
    code_to_summary=code_to_summary_model,
    summary_to_code=summary_to_code_model,
)

trainer.train()
