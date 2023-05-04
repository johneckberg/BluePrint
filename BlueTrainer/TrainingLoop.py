import torch
import torch.nn as nn
import torch.optim as optim
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset




def load_alpaca_lora_llama_model(base_model, lora_weights, device):
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model, lora_weights, torch_dtype=torch.float16, force_download=True
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    return model


class Code2CodeTranslationModel(nn.Module):
    def __init__(self, model_1, model_2):
        super().__init__()
        self.code_to_summary = model_1
        self.summary_to_code = model_2

    def forward(self, input_code):
        summary_logits = self.code_to_summary(input_code).logits
        generated_code_logits = self.summary_to_code(summary_logits).logits
        return summary_logits, generated_code_logits


# Load the GitHub Code dataset and filter for Python code
ds = load_dataset("codeparrot/github-code", streaming=True, split="train", languages=["Python"])
print("Github Loaded")

# Convert the dataset to an iterable format and preprocess the data
num_samples = 1000
train_dataset = list(ds.take(num_samples))
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Load the Alpaca-LoRA Llama models
BASE_MODEL = "decapoda-research/llama-7b-hf"
LORA_WEIGHTS = "tloen/alpaca-lora-7b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

model_1 = load_alpaca_lora_llama_model(BASE_MODEL, LORA_WEIGHTS, device)
model_2 = load_alpaca_lora_llama_model(BASE_MODEL, LORA_WEIGHTS, device)
model = Code2CodeTranslationModel(model_1, model_2)
model.to(device)

# Set up the optimizer and training parameters
learning_rate = 5e-5
num_epochs = 5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Custom loss function
def custom_loss_function(summary_logits, generated_code_logits, input_code, summaries, tokenizer):
    summary_quality_loss = nn.CrossEntropyLoss()(summary_logits, summaries)
    code_similarity_loss = nn.KLDivLoss()(generated_code_logits.log_softmax(dim=-1), input_code.softmax(dim=-1))
    summary_length_penalty = torch.mean(torch.sum((summaries != tokenizer.pad_token_id).float(), dim=1))

    loss = summary_quality_loss + code_similarity_loss + summary_length_penalty
    return loss


# Fine-tuning loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch in train_dataloader:
        # Prepare the input tensors
        input_code = tokenizer(batch["text"], return_tensors="pt", padding=True)[
            "input_ids"]  # Change 'input_code' to 'text'
        input_code = input_code.to(device)

        # Generate summaries with the first Alpaca model
        summaries = model.generate_summaries(input_code)

        # Forward pass
        summary_logits, generated_code_logits = model(input_code, summaries)

        # Calculate the loss
        loss = custom_loss_function(summary_logits, generated_code_logits, input_code, summaries, tokenizer)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_dataloader)}")
