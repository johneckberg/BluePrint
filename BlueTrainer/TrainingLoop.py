import torch
import os
import torch.nn as nn
import torch.optim as optim
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from torch.utils.data import Dataset


class StreamingCodeDataset(Dataset):
    def __init__(self, dataset, tokenizer, num_samples):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = next(iter(self.dataset.skip(idx).take(1)))
        input_code = self.tokenizer.encode(item['code'], return_tensors='pt')
        return {'input_code': input_code[0]}


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


# Create the 'model_checkpoints' directory if it does not exist
if not os.path.exists("model_checkpoints"):
    os.makedirs("model_checkpoints")

# Load the GitHub Code dataset and filter for Python code
# Links: https://huggingface.co/datasets/codeparrot/github-code
ds = load_dataset("codeparrot/github-code", streaming=True, split="train", languages=["Python"])
print("Github Loaded")


# Load the Alpaca-LoRA Llama models
# Links: https://huggingface.co/decapoda-research/llama-7b-hf
# Links: https://huggingface.co/tloen/alpaca-lora-7b
BASE_MODEL = "decapoda-research/llama-7b-hf"
LORA_WEIGHTS = "tloen/alpaca-lora-7b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

# Convert the dataset to an iterable format and preprocess the data
num_samples = 1000
train_dataset = StreamingCodeDataset(ds, tokenizer, num_samples)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model_1 = load_alpaca_lora_llama_model(BASE_MODEL, LORA_WEIGHTS, device)
model_2 = load_alpaca_lora_llama_model(BASE_MODEL, LORA_WEIGHTS, device)
model = Code2CodeTranslationModel(model_1, model_2)
model.to(device)

# Set up the optimizer and training parameters
learning_rate = 5e-5
num_epochs = 5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Custom loss function
# TODO introduce weighting hyperparams
def custom_loss_function(summary_logits, generated_code_logits, input_code, summaries, tokenizer):
    summary_quality_loss = nn.CrossEntropyLoss()(summary_logits, summaries)  # exp with bertscore
    code_similarity_loss = nn.KLDivLoss()(generated_code_logits.log_softmax(dim=-1), input_code.softmax(dim=-1))
    summary_length_penalty = torch.mean(torch.sum((summaries != tokenizer.pad_token_id).float(), dim=1))

    loss = summary_quality_loss + code_similarity_loss + summary_length_penalty
    return loss


# Fine-tuning loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch in train_dataloader:
        # Prepare the input tensors with the prompt "Summarize this code:"
        # this lack of respect to the prompt format of the alpaca model might mess this up, but huggingface did it too
        # & ill just have to use a llama. but I think it will be okay?
        input_code = tokenizer("Summarize this code: " + batch["text"], return_tensors="pt", padding=True)[
            "input_ids"]
        input_code = input_code.to(device)

        # Generate summaries with the first Alpaca model
        summaries = model.generate_summaries(input_code)

        # Update input tensor with the prompt: "Convert this code Summary into the Code it is describing:"
        summary_prompt = "Convert this code Summary into the code it is describing: "
        summaries_with_prompt = tokenizer(summary_prompt, return_tensors="pt", padding=True)["input_ids"]
        summaries_with_prompt = torch.cat([summaries_with_prompt, summaries], dim=1).to(device)

        # Forward pass
        summary_logits, generated_code_logits = model(input_code, summaries_with_prompt)

        # Calculate the loss
        loss = custom_loss_function(summary_logits, generated_code_logits, input_code, summaries, tokenizer)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_dataloader)}")

# Save the weights
save_path = "model_checkpoints/epoch_{}_model.pt".format(epoch + 1)
torch.save(model.state_dict(), save_path)
