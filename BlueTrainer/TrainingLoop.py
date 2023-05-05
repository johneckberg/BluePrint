import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from torch.utils.data import Dataset


# The main idea is to use one Alpaca model to generate a summary of a given code snippet
# and then use another Alpaca model to translate the summary back into code.
# The model is trained on the GitHub Code dataset filtered for Python code.


# use nccl instead of gloo if not running on mac
# like this can even run on my mac anyways...


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
        return input_code[0]


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


def save_summaries(summaries, tokenizer, epoch, batch_idx):
    decoded_summaries = [tokenizer.decode(summary, skip_special_tokens=True) for summary in summaries]
    save_path = "model_checkpoints/summaries_epoch_{}.txt".format(epoch + 1)

    with open(save_path, "a") as f:
        for idx, summary in enumerate(decoded_summaries):
            f.write(f"Batch {batch_idx}, Sample {idx}:\n{summary}\n\n")


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

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
dist.init_process_group("gloo", rank=0, world_size=1) # nccl

train_sampler = DistributedSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, sampler=train_sampler)

model_1 = load_alpaca_lora_llama_model(BASE_MODEL, LORA_WEIGHTS, device)
model_2 = load_alpaca_lora_llama_model(BASE_MODEL, LORA_WEIGHTS, device)
model = Code2CodeTranslationModel(model_1, model_2)
# pass model to device when starting loop

# Set up the optimizer and training parameters
learning_rate = 5e-5
num_epochs = 5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Custom loss function
# TODO introduce weighting hyperparams
def custom_loss_function(summary_logits, generated_code_logits, input_code, summaries, tokenizer):
    # Calculate the Cross-Entropy Loss between the summary_logits and the ground-truth summaries
    summary_quality_loss = nn.CrossEntropyLoss()(summary_logits, summaries) #TODO BertScore

    # Calculate the Kullback-Leibler Divergence Loss between the generated_code_logits and the input_code
    code_similarity_loss = nn.KLDivLoss()(generated_code_logits.log_softmax(dim=-1), input_code.softmax(dim=-1))

    # Compute the length penalty for the generated summaries by counting non-padding tokens
    summary_length_penalty = torch.mean(torch.sum((summaries != tokenizer.pad_token_id).float(), dim=1))

    # Combine the three loss components (summary_quality_loss, code_similarity_loss, and summary_length_penalty)
    loss = summary_quality_loss + code_similarity_loss + summary_length_penalty

    return loss  # Return the combined loss



def init_process(rank, model, world_size, backend='gloo'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Fine-tuning loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        for batch in train_dataloader:
            batch_idx = 0
            # Prepare the input tensors with the prompt "Summarize this code:"
            # this lack of respect to the prompt format of the alpaca model might mess this up
            # but huggingface staff did it too.
            # ill just have to use a llama if broken. but I think it will be okay?
            input_code = tokenizer("Summarize this code: " + batch["text"], return_tensors="pt", padding=True)[
                "input_ids"]
            input_code = input_code.to(device)

            # Generate summaries with the first Alpaca model
            summaries = model.generate_summaries(input_code)

            # save summaries
            save_summaries(summaries, tokenizer, epoch, batch_idx)

            # Update input tensor with the prompt: "Convert this code Summary into the Code it is describing:"
            summary_prompt = "Convert this code Summary into the code it is describing: "
            summaries_with_prompt = tokenizer(summary_prompt, return_tensors="pt", padding=True)["input_ids"]
            summaries_with_prompt = torch.cat([summaries_with_prompt, summaries], dim=1).to(device)

            # Forward pass with tokenized input
            summary_logits, generated_code_logits = model(input_code, summaries_with_prompt)

            # Calculate the loss
            loss = custom_loss_function(summary_logits, generated_code_logits, input_code, summaries, tokenizer)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            batch_idx += 1

        print(f" Epoch: {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_dataloader)}")

    # Save the weights
    save_path = "model_checkpoints/epoch_{}_model.pt".format(epoch + 1)
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(init_process, args=(world_size,), nprocs=world_size, join=True)
