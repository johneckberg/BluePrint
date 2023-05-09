import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch
from peft import get_peft_model, LoraConfig, TaskType
from torch.nn.functional import cross_entropy


class RewardModel:
    def __init__(self, summary_to_code_model):
        self.summary_to_code_model = summary_to_code_model
        self._create_and_save_reference_model()

    def _create_and_save_reference_model(self):
        self.reference_model = create_reference_model(self.summary_to_code_model)
        self.reference_model.save_pretrained("path/to/save/reference/model")

    def compute_reward(self, code, summary):
        # Tokenize code and summary
        code_tokens = self.summary_to_code_model.tokenizer.encode(code, return_tensors="pt")
        summary_tokens = self.summary_to_code_model.tokenizer.encode(summary, return_tensors="pt")

        # Get the output logits from the summary-to-code model?
        output_logits = self.summary_to_code_model(summary_tokens)[0]

        # Compute cross entropy loss
        cross_entropy_loss = cross_entropy(output_logits.squeeze(0), code_tokens.squeeze(0))

        # Compute length penalty
        length_penalty = len(summary_tokens[0]) / len(code_tokens[0])

        # Reward is negative cross entropy loss minus length penalty
        reward = -cross_entropy_loss - length_penalty

        # get logits and Add penalty (cross entropy between summary tokens and reference model tokens)
        reference_logits = self.reference_model(summary_tokens)[0]
        reference_cross_entropy_loss = cross_entropy(reference_logits.squeeze(0), summary_tokens.squeeze(0))
        reward -= reference_cross_entropy_loss

        return reward


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


class BluePrint:
    def __init__(self, code_to_summary_model_name, summary_to_code_model_name, num_epochs=3, batch_size=8,
                 max_length=1000):
        self.tokenizer = AutoTokenizer.from_pretrained(code_to_summary_model_name)
        self.code_to_summary_model = AutoModelForCausalLMWithValueHead.from_pretrained(code_to_summary_model_name)
        self.summary_to_code_model = AutoModelForCausalLMWithValueHead.from_pretrained(summary_to_code_model_name)

        # set
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_length = max_length

        # PEFT config and wrapping models with PEFT
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )
        self.code_to_summary_model = get_peft_model(self.code_to_summary_model, peft_config)
        self.summary_to_code_model = get_peft_model(self.summary_to_code_model, peft_config)

        # reward model for PPO
        self.reward = self.reward_model = RewardModel(self.summary_to_code_model)

        # Load the GitHub Code dataset and filter for Python code
        # Links: https://huggingface.co/datasets/codeparrot/github-code
        dataset = load_dataset("codeparrot/github-code", streaming=True, split="train", languages=["Python"])

        # Convert the dataset to an iterable format and preprocess the data
        num_samples = 1000
        train_dataset = StreamingCodeDataset(dataset, self.tokenizer, num_samples)
        train_sampler = DistributedSampler(train_dataset)
        self.train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, sampler=train_sampler)
        print("Github Loaded")

        # PPO config and trainer
        ppo_config = PPOConfig(batch_size=self.batch_size)
        self.ppo_trainer = PPOTrainer(ppo_config, self.code_to_summary_model, self.summary_to_code_model,
                                      self.tokenizer)

        # optimizer and learning rate
        self.lr = 1e-3
        self.optimizer = torch.optim.AdamW(self.summary_to_code_model.parameters(), lr=self.lr)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(self.train_dataloader) * num_epochs),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.code_to_summary_model.to(device)
        self.summary_to_code_model.to(device)

    def _train_code_to_summary_with_ppo(self, code):
        prompt = f"Summarize the following code using the following format:\n" \
                 f"Task description:\nData sources:\nDesired features:\nConstraints and requirements:\n{code}"
        input_tensor = self.tokenizer.encode(prompt, return_tensors="pt")
        response_tensor = respond_to_batch(self.code_to_summary_model, input_tensor)
        summary = self.tokenizer.decode(response_tensor[0])

        reward_value = self.reward_model.compute_reward(code, summary)
        reward = [torch.FloatTensor([reward_value])]
        self.ppo_trainer.step([input_tensor[0]], [response_tensor[0]], reward)

    def train_summary_to_code(self, summary, code):
        summary_prompt = f"Convert this code Summary into the code it is describing: {summary}"
        input_tensor = self.tokenizer.encode(summary_prompt, return_tensors="pt")

        # Prepare the target tensor
        target_tensor = self.tokenizer.encode(code, return_tensors="pt")

        # Forward pass
        outputs = self.summary_to_code_model(input_tensor, labels=target_tensor)
        logits = outputs.logits  # Extract logits from the outputs

        # Compute cross entropy loss
        loss = cross_entropy(logits.view(-1, logits.size(-1)), target_tensor.view(-1))

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

    def train(self, num_epochs, ppo_steps, supervised_steps):
        for epoch in range(num_epochs):
            for code_batch in self.train_dataloader:
                for code in code_batch:
                    # Train code-to-summary model with PPO
                    for _ in range(ppo_steps):
                        self._train_code_to_summary_with_ppo(code)

                    # Train summary-to-code model with supervised learning
                    for _ in range(supervised_steps):
                        # Generate summary from code
                        prompt = f"Summarize the following code using the following format:\n" \
                                 f"Task description:\nData sources:\nDesired features:\nConstraints and requirements:\n{code} "
                        input_tensor = self.tokenizer.encode(prompt, return_tensors="pt")
                        response_tensor = respond_to_batch(self.code_to_summary_model, input_tensor)
                        summary = self.tokenizer.decode(response_tensor[0])

                        # Train summary-to-code model
                        self.train_summary_to_code(summary, code)

    def save_models(self, code_to_summary_output_path, summary_to_code_output_path):
        self.code_to_summary_model.save_pretrained(code_to_summary_output_path)
        self.summary_to_code_model.save_pretrained(summary_to_code_output_path)


# Instantiate and train BluePrint models
blueprint = BluePrint("path/to/code_to_summary/model", "path/to/summary_to_code/model")
blueprint.train(num_epochs=3, ppo_steps=10, supervised_steps=10)
blueprint.save_models("path/to/save/code_to_summary/model", "path/to/save/summary_to_code/model")
