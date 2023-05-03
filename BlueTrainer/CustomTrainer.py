import torch
from torch import nn
from transformers import Trainer
from bert_score import score as bert_score


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Input code and target code (assuming they are the same in this case)
        input_code = inputs.get("input_code")
        target_code = inputs.get("target_code")

        # Forward pass through the code-to-summary model
        summary_outputs = model.code_to_summary(**inputs)
        summaries = summary_outputs.get("logits")

        # Forward pass through the summary-to-code model
        code_outputs = model.summary_to_code(summaries)
        generated_code = code_outputs.get("logits")

        # Compute BERTScore for summary quality
        with torch.no_grad():
            input_code_strings = tokenizer.batch_decode(input_code, skip_special_tokens=True)
            summaries_strings = tokenizer.batch_decode(summaries, skip_special_tokens=True)
            P, R, F1 = bert_score(summaries_strings, input_code_strings, model_type="bert-base-uncased",
                                  device=self.args.device)

        summary_quality = F1.mean()

        # Compute KL divergence between input and output code
        kl_div = nn.KLDivLoss(reduction="batchmean")
        kl_loss = kl_div(torch.log_softmax(generated_code.view(-1), dim=-1), torch.softmax(input_code.view(-1), dim=-1))

        # Compute summary length penalty (using mean length as an example)
        summary_length_penalty = torch.mean(torch.tensor([len(summary) for summary in summaries]))

        # Set the weighting factors for each loss component
        summary_quality_weight = 1.0
        kl_loss_weight = 1.0
        summary_length_penalty_weight = 1.0

        # Calculate the combined custom loss
        loss = (
            -summary_quality_weight * summary_quality
            + kl_loss_weight * kl_loss
            + summary_length_penalty_weight * summary_length_penalty
        )

        return (loss, summary_outputs, code_outputs) if return_outputs else loss

