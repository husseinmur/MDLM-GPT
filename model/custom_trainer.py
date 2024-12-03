from transformers import Trainer
import torch
from model.helpers import *

class CustomTrainer(Trainer):
    def __init__(self, *args, kdes, fasta, tokenizer, **kwargs):
        super().__init__(*args, **kwargs)
        self.kdes = kdes
        self.fasta = fasta
        self.tokenizer = tokenizer
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        angles_loss = get_angles_loss(outputs.logits, self.tokenizer, self.kdes, self.fasta)
        
        if labels is not None:
            loss = self.label_smoother(outputs, labels) if self.label_smoother is not None else outputs["loss"]
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
        combined_loss = loss + 0.2 * angles_loss

        return (combined_loss, outputs) if return_outputs else combined_loss