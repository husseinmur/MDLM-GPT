import torch
from transformers import BertTokenizerFast
import transformers

class ModelSetup:
    def __init__(self, tokenizer_path, model_path, model_name="GPTJForCausalLM"):
        self.tokenizer = self._setup_tokenizer(tokenizer_path)
        self.model = self._setup_model(model_path, model_name)
    
    def _setup_tokenizer(self, tokenizer_path):
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, max_len=512, do_lower_case=False)
        special_tokens_dict = {'additional_special_tokens': ['.']}
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer
    
    def _setup_model(self, model_path, model_name):
        model = getattr(transformers, model_name).from_pretrained(model_path)
        model.to('cuda')
        return model