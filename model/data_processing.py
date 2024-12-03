from datasets import load_dataset
from transformers import BertTokenizerFast

def setup_tokenizer(tokenizer_path):
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, max_len=512, do_lower_case=False)
    special_tokens_dict = {'additional_special_tokens': ['.']}
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer

def load_and_process_datasets(train_path, valid_path, tokenizer, block_size=252):
    datasets = load_dataset("text", data_files={"train": train_path, "validation": valid_path})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        
        chunks = []
        chunk_start = 0
        
        while chunk_start < total_length:
            chunk_end = min(chunk_start + block_size, total_length)
            
            if chunk_end < total_length:
                for i in range(chunk_end - 1, chunk_start, -1):
                    if concatenated_examples['input_ids'][i] == tokenizer.convert_tokens_to_ids("."):
                        chunk_end = min(i + 2, total_length)
                        break
            
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end
        
        result = {}
        for key in concatenated_examples.keys():
            result[key] = []
            for start, end in chunks:
                chunk = concatenated_examples[key][start:end]
                
                if len(chunk) < block_size:
                    chunk = chunk + [tokenizer.pad_token_id] * (block_size - len(chunk))
                elif len(chunk) > block_size:
                    chunk = chunk[:block_size]
                
                result[key].append(chunk)
        
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )