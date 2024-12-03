from transformers import GPTJConfig

def create_model_config(tokenizer, args):
    return GPTJConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=args.n_position,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        rotary_dim=30
    )