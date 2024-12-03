import logging
import pickle
from transformers import GPTJForCausalLM, DataCollatorForLanguageModeling
from config import get_arg_parser
from model.model_config import create_model_config
from model.data_processing import setup_tokenizer, load_and_process_datasets
from model.custom_trainer import CustomTrainer
from utils.training_utils import get_training_args
from KDE.KDE2D import KDE2D

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    args = get_arg_parser().parse_args()
    
    # Setup tokenizer and load datasets
    tokenizer = setup_tokenizer(args.tokenizer)
    lm_datasets = load_and_process_datasets(args.train_data, args.valid_data, tokenizer)
    
    # Create model configuration and initialize model
    config = create_model_config(tokenizer, args)
    model = GPTJForCausalLM(config=config)
    
    # Setup training arguments and data collator
    training_args = get_training_args(args)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.1
    )
    
    # Load KDE functions
    with open('KDE/low_memory_kde_functions.pkl', 'rb') as f:
        kdes = pickle.load(f)
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        fasta="GYDPETGTWG",
        kdes=kdes,
    )
    
    # Train and save model
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()