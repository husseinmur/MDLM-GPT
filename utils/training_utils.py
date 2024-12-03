from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import TrainingArguments

def get_training_args(args):
    return TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.valid_batch,
        save_steps=100,
        save_total_limit=80,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        prediction_loss_only=True,
        gradient_accumulation_steps=2,
        eval_steps=10,
        bf16=True,
        load_best_model_at_end=True,
    )

def get_optimizer_and_scheduler(model, training_args, dataset_length):
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    
    total_steps = (dataset_length // (training_args.per_device_train_batch_size * 
                  training_args.gradient_accumulation_steps)) * training_args.num_train_epochs
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=total_steps // 10,
        T_mult=2,
        eta_min=1e-6
    )
    return optimizer, scheduler