from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from huggingface_hub import login
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
import os
import hydra
import wandb


def setup():
    load_dotenv()
    login(os.environ.get("HUGGINGFACE_API_KEY"))
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

def load_model_and_tokenizer(args):

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    return model, tokenizer

@hydra.main(config_path="configs", config_name="config")
def main(args):
    setup()
    model, tokenizer = load_model_and_tokenizer(args)
    print("Model and tokenizer loaded successfully.")

    dataset = load_from_disk(args.dataset.train_file)
    print("Datasets loaded successfully.")

    config = SFTConfig(
        per_device_train_batch_size=args.train.per_device_train_batch_size,
        gradient_accumulation_steps=args.train.gradient_accumulation_steps,
        learning_rate=args.train.learning_rate,
        max_steps=args.train.max_steps,
        per_device_eval_batch_size=args.train.per_device_eval_batch_size,
        max_seq_length=args.train.max_seq_length,
        logging_steps=args.train.logging_steps,
        report_to="wandb"
    )
    trainer = SFTTrainer(
        model=model, 
        processing_class=tokenizer, 
        config=config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        callback=None,
    )
    print("Trainer initialized successfully.")

    trainer.train()


if __name__ == "__main__":
    main()