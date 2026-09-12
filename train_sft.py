from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from huggingface_hub import login
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
from pathlib import Path
from omegaconf import OmegaConf
import os
import hydra
import wandb


PROJECT_ROOT = Path(__file__).resolve().parent


def setup():
    load_dotenv(PROJECT_ROOT / ".env")

    huggingface_api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if huggingface_api_key:
        login(token=huggingface_api_key)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb_project = os.environ.get("WANDB_PROJECT")
    if wandb_api_key:
        if not wandb_project:
            raise ValueError(
                "WANDB_PROJECT must be set when WANDB_API_KEY is configured."
            )
        wandb.login(key=wandb_api_key)
        wandb.init(project=wandb_project)

    return "wandb" if wandb_api_key else "none"


def load_model_and_tokenizer(args):
    model = AutoModelForCausalLM.from_pretrained(args.train.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.train.model_name)

    return model, tokenizer


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(args):
    report_to = setup()
    model, tokenizer = load_model_and_tokenizer(args)
    print("Model and tokenizer loaded successfully.")

    dataset_path = PROJECT_ROOT / args.dataset.file
    dataset = load_from_disk(str(dataset_path))
    print("Datasets loaded successfully.")

    config = SFTConfig(
        per_device_train_batch_size=args.train.per_device_train_batch_size,
        gradient_accumulation_steps=args.train.gradient_accumulation_steps,
        learning_rate=args.train.learning_rate,
        max_steps=args.train.max_steps,
        per_device_eval_batch_size=args.train.per_device_eval_batch_size,
        max_length=args.train.max_seq_length,
        bf16=args.train.bf16,
        fp16=args.train.fp16,
        use_cpu=args.train.use_cpu,
        logging_steps=args.train.logging_steps,
        eval_strategy=args.train.eval_strategy,
        eval_steps=args.train.eval_steps,
        save_strategy=args.train.checkpointing.save_strategy,
        save_steps=args.train.checkpointing.save_steps,
        save_total_limit=args.train.checkpointing.save_total_limit,
        output_dir=str(PROJECT_ROOT / "outputs"),
        report_to=report_to,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    print("Trainer initialized successfully.")

    trainer.train()

    if args.train.final_model.save:
        run_name = wandb.run.name if report_to == "wandb" else "local-run"
        final_model_dir = PROJECT_ROOT / args.train.final_model.output_dir / run_name
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        OmegaConf.save(args, final_model_dir / "train.yaml")
        print(f"Final model, tokenizer, and configuration saved to {final_model_dir}")

    if report_to == "wandb":
        wandb.finish()


if __name__ == "__main__":
    main()
