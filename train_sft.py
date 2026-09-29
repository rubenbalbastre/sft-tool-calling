from datasets import load_from_disk
from functools import partial
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, set_seed
from dotenv import load_dotenv
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import wandb


PROJECT_ROOT = Path(__file__).resolve().parent

from src.training.setup import load_model_and_tokenizer, setup
from src.training.preprocessing import format_sft_example, prepare_sft_source
from src.training.lora import build_lora_config


@hydra.main(config_path="config", config_name="train_sft", version_base=None)
def main(args):

    load_dotenv(PROJECT_ROOT / ".env")
    set_seed(args.train.seed, deterministic=True)

    report_to, run_name = setup(args.train.run_name)
    run_dir = PROJECT_ROOT / args.train.final_model.output_dir / run_name
    checkpoints_dir = run_dir / "checkpoints"
    final_model_dir = run_dir / "final_model"
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(args, config_dir / "train.yaml")

    model, tokenizer = load_model_and_tokenizer(args)
    peft_config = build_lora_config(args.lora, model)
    print("Model and tokenizer loaded successfully.")

    dataset_path = PROJECT_ROOT / args.dataset.file
    dataset = load_from_disk(str(dataset_path))
    train_dataset = prepare_sft_source(dataset[args.dataset.train_split])
    validation_dataset = prepare_sft_source(
        dataset[args.dataset.validation_split]
    )
    print("Datasets loaded successfully.")

    config = SFTConfig(
        seed=args.train.seed,
        data_seed=args.train.seed,
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
        load_best_model_at_end=args.train.early_stopping.enabled,
        metric_for_best_model=args.train.early_stopping.metric,
        greater_is_better=args.train.early_stopping.greater_is_better,
        output_dir=str(checkpoints_dir),
        report_to=report_to,
    )
    callbacks = []
    if args.train.early_stopping.enabled:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.train.early_stopping.patience,
            early_stopping_threshold=args.train.early_stopping.threshold,
        ))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        callbacks=callbacks,
        formatting_func=partial(format_sft_example, tokenizer=tokenizer),
        peft_config=peft_config,
    )
    print("Trainer initialized successfully.")

    trainer.train()

    if args.train.final_model.save:
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        print(f"Run artifacts saved to {run_dir}")

    if report_to == "wandb":
        wandb.finish()


if __name__ == "__main__":
    main()
