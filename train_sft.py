from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, set_seed
from dotenv import load_dotenv
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import wandb


PROJECT_ROOT = Path(__file__).resolve().parent

from src.training.setup import load_model_and_tokenizer, setup
from src.training.preprocessing import (
    enable_assistant_tool_call_mask,
    prepare_sft_dataset,
)
from src.training.lora import build_lora_config
from src.training.callbacks import (
    EnvironmentValidationCallback,
    validation_rollout_rows,
)


def balanced_subset(dataset, size, seed):
    """Select a seeded subset stratified by task type."""
    if size >= len(dataset):
        return dataset
    encoded = dataset.class_encode_column("trajectory_type")
    return encoded.train_test_split(
        train_size=size,
        stratify_by_column="trajectory_type",
        seed=seed,
    )["train"]


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
    enable_assistant_tool_call_mask(tokenizer)
    peft_config = build_lora_config(args.lora, model)
    print("Model and tokenizer loaded successfully.")

    dataset_path = PROJECT_ROOT / args.dataset.file
    dataset = load_from_disk(str(dataset_path))
    train_dataset = prepare_sft_dataset(
        dataset[args.dataset.train_split],
        tokenizer,
        args.train.enable_thinking,
        args.train.reasoning_effort,
        args.train.max_seq_length,
    )
    validation_source = balanced_subset(
        dataset[args.dataset.validation_split],
        args.dataset.validation_examples,
        args.train.seed,
    )
    validation_dataset = prepare_sft_dataset(
        validation_source,
        tokenizer,
        args.train.enable_thinking,
        args.train.reasoning_effort,
        args.train.max_seq_length,
    )
    print("Datasets loaded successfully.")

    config = SFTConfig(
        seed=args.train.seed,
        data_seed=args.train.seed,
        per_device_train_batch_size=args.train.per_device_train_batch_size,
        gradient_accumulation_steps=args.train.gradient_accumulation_steps,
        learning_rate=args.train.learning_rate,
        max_steps=args.train.max_steps,
        packing=args.train.packing,
        padding_free=args.train.padding_free,
        gradient_checkpointing=args.train.gradient_checkpointing,
        per_device_eval_batch_size=args.train.per_device_eval_batch_size,
        max_length=args.train.max_seq_length,
        assistant_only_loss=False,
        dataset_kwargs={"skip_prepare_dataset": True},
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
    rollout_rows = validation_rollout_rows(
        validation_source,
        args.train.validation_rollout_scenarios,
        args.train.seed,
    )
    callbacks.append(EnvironmentValidationCallback(
        rollout_rows,
        tokenizer,
        args.train.enable_thinking,
        args.train.reasoning_effort,
        args.train.validation_rollout_steps,
    ))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        callbacks=callbacks,
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
