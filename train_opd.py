from trl import DistillationTrainer, DistillationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from huggingface_hub import login
from datasets import load_from_disk
from pathlib import Path
from omegaconf import OmegaConf
import os
import hydra
import wandb
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent

from src.environment.tools import check_location, ask_for_clarification, request_new_location, can_fulfill_material_request


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

    if wandb_api_key:
        return "wandb", wandb.run.name

    return "none", "local-run"


def load_model_and_tokenizer(args):
    model = AutoModelForCausalLM.from_pretrained(args.train.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.train.model_name)

    return model, tokenizer


@hydra.main(config_path="config", config_name="train_opd", version_base=None)
def main(args):

    set_seed(args.train.seed, deterministic=True)

    report_to, run_name = setup()
    run_dir = PROJECT_ROOT / args.train.final_model.output_dir / run_name
    checkpoints_dir = run_dir / "checkpoints"
    final_model_dir = run_dir / "final_model"
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(args, config_dir / "train.yaml")

    model, tokenizer = load_model_and_tokenizer(args)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.train.teacher_model)
    print("Model and tokenizer loaded successfully.")

    dataset_path = PROJECT_ROOT / args.dataset.file
    dataset = load_from_disk(str(dataset_path))
    print("Datasets loaded successfully.")

    config = DistillationConfig(
        per_device_train_batch_size=args.train.per_device_train_batch_size,
        learning_rate=args.train.learning_rate,
        max_steps=args.train.max_steps,
        gradient_accumulation_steps=args.train.gradient_accumulation_steps,
        # generation
        temperature=args.train.temperature,
        top_p=args.train.top_p,
        top_k=args.train.top_k,
        max_completion_length=args.train.max_completion_length,
        max_tool_calling_iterations=args.train.max_tool_calling_iterations,
        # precision
        use_cpu=args.train.use_cpu,
        bf16=args.train.bf16,
        fp16=args.train.fp16,
        # logging
        report_to=report_to,
        logging_steps=args.train.logging_steps,
        log_completions=args.train.log_completions,
        num_completions_to_print=args.train.num_completions_to_print,
    )

    trainer = DistillationTrainer(
        model=model,
        teacher_model=teacher_model,
        processing_class=tokenizer,
        tools=[check_location, ask_for_clarification, request_new_location, can_fulfill_material_request],
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
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