from trl import DistillationTrainer, DistillationConfig
from transformers import AutoModelForCausalLM, set_seed
from datasets import load_from_disk
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import wandb
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent

from src.environment.tools import check_location, ask_for_clarification, request_new_location, can_fulfill_material_request
from src.training.setup import load_model_and_tokenizer, setup
from src.training.lora import build_lora_config


@hydra.main(config_path="config", config_name="train_opd", version_base=None)
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
        chat_template_kwargs={
            "enable_thinking": args.train.enable_thinking,
        },
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
        train_dataset=dataset[args.dataset.train_split],
        eval_dataset=dataset[args.dataset.validation_split],
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
