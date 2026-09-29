import os
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import wandb


def load_model_and_tokenizer(args):
    model = AutoModelForCausalLM.from_pretrained(args.train.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.train.model_name)

    return model, tokenizer


def setup(run_name=None):

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
        wandb.init(project=wandb_project, name=run_name)

    if wandb_api_key:
        return "wandb", run_name or wandb.run.name

    return "none", run_name or "local-run"
