"""Upload a local model to Hugging Face and add it to the project collection."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTION = (
    "rubenbalbastre/"
    "2b-tool-calling-using-sft"
)
DEFAULT_NAMESPACE = "rubenbalbastre"
DEFAULT_REPO_PREFIX = "procurement-function-calling"
DEFAULT_MODEL_CARD = PROJECT_ROOT / "src/training/model/README.md"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path, help="Local model or adapter directory")
    parser.add_argument(
        "repo_id",
        nargs="?",
        help=(
            "Destination repository (default: "
            "rubenbalbastre/procurement-function-calling-<model-name>)"
        ),
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--note", help="Optional collection note (maximum 500 characters)")
    parser.add_argument("--card", type=Path, default=DEFAULT_MODEL_CARD)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def default_repo_id(model_dir):
    model_name = (
        model_dir.parent.name if model_dir.name == "final_model" else model_dir.name
    )
    if not model_name.startswith(DEFAULT_REPO_PREFIX):
        model_name = f"{DEFAULT_REPO_PREFIX}-{model_name}"
    return f"{DEFAULT_NAMESPACE}/{model_name}"


def main():
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    card_path = args.card.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not card_path.is_file():
        raise FileNotFoundError(f"Model card not found: {card_path}")
    if not any((model_dir / name).is_file() for name in (
        "config.json", "adapter_config.json"
    )):
        raise ValueError(
            f"{model_dir} is not a recognized model or PEFT adapter directory"
        )

    repo_id = args.repo_id or default_repo_id(model_dir)

    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    api = HfApi(token=token)

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=model_dir,
        commit_message=f"Upload {model_dir.name}",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        commit_message="Update model card",
    )
    api.add_collection_item(
        collection_slug=args.collection,
        item_id=repo_id,
        item_type="model",
        note=args.note,
        exists_ok=True,
    )
    print(f"Published https://huggingface.co/{repo_id}")
    print(f"Added to https://huggingface.co/collections/{args.collection}")


if __name__ == "__main__":
    main()
