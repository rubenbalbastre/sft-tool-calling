"""Publish an existing local Hugging Face dataset without regenerating it."""

import argparse
import os
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from dotenv import load_dotenv
from huggingface_hub import HfApi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data/pipeline/hf_dataset"
DEFAULT_REPO_ID = "rubenbalbastre/supply-chain-tool-calling"
DEFAULT_CARD = PROJECT_ROOT / "src/data_generation/README.md"
SPLITS = ("sft_train", "sft_validation", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload an existing SFT DatasetDict to Hugging Face."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset saved with save_to_disk (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "repo_id",
        nargs="?",
        default=DEFAULT_REPO_ID,
        help=f"Destination dataset repository (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    card_path = args.card.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not card_path.is_file():
        raise FileNotFoundError(f"Dataset card not found: {card_path}")

    dataset = load_from_disk(str(dataset_dir))
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected a DatasetDict in {dataset_dir}, got {type(dataset)}")
    missing = set(SPLITS) - set(dataset)
    if missing:
        raise ValueError(f"Dataset is missing required splits: {sorted(missing)}")
    dataset = DatasetDict({name: dataset[name] for name in SPLITS})

    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        commit_message="Update dataset card",
    )
    dataset.push_to_hub(
        repo_id=args.repo_id,
        config_name=args.config_name,
        private=args.private,
        token=token,
    )
    print(f"Published https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
