from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data_generation.generate_sft_data import build_pipeline_dataset


PROJECT_ROOT = Path(__file__).resolve().parent


@hydra.main(config_path="config", config_name="data_generation", version_base=None)
def main(config: DictConfig) -> None:
    split_sizes = OmegaConf.to_container(config.splits, resolve=True)
    dataset = build_pipeline_dataset(
        split_sizes,
        int(config.seed),
        languages=list(config.prompt_generation.languages),
        template_splits={
            name: list(indices)
            for name, indices in config.prompt_generation.templates.items()
        },
    )

    output_dir = PROJECT_ROOT / config.output_dir
    dataset.save_to_disk(str(output_dir))
    print(dataset)
    print(f"Saved Hugging Face dataset to {output_dir}")


if __name__ == "__main__":
    main()
