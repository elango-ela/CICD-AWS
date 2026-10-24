"""Run the whole pipeline end to end (same as `dvc repro`, without caching)."""
from pipeline import (stage_01_data_ingestion, stage_02_preprocessing,
                      stage_03_training, stage_04_evaluation)

STAGES = [
    ("Data ingestion", stage_01_data_ingestion),
    ("Preprocessing", stage_02_preprocessing),
    ("Training", stage_03_training),
    ("Evaluation", stage_04_evaluation),
]

if __name__ == "__main__":
    for name, stage in STAGES:
        print(f"\n>>>>>> {name} started <<<<<<")
        stage.main()
        print(f">>>>>> {name} completed <<<<<<")
