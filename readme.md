# Image Classification: Cats vs Dogs

## Repository layout
- data/ — raw dataset folders (e.g. `PetImages/Cat`, `PetImages/Dog`)
- data_processed/ — outputs from the data split (train / test)
- pretrained_models/ — local weights (e.g. `efficientnet_b0.pth`)
- outputs/ — experiment outputs and logs
- wandb/ — recorded WandB runs
- Scripts:
  - `run_cv_pipeline.py` — main pipeline orchestration
  - `data_setup.py` — data preparation and splitting
  - `train.py` — training logic
  - `engine.py` — training / evaluation loop helpers
  - `utils.py` — utility helpers
  - `download_model.py` — download helper for pretrained weights
  - `catvsdog.ipynb` — exploratory notebook

## Quickstart

1. Install dependencies (see the recorded run `requirements.txt` if needed).
2. Edit configuration: `config_cv.yaml`
3. Prepare data (if using raw `data/source_dirs`):
   - The pipeline uses [`prepare_data`](data_setup.py) to create `data_processed/train` and `data_processed/test`.
4. Run the pipeline:
   - Example:
     ```sh
     python run_cv_pipeline.py
     ```
   - `run_cv_pipeline.py` creates a timestamped run folder under the base `outputs` directory.

## Important files
- Configuration: `config_cv.yaml` — dataset paths, model name, training params.
- Data prep: `data_setup.py` — includes the [`prepare_data`](data_setup.py) function.
- Main pipeline: `run_cv_pipeline.py` — contains the pipeline [`main`](run_cv_pipeline.py).
- Training: `train.py` — model training and evaluation code.
- Pretrained weights: `pretrained_models/` (e.g. `efficientnet_b0.pth`).

## Logging & tracking
- Local experiment outputs go to `outputs/`.
