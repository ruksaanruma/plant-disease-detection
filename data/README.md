# Data

The raw dataset is **not** committed to the repo (it is large and lives in `.gitignore`).
Download it into `data/raw/` before training.

## Source
**PlantVillage** — 54,305 labeled leaf images, 38 crop/disease classes, all 256×256 RGB.

- Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- GitHub mirror: https://github.com/spMohanty/PlantVillage-Dataset

## Get it

```bash
# Only the 6 classes used for the demo build (fast, ~5k images):
bash scripts/download_data.sh github-subset

# The full 38-class dataset via the GitHub mirror (~2 GB):
bash scripts/download_data.sh github-full

# The full dataset via Kaggle (needs ~/.kaggle/kaggle.json):
bash scripts/download_data.sh kaggle
```

## Expected layout
```
data/raw/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Apple___healthy/
├── Potato___Early_blight/
├── Potato___Late_blight/
└── Potato___healthy/        # ... (38 folders when using the full set)
```
Folder names follow `Crop___Condition`; `___healthy` marks the healthy class for each crop.
