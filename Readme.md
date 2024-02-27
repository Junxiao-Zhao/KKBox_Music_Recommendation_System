# KKBox's Music Recommendation System

## Project structure

```
project/
│
├── docs/                   # Documentation files
│
├── src/                    # Source files
│   ├── models/             # each one's folder
│
├── data/                   # CSV files
│
├── notebooks/              # Jupyter notebooks
│
├── scripts/                # Scripts
│
├── config/                 # Configuration files
|
├── tensorboard/            # TensorBoard
|
├── checkpoints/            # Checkpoints
│
├── .gitignore              # Specifies intentionally untracked files to ignore
│
├── README.md               # Project overview and setup instructions
│
└── requirements.txt        # The dependencies file

```

## Data
- We randomly sampled 30% of the data from "train.csv": the first $\displaystyle{\frac{5}{6}}$ is the sampled training data, and last $\displaystyle{\frac{1}{6}}$ is the pseduo test set.
- In the sampled training data, we use the first 80% as the training set, and last 20% as the validation set, which is the same as the baseline.

## Notes
- Only push codes. Do not push csv files.
- Do not push to `main` directly. Each person pushes to their own branch (use the first name as the branch name), then merge to `main`.