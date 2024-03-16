# KKBox's Music Recommendation System

## Project structure

```
project/
│
├── src/                    # Source files
│   ├── models/             # each one's folder
│
├── data/                   # CSV files
│
├── notebooks/              # Jupyter notebooks
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
- In the training data, we use the first 80% as the training set, and last 20% as the validation set, which is the same as the baseline.

## Setup Env
```bash
pip install -r requirements.txt
```

## Train
Uncomment or add `train_pipeline` in [train.py](./src/train.py), then
```bash
cd ./src
python train.py
```

## Test
```bash
cd ./src
python bagging.py
```