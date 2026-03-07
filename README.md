# SemiHOC

This repository contains the reference implementation of **SemiHOC**, a method for **semi-supervised hierarchical open-set classification** from the [WACV 2026 paper](https://arxiv.org/abs/2601.16541).

# Dependencies

The Python requirements are listed in requirements.txt.

# Datasets

This code expects datasets organized in a standard image folder structure:

```
dataset/
    train/
        class1/
        class2/
        ...
    test/
        class1/
        class2/
        ...
```

The hierarchies for the respective datasets are defined in json files in the `hierarchies` directory.
The repository includes hierarchy definitions and ID splits for several datasets:

# Setup

```
DATASET=/path/to/your/dataset/directory/e.g./inat19/
HIERARCHY=/path/to/your/hierarchy/definition/e.g./inat19.json
IDSPLIT=/path/to/file/defining/in-distribution/classes/e.g./data/inat19-id-labels.csv
HUBPATH=/path/to/your/torch/hub/dinov2/models/
FEATSDIR=/directory/for/storing/dino/features/
OOHFEATS=/path/to/directory/containing/out/of/hierarchy/features/e.g./office31/
TRAINDIR=/directory/for/storing/checkpoints/and/results/
```

# Step 1 — Extract DINOv2 Features

SemiHOC uses **precomputed DINOv2 features**.

Run:

```
python gather_dinofeats.py \
  --batch_size 128 \
  --datadir $DATASET \
  --savedir $FEATSDIR \
  --hierarchy $HIERARCHY \
  --dino_backbone "dinov2_vitl14_reg" \
  --torchhub_path $HUBPATH
```

# Step 2 — Train SemiHOC

After feature extraction, train the SemiHOC model. For example

```
python semihoc.py \
  --batch_size 128 \
  --n_per_class 20 \
  --epochs 400 \
  --threshold_schedule cosine \
  --lr 0.001 \
  --dropout 0.3 \
  --threshold 0.95 \
  --lambda_ul 1.0 \
  --datadir $DATASET \
  --featsdir $FEATSDIR \
  --oohfeatsdir $OOHFEATS \
  --hierarchy $HIERARCHY \
  --traindir $TRAINDIR \
  --id_split $IDSPLIT
```

# Results

Training logs and outputs are written to the directory specified `$TRAINDIR`.
