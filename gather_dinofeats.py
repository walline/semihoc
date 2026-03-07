import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

# Only needed when a hierarchy is provided
try:
    from lib.utils.hierarchy_utils import get_leaves_from_json
    from lib.utils.dataset_util import SubsetImageFolder
except Exception:
    get_leaves_from_json = None
    SubsetImageFolder = None

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--savedir", type=str, required=True)
    parser.add_argument("--torchhub_path", type=str, required=True)
    parser.add_argument("--dino_backbone", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--cropsize", type=int, default=224)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=1)

    # New: either provide a hierarchy OR use all classes
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--hierarchy", type=str, default=None,
                       help="Path to a hierarchy JSON. If provided, only leaf classes are used.")
    group.add_argument("--all-classes", action="store_true",
                       help="Use all classes found in datadir/train and datadir/val.")

    return parser.parse_args()

def build_transforms(resize, cropsize):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        normalize,
    ])

def make_datasets(datadir, transform, use_all_classes, hierarchy_path):
    train_dir = os.path.join(datadir, "train")
    val_dir   = os.path.join(datadir, "val")

    if use_all_classes or (hierarchy_path is None):
        # Plain ImageFolder over all classes
        train_ds = datasets.ImageFolder(train_dir, transform=transform)
        val_ds   = datasets.ImageFolder(val_dir,   transform=transform)
        # Was: subset_name = "all_classes"
        subset_name = Path(datadir).resolve().name  # e.g., "office31"
    else:
        # Subset defined by hierarchy leaves
        if get_leaves_from_json is None or SubsetImageFolder is None:
            raise RuntimeError("Hierarchy mode requested but required utilities are unavailable.")
        include_classes = get_leaves_from_json(hierarchy_path)
        subset_name = Path(hierarchy_path).stem
        train_ds = SubsetImageFolder(train_dir, include_classes, transform=transform)
        val_ds   = SubsetImageFolder(val_dir,   include_classes, transform=transform)

    return train_ds, val_ds, subset_name

def infer_feat_dim(model, cropsize):
    with torch.no_grad():
        dummy = torch.zeros(1, 3, cropsize, cropsize, device=DEVICE)
        out = model(dummy)
        # Handle models that might return tuples/dicts
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            # pick first tensor entry
            for v in out.values():
                out = v
                break
        return out.shape[-1]

def run():
    args = parse_args()
    transform = build_transforms(args.resize, args.cropsize)

    train_ds, val_ds, subset_name = make_datasets(
        args.datadir, transform, args.all_classes, args.hierarchy
    )

    trainloader = DataLoader(train_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    valloader   = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    torch.hub.set_dir(args.torchhub_path)
    model = torch.hub.load('facebookresearch/dinov2', args.dino_backbone)
    model.eval().to(DEVICE)

    feat_dim = infer_feat_dim(model, args.cropsize)

    n_train, n_val = len(train_ds), len(val_ds)
    print(f"Device: {DEVICE} | Backbone: {args.dino_backbone} | feat_dim={feat_dim}")
    print(f"Train images: {n_train} | Val images: {n_val} | Subset: {subset_name}")

    train_feats = torch.empty((n_train, feat_dim), dtype=torch.float)
    train_labels = torch.empty(n_train, dtype=torch.long)
    val_feats = torch.empty((n_val, feat_dim), dtype=torch.float)
    val_labels = torch.empty(n_val, dtype=torch.long)

    save_path = os.path.join(args.savedir, subset_name)
    os.makedirs(save_path, exist_ok=True)

    # Predict training set
    index = 0
    for inputs, targets in tqdm(trainloader, desc="Predicting training set"):
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            feats_batch = model(inputs)
            if isinstance(feats_batch, (tuple, list)):
                feats_batch = feats_batch[0]
            if isinstance(feats_batch, dict):
                feats_batch = next(iter(feats_batch.values()))
        bs = inputs.size(0)
        train_feats[index:index+bs] = feats_batch.detach().cpu()
        train_labels[index:index+bs] = targets
        index += bs

    # Predict validation set
    index = 0
    for inputs, targets in tqdm(valloader, desc="Predicting validation set"):
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            feats_batch = model(inputs)
            if isinstance(feats_batch, (tuple, list)):
                feats_batch = feats_batch[0]
            if isinstance(feats_batch, dict):
                feats_batch = next(iter(feats_batch.values()))
        bs = inputs.size(0)
        val_feats[index:index+bs] = feats_batch.detach().cpu()
        val_labels[index:index+bs] = targets
        index += bs

    torch.save(train_feats, os.path.join(save_path, f"train_feats_{args.dino_backbone}.pt"))
    torch.save(train_labels, os.path.join(save_path, "train_labels.pt"))
    torch.save(val_feats,   os.path.join(save_path, f"val_feats_{args.dino_backbone}.pt"))
    torch.save(val_labels,  os.path.join(save_path, "val_labels.pt"))

if __name__ == "__main__":
    run()
