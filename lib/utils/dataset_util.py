import os
from dataclasses import dataclass

import torch
import torchvision.datasets as datasets
from torch.utils.data import Subset

from collections import defaultdict
import random

_TENSOR_CACHE = {}

def repeated_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def load_tensor_cached(path, map_location="cpu"):
    t = _TENSOR_CACHE.get(path)
    if t is None:
        t = torch.load(path, map_location=map_location)
        if isinstance(t, torch.Tensor):
            t.share_memory_()  # enables sharing across DataLoader workers
        _TENSOR_CACHE[path] = t
    return t


class IndexedDataset(torch.utils.data.Dataset):
    
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        feature, label = self.base_dataset[idx]
        return feature, label, idx


class FeatDataset(torch.utils.data.Dataset):
    
    def __init__(self, all_features, all_labels, source_classes, include_classes):

        assert all_features.shape[0] == all_labels.shape[0]
        assert source_classes is not None
        assert include_classes is not None
        assert source_classes == sorted(source_classes)

        include_classes = sorted(include_classes)

        is_subset = set(include_classes).issubset(source_classes)
        if not is_subset:
            raise ValueError("Specified classes must be a subset of the source classes.")

        # big base tensors
        self._features = all_features
        self._labels = all_labels

        class2idx = {name: idx for idx, name in enumerate(source_classes)}
        self.include_idx = torch.tensor([class2idx[c] for c in include_classes], dtype=torch.long)

        keep = torch.isin(self._labels, self.include_idx)
        self._keep_indices = keep.nonzero(as_tuple=False).squeeze(1)

    def __len__(self):
        return int(self._keep_indices.numel())
    
    def __getitem__(self, i):
        j = int(self._keep_indices[i])
        x = self._features[j]
        y = int(self._labels[j])
        return x, y

class SubsetImageFolder(datasets.ImageFolder):

    def __init__(self,
                 root,
                 include_folders,
                 transform=None,
                 target_transform=None):

        assert len(include_folders) == len(set(include_folders))

        self.include_folders = include_folders
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform)

    def find_classes(self, directory):

        all_classes = [entry.name for entry in os.scandir(directory) if entry.is_dir()]

        if not all_classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        is_subset = set(self.include_folders).issubset(all_classes)
        if not is_subset:
            raise ValueError("Specified classes must be a subset of existing classes.")
        
        classes = sorted(cls for cls in all_classes if cls in self.include_folders)
        
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx 


def subsample_featdataset(dataset, n_per_class: int, seed: int, allow_fewer: bool = False):
    """
    Subsample FeatDataset without assuming remapped/contiguous labels.
    Iterates over the classes specified by dataset.include_idx (global IDs);
    falls back to the unique labels present if include_idx is unavailable.
    Returns a Subset whose indices are *local* positions w.r.t. dataset._keep_indices.
    """
    # Labels of kept items, in *global* id space
    if dataset._keep_indices.numel() == 0:
        return Subset(dataset, [])

    labels_global = dataset._labels[dataset._keep_indices].long()

    # Class iteration order: prefer the declared include set; else use uniques present
    class_ids = getattr(dataset, "include_idx", None)
    if class_ids is None:
        class_ids = torch.unique(labels_global, sorted=True)
    else:
        class_ids = class_ids.long()  # already global ids

    g = torch.Generator().manual_seed(seed)
    chosen_local_all = []

    for cid in class_ids.tolist():
        # local positions (0..len(dataset)-1) within the kept slice
        cls_local = torch.nonzero(labels_global == cid, as_tuple=False).squeeze(1)
        n = int(cls_local.numel())

        if n == 0:
            if allow_fewer:
                # skip entirely if class not present and allow_fewer=True
                continue
            raise ValueError(f"Class (global id) {cid} has 0 samples (requested {n_per_class}).")

        if n < n_per_class and not allow_fewer:
            raise ValueError(f"Class (global id) {cid} has only {n} samples (requested {n_per_class}).")

        k = min(n_per_class, n)
        perm = torch.randperm(n, generator=g)[:k]
        chosen_local_all.append(cls_local[perm])

    if not chosen_local_all:
        return Subset(dataset, [])

    selected_local_indices = torch.cat(chosen_local_all).tolist()
    assert max(selected_local_indices) < len(dataset)  # local sanity check
    return Subset(dataset, selected_local_indices)


def get_id_classes(id_classes_fn):

    with open(id_classes_fn, "r") as f:
        lines = [line.strip() for line in f]

    return sorted(lines)

@dataclass
class LabelView:

    # local view idx -> node idx
    view2node: torch.LongTensor # [V]

    # node idx -> local view idx (or -1 if not present)
    node2view: torch.LongTensor # [N_nodes]

    @classmethod
    def from_local_label_space(cls, hierarchy, local_classes, device="cpu"):

        n_local = len(local_classes)
        node_names = hierarchy.id_node_list
        n_nodes = len(node_names)

        name2idx_node = {n: i for i, n in enumerate(node_names)}
        local_set = set(local_classes)
        name2idx_local = {n: i for i, n in enumerate(local_classes)}

        view2node = torch.empty(n_local, dtype=torch.long)
        node2view = torch.full((n_nodes,), -1, dtype=torch.long)

        for i, local_class in enumerate(local_classes):
            c = local_class
            while c not in name2idx_node:
                c = hierarchy.child2parent.get(c, None)
                if c is None:
                    raise ValueError(f"Class {local_class} has no ancestor in the hierarchy")

            view2node[i] = name2idx_node[c]

        for i, node_name in enumerate(node_names):
            c = node_name

            while c not in local_set:
                c = hierarchy.child2parent.get(c, None)
                if c is None:
                    node2view[i] = -1
                    break
            else:
                node2view[i] = name2idx_local[c]

        return cls(view2node=view2node.to(device),
                   node2view=node2view.to(device))
