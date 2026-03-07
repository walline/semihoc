import argparse
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from tqdm import tqdm

from ssl_base import SSLBase
from lib.utils.dataset_util import (FeatDataset,
                                    load_tensor_cached,
                                    LabelView)


class ProHOC(SSLBase):


    def setup_data(self, args):
        # Label spaces
        all_classes = sorted(self.id_classes + self.ood_classes)
        ds_label_space = all_classes + ["root"]

        # Feature file names
        feats_train_name = f"train_feats_{args.dino_backbone}.pt"
        feats_val_name = f"val_feats_{args.dino_backbone}.pt"
        labels_train_name = "train_labels.pt"
        labels_val_name = "val_labels.pt"

        # Paths
        feats_train_path = os.path.join(args.featsdir, feats_train_name)
        labels_train_path = os.path.join(args.featsdir, labels_train_name)
        feats_val_path = os.path.join(args.featsdir, feats_val_name)
        labels_val_path = os.path.join(args.featsdir, labels_val_name)

        # Load tensors (match SSLBase)
        feats_train = load_tensor_cached(feats_train_path)
        labels_train = load_tensor_cached(labels_train_path)
        feats_val = load_tensor_cached(feats_val_path)
        labels_val = load_tensor_cached(labels_val_path)

        # ID train/val and OOD val from the in-hierarchy validation set
        id_train_ds = FeatDataset(feats_train, labels_train, all_classes, self.id_classes)
        id_val_ds = FeatDataset(feats_val, labels_val, all_classes, self.id_classes)
        ood_val_ds = FeatDataset(feats_val, labels_val, all_classes, self.ood_classes)

        # Optionally append 'root' data into OOD val (like SSLBase)
        if args.oohfeatsdir:
            feats_root_val_path = os.path.join(args.oohfeatsdir, feats_val_name)
            val_feats_root = load_tensor_cached(feats_root_val_path)

            # Sample exactly sizeroot items (fall back to all if smaller)
            n_take = min(args.sizeroot, len(val_feats_root))
            idx = torch.randperm(len(val_feats_root))[:n_take]
            val_feats_root = val_feats_root[idx]

            # Label index for 'root' is len(all_classes)
            root_labels_val = torch.full((val_feats_root.size(0),),
                                         len(all_classes),
                                         dtype=torch.long)

            root_val_ds = FeatDataset(val_feats_root,
                                      root_labels_val,
                                      ds_label_space,
                                      ds_label_space)
            ood_val_ds = ConcatDataset([ood_val_ds, root_val_ds])

        # Feature dim for head construction
        tmp_feat, _ = id_train_ds[0]
        self.feat_dim = tmp_feat.size(-1)

        # Expose datasets for evaluate()
        self.id_train_ds = id_train_ds
        self.id_val_ds = id_val_ds
        self.ood_val_ds = ood_val_ds

        # Dataset → node mapping including 'root'
        self.ds_view = LabelView.from_local_label_space(self.hierarchy,
                                                        ds_label_space,
                                                        device=self.device)


    def create_data_loaders(self, args):

        # Loaders (match names expected by SSLBase.evaluate)
        self.train_loader = DataLoader(self.id_train_ds,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       drop_last=True,
                                       pin_memory=True)

        self.id_val_loader = DataLoader(self.id_val_ds,
                                        batch_size=args.batch_size,
                                        pin_memory=True)

        self.ood_val_loader = DataLoader(self.ood_val_ds,
                                         batch_size=args.batch_size,
                                         pin_memory=True)

        # For train-set eval metrics in SSLBase.evaluate
        self.labeled_eval_loader = DataLoader(self.id_train_ds,
                                              batch_size=args.batch_size,
                                              pin_memory=True)
        

    def train_step(self, inputs, targets, epoch, global_step, args):
        outputs = self.models(inputs, dropout=args.dropout)

        # dataset labels -> node labels -> local view per height
        node_targets = self.ds_view.view2node[targets]

        loss = 0.0
        for h in range(self.nr_heights):
            local_targets = self.depth_transforms[h].node2view[node_targets]
            preds = outputs[h]
            height_loss = F.cross_entropy(preds, local_targets)
            self.train_metrics[f"train/loss_height{h}"].update_state(height_loss.item(), 1)
            loss += height_loss

        self.train_metrics["train/loss"].update_state(loss.item(), 1)
        return loss

    
    def train(self, args):
        global_step = 0
        for epoch in range(args.epochs):
            self.models.train()
            for inputs, targets in tqdm(self.train_loader, desc=f"Training epoch {epoch}"):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                loss = self.train_step(inputs, targets, epoch, global_step, args)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema.update(self.models)

                global_step += 1

                if global_step % args.log_interval == 0:
                    self.summary_writer.add_scalar("train/epoch", epoch, global_step)
                    self.summary_writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], global_step)
                    for key, val in self.train_metrics.items():
                        res = val.result()
                        if not torch.isnan(res):
                            self.summary_writer.add_scalar(key, res, global_step)
                        val.reset_state()

            if args.epochs_warmup <= epoch and args.lr_decay:
                self.scheduler.step()

            eval_every = max(1, int(getattr(args, "eval_every", 1)))
            is_eval_epoch = (epoch == 0) or ((epoch + 1) % eval_every == 0) or (epoch == args.epochs - 1)

            if is_eval_epoch:
                self.evaluate(epoch, args)


def main(args):

    model = ProHOC(args)
    model.setup_data(args)
    model.create_data_loaders(args)
    model.setup(args)
    model.train(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = SSLBase.get_base_args(parser)
    parser = SSLBase.get_custom_args(parser)
    args = parser.parse_args()

    main(args)
