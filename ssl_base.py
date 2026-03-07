import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os
import json
import importlib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from functools import partial

from lib.utils.prediction_util import fuse_predictions
from lib.utils.metric_util import AverageMetric, Accuracy
from lib.utils.schedule_util import threshold_schedule

from lib.models import BatchedMLPs, ModelEMA
from lib.hierarchy import Hierarchy
from lib.utils.dataset_util import (get_id_classes,
                                    load_tensor_cached,
                                    FeatDataset,
                                    subsample_featdataset,
                                    repeated_dataloader,
                                    LabelView)

from lib.utils.hierarchy_utils import (get_multidepth_classes,
                                       get_children_and_group_maps,
                                       get_target_distributions,
                                       get_path_indices,
                                       get_expected_hdist,
                                       get_hdist_matrix,
                                       get_leaf_mask,
                                       )

from lib.eval import get_results


class SSLBase():

    @classmethod
    def get_base_args(cls, parser):
        
        parser.add_argument("--epochs", type=int, default=90)
        parser.add_argument("--eval_every", type=int, default=10)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--hidden_size", type=int, default=512)
        parser.add_argument("--nlayers", type=int, default=4)
        parser.add_argument("--lr", type=float, default=0.05)
        parser.add_argument("--momentum", type=float, default=0.90)
        parser.add_argument("--weight_decay", type=float, default=1e-3)
        parser.add_argument("--nesterov", type=bool, default=False)
        parser.add_argument("--datadir", type=str, required=True)
        parser.add_argument("--featsdir", type=str, required=True)
        parser.add_argument("--oohfeatsdir", type=str, default=None) # features for out-of-hierarchy data
        parser.add_argument("--enable_root", action="store_true")
        parser.add_argument("--sizeroot", type=int, default=100) # nsamples at root
        parser.add_argument("--dino_backbone", type=str, required=True)
        parser.add_argument("--hierarchy", type=str, required=True)
        parser.add_argument("--tag", type=str, default="")
        parser.add_argument("--traindir", type=str, required=True)
        parser.add_argument("--extradir", type=str, default="")
        parser.add_argument("--epochs_warmup", type=int, default=0)
        parser.add_argument("--id_split", type=str, required=True)
        parser.add_argument("--log_interval", type=int, default=1000)
        parser.add_argument("--ema_decay", type=float, default=0.999)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--lr_decay", action="store_true")
        parser.add_argument("--uncertainty_method", type=str, required=True)
        parser.add_argument("--uncertainty_args", type=str, default="{}")
        parser.add_argument("--extra", type=float, default=0.0)

        return parser

    @classmethod
    def get_custom_args(cls, parser):

        parser.add_argument("--n_per_class", type=int, default=100)
        parser.add_argument("--mu", type=int, default=4)
        parser.add_argument("--labelseed", type=int, default=123)
        parser.add_argument("--allow_fewer", default=False, action="store_true")
        parser.add_argument("--threshold", type=float, default=0.95)
        parser.add_argument("--lambda_ul", type=float, default=1.0)
        parser.add_argument("--threshold_schedule",
                            type=str,
                            choices=["constant", "cosine", "linear", "inverse_sqrt"],
                            default="constant",)

        return parser

    
    def extra_tensor(self, size, device="cuda"):
        n_bytes = int(size * (1024**3))
        buf = torch.empty(n_bytes, dtype=torch.uint8, device=device)
        torch.cuda.synchronize()
        return buf


    def __init__(self, args):

        self.id_classes = get_id_classes(args.id_split)
        self.hierarchy = Hierarchy(self.id_classes, args.hierarchy)
        self.ood_classes = self.hierarchy.ood_train_classes

        self.nr_heights = self.hierarchy._max_depth
        
        self.multi_classes = get_multidepth_classes(self.hierarchy,
                                                    self.id_classes)

        self.flattened_classes = [item for sublist in reversed(self.multi_classes)
                                  for item in sublist]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.extra = self.extra_tensor(args.extra, device=self.device)

        self.uncertainty_args = json.loads(args.uncertainty_args)

        flat_labelview = LabelView.from_local_label_space(self.hierarchy,
                                                          self.flattened_classes,
                                                          device=self.device)
        self.flat2node = flat_labelview.view2node
        self.node2flat = flat_labelview.node2view

        score_module = importlib.import_module("lib.utils.score_util")
        self.uncertainty_method = getattr(score_module, args.uncertainty_method)

        self.enable_root = args.enable_root

        gt_dists_mat, pred_dists_mat = get_hdist_matrix(self.hierarchy,
                                                        range(len(self.hierarchy.id_node_list)),
                                                        return_pair=True)

        hdist_mat = gt_dists_mat + pred_dists_mat        
        self.hdist_mat = hdist_mat.float().to(self.device)

        self.gt_dists_mat = gt_dists_mat.long()
        self.pred_dists_mat = pred_dists_mat.long()

        descendant_mask = self.hierarchy.gen_descendant_mask(self.hierarchy.id_node_list)
        self.descendant_mask = descendant_mask.to(self.device)

        self.depth_map = self.hierarchy.gen_depth_map(self.hierarchy.id_node_list)
        self.nchild_map = self.hierarchy.gen_nchild_map(self.hierarchy.id_node_list)        
        
        hierarchy_name = Path(args.hierarchy).stem
        id_split_name = Path(args.id_split).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.__class__.__name__

        # Build experiment name with key hyperparams
        ooh_str = f"ooh={Path(args.oohfeatsdir).name}" if args.oohfeatsdir else "ooh=None"

        experiment_name = (
            f"{model_name}"
            f"-umethod={args.uncertainty_method}"
            f"_nlab={args.n_per_class}"
            f"_do={args.dropout}"
            f"_lr={args.lr}"
            f"_th={args.threshold}"
            f"_wd={args.weight_decay}"
            f"_{ooh_str}"
            f"-{args.tag}-{timestamp}"
        )

        experiment_dir = os.path.join(args.traindir,
                                      hierarchy_name,
                                      id_split_name,
                                      args.extradir,
                                      experiment_name)

        self.experiment_dir = experiment_dir

        self.checkpoint_fn = os.path.join(experiment_dir, "checkpoint.pt")
        print('checkpoint filename: %s', self.checkpoint_fn)

        tensorboard_dir = os.path.join(experiment_dir, "tensorboard")
        self.summary_writer = SummaryWriter(tensorboard_dir)

        
    def setup(self, args):

        n_classes = [len(classes) for classes in reversed(self.multi_classes)]

        models = BatchedMLPs(self.nr_heights,
                             self.feat_dim,
                             args.hidden_size,
                             n_classes,
                             n_layers=args.nlayers)

        print(models)
        print(f"Device: {self.device}")
        self.models = models.to(self.device)

        self.ema = ModelEMA(self.device, self.models, args.ema_decay)

        self.depth_transforms = []
        for h in range(self.nr_heights):
            local_classes = self.multi_classes[-(h+1)]
            label_view = LabelView.from_local_label_space(self.hierarchy,
                                                          local_classes,
                                                          device=self.device)
            self.depth_transforms.append(label_view)

        self.train_metrics = {
            "train/loss_labeled": AverageMetric(),
            "train/loss_ul": AverageMetric(),
            "train/loss": AverageMetric(),
            "train/mask": AverageMetric(),
            "train/mask_leafrate": AverageMetric(),
            "train/k": AverageMetric(),
            "train/pred_id_ratio": AverageMetric(),
            "train/id_purity": AverageMetric(),
            "train/ood_purity": AverageMetric(),
        }

        self.test_metrics = {}

        for h in range(self.nr_heights):
            self.train_metrics[f"train/loss_height{h}"] = AverageMetric()
            self.test_metrics[f"test/acc_height{h}"] = Accuracy((1,))
            self.test_metrics[f"test/ood_acc_height{h}"] = Accuracy((1,))

        self.target_distributions = get_target_distributions(self.hierarchy,
                                                             self.multi_classes,
                                                             device=self.device)
        self.leaf_mask = get_leaf_mask(self.hierarchy, device=self.device)

        self.optimizer = optim.SGD(self.models.parameters(),
                                   lr=args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay,
                                   nesterov=args.nesterov)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                              args.epochs - args.epochs_warmup,
                                                              eta_min=0)

        children_maps, group_sizes = get_children_and_group_maps(self.hierarchy,
                                                                 self.multi_classes,
                                                                 device=self.device)

        self.children_maps = children_maps
        self.group_sizes = group_sizes

        leaf_height = 0
        path_indices = get_path_indices(self.hierarchy, self.multi_classes, leaf_height)
        path_indices = [torch.tensor(x, dtype=torch.long, device=self.device) for x in path_indices]
        self.path_indices = path_indices

        self.threshold_scheduler = partial(
            threshold_schedule,
            schedule=args.threshold_schedule,
            start=1.0,
            end=args.threshold,
            warmup_epochs=args.epochs_warmup,
            total_epochs=args.epochs)

        self.threshold = self.threshold_scheduler(0)

    def setup_data(self, args):

        all_classes = sorted(self.id_classes + self.ood_classes)
        ds_label_space = all_classes + ["root"]

        feats_train_name = f"train_feats_{args.dino_backbone}.pt"
        labels_train_name = "train_labels.pt"

        feats_val_name = f"val_feats_{args.dino_backbone}.pt"
        labels_val_name = "val_labels.pt"

        feats_train_path = os.path.join(args.featsdir, feats_train_name)
        labels_train_path = os.path.join(args.featsdir, labels_train_name)

        feats_val_path = os.path.join(args.featsdir, feats_val_name)
        labels_val_path = os.path.join(args.featsdir, labels_val_name)

        print("Loading data tensors...")
        feats_train = load_tensor_cached(feats_train_path)
        labels_train = load_tensor_cached(labels_train_path)
        feats_val = load_tensor_cached(feats_val_path)
        labels_val = load_tensor_cached(labels_val_path)

        print("Loading labeled training data...")
        id_train_ds_all = FeatDataset(feats_train,
                                      labels_train,
                                      all_classes,
                                      self.id_classes)

        print("Subsampling the labeled training set...")
        id_train_ds = subsample_featdataset(id_train_ds_all,
                                            args.n_per_class,
                                            args.labelseed,
                                            allow_fewer=args.allow_fewer)

        print("Loading ID test data...")
        id_val_ds = FeatDataset(feats_val,
                                labels_val,
                                all_classes,
                                self.id_classes)

        print("Loading OOD test data...")
        ood_val_ds = FeatDataset(feats_val,
                                 labels_val,
                                 all_classes,
                                 self.ood_classes)

        print("Loading unlabeled training data...")
        ul_train_ds = FeatDataset(feats_train,
                                  labels_train,
                                  all_classes,
                                  all_classes)

        if args.oohfeatsdir is not None:
            feats_root_train_path = os.path.join(args.oohfeatsdir, feats_train_name)
            feats_root_val_path = os.path.join(args.oohfeatsdir, feats_val_name)
            train_feats_root = load_tensor_cached(feats_root_train_path)
            val_feats_root = load_tensor_cached(feats_root_val_path)

            # random sample
            indices_train = torch.randperm(len(train_feats_root))[:args.sizeroot]
            train_feats_root = train_feats_root[indices_train]

            indices_val = torch.randperm(len(val_feats_root))[:args.sizeroot]
            val_feats_root = val_feats_root[indices_val]

            root_labels = torch.full((args.sizeroot,), len(all_classes), dtype=torch.long)

            root_train_ds = FeatDataset(train_feats_root, root_labels, ds_label_space, ds_label_space)
            ul_train_ds = ConcatDataset([ul_train_ds, root_train_ds])

            root_val_ds = FeatDataset(val_feats_root, root_labels, ds_label_space, ds_label_space)
            ood_val_ds = ConcatDataset([ood_val_ds, root_val_ds])

        tmp_feat, _ = id_train_ds[0]
        self.feat_dim = tmp_feat.size(-1)

        self.id_train_ds = id_train_ds
        self.ul_train_ds = ul_train_ds
        self.id_val_ds = id_val_ds
        self.ood_val_ds = ood_val_ds

        self.ds_view = LabelView.from_local_label_space(self.hierarchy,
                                                        ds_label_space,
                                                        device=self.device)

        print(f"Feature dimension: {self.feat_dim}")
        print(f"Labeled training set size (ID): {len(id_train_ds)}")
        print(f"Validation set size (ID): {len(id_val_ds)}")
        print(f"Validation set size (OOD): {len(ood_val_ds)}")
        print(f"Unlabeled training set size: {len(ul_train_ds)}")        

    
    def create_data_loaders(self, args):
        
        self.labeled_loader = DataLoader(self.id_train_ds,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         pin_memory=True,
                                         drop_last=True,
                                         )

        self.ul_loader = DataLoader(self.ul_train_ds,
                                    batch_size=args.mu * args.batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=False,
                                    )

        self.id_val_loader = DataLoader(self.id_val_ds,
                                        batch_size=args.batch_size,
                                        pin_memory=True,
                                        )
        
        self.ood_val_loader = DataLoader(self.ood_val_ds,
                                         batch_size=args.batch_size,
                                         pin_memory=True,
                                         )


        self.labeled_eval_loader = DataLoader(self.id_train_ds,
                                              batch_size=args.batch_size,
                                              pin_memory=True,
                                              )



    def train_step(self, inputs, targets, ul_inputs, ul_targets, epoch, global_step, args):

        labeled_batch_size = inputs.size(0)
        ul_batch_size = ul_inputs.size(0)
        
        x = torch.concat([inputs, ul_inputs], dim=0)
        outputs = self.models(x, dropout=args.dropout)

        with torch.no_grad():
            teacher_outputs = self.ema.ema(ul_inputs, dropout=0.0)

        loss_labeled = 0.0

        softmax_preds_ul = []
        softmax_preds_teacher = []

        node_targets = self.ds_view.view2node[targets]        

        for h in range(self.nr_heights):

            local_targets = self.depth_transforms[h].node2view[node_targets]
            preds = outputs[h]
            preds_l = preds[:labeled_batch_size]
            preds_ul = preds[labeled_batch_size:]

            preds_teacher = teacher_outputs[h]

            softmax_preds = torch.softmax(preds_ul, dim=-1)
            softmax_preds_ul.append(softmax_preds)

            p_teacher = torch.softmax(preds_teacher, dim=-1)
            softmax_preds_teacher.append(p_teacher)

            height_loss = F.cross_entropy(preds_l, local_targets)
            self.train_metrics[f"train/loss_height{h}"].update_state(height_loss.item(), 1)

            loss_labeled += height_loss

        with torch.no_grad():
            fused_p = fuse_predictions(softmax_preds_teacher,
                                       self.hierarchy,
                                       self.multi_classes,
                                       self.children_maps,
                                       self.group_sizes,
                                       self.path_indices,
                                       self.flat2node,
                                       self.uncertainty_method,
                                       self.uncertainty_args,
                                       enable_root=self.enable_root)

            max_probs, pred_classes = torch.max(fused_p, dim=1)

        mask = max_probs > self.threshold
        mask_float = mask.float()

        mask_classes = pred_classes[mask]

        mask_id_mask = self.leaf_mask[mask_classes]

        mask_classes_is_leaf = self.leaf_mask[mask_classes]

        ul_node_labels = self.ds_view.view2node[ul_targets]
        mask_labels = ul_node_labels[mask]

        mask_id_purity = (mask_classes[mask_id_mask] == mask_labels[mask_id_mask]).float().mean()
        mask_ood_purity = (mask_classes[~mask_id_mask] == mask_labels[~mask_id_mask]).float().mean()

        loss_ul = 0.0

        # if pretraining use only ID pseudolabels
        if args.epochs_warmup >= epoch:
            mask_float[mask] *= mask_classes_is_leaf.float()

        for h in range(self.nr_heights):
            target_dists = self.target_distributions[h][pred_classes]
            log_probs = torch.log(softmax_preds_ul[h] + 1e-8)
            kl_loss = F.kl_div(log_probs, target_dists, reduction="none").sum(dim=-1)
            kl_loss = kl_loss * mask_float
            loss_ul += kl_loss.sum() / ul_batch_size
        
        loss = loss_labeled + args.lambda_ul * loss_ul

        self.train_metrics["train/loss_labeled"].update_state(loss_labeled.item(), 1)
        self.train_metrics["train/loss_ul"].update_state(loss_ul.item(), 1)
        self.train_metrics["train/loss"].update_state(loss.item(), 1)
        self.train_metrics["train/mask"].update_state(mask_float.mean().item(), 1)
        self.train_metrics["train/id_purity"].update_state(mask_id_purity.item(), 1)
        self.train_metrics["train/ood_purity"].update_state(mask_ood_purity.item(), 1)

        return loss


    def train(self, args):

        global_step = 0

        labeled_iterator = repeated_dataloader(self.labeled_loader)

        for epoch in range(args.epochs):

            self.models.train()

            self.threshold = self.threshold_scheduler(epoch)

            for ul_inputs, ul_targets in tqdm(self.ul_loader, desc=f"Training epoch {epoch}"):

                ul_inputs = ul_inputs.to(self.device, non_blocking=True)
                ul_targets = ul_targets.to(self.device, non_blocking=True)

                inputs, targets = next(labeled_iterator)
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                loss = self.train_step(inputs, targets, ul_inputs, ul_targets, epoch, global_step, args)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.ema.update(self.models)

                global_step += 1

                if global_step % args.log_interval == 0:

                    self.summary_writer.add_scalar("train/epoch", epoch, global_step)
                    self.summary_writer.add_scalar("train/threshold", self.threshold, global_step)
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


    def evaluate(self, epoch, args):

        self.models.eval()

        def eval_loop(loader, dset):

            index = 0

            n_samples = len(loader.dataset)

            labels = torch.empty(n_samples, dtype=torch.long, pin_memory=True)
            preds = torch.empty(n_samples, dtype=torch.long, pin_memory=True)
            preds_minhdist = torch.zeros(n_samples, dtype=torch.long, pin_memory=True)

            with torch.inference_mode():

                for inputs, targets in tqdm(loader, desc=f"Evaluating {dset} epoch {epoch}"):

                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    size_batch = inputs.size(0)

                    node_targets = self.ds_view.view2node[targets]
                    
                    outputs = self.ema.ema(inputs)

                    softmax_preds = []

                    for h in range(self.nr_heights):
                        logits = outputs[h]
                        p = torch.softmax(logits, dim=-1)
                        softmax_preds.append(p)

                        local_targets = self.depth_transforms[h].node2view[node_targets]

                        if dset == "id":
                            self.test_metrics[f"test/acc_height{h}"].update_state(logits,
                                                                                  local_targets)

                        if dset == "ood":
                            valid_mask = local_targets != -1
                            if valid_mask.any():
                                valid_logits = logits[valid_mask]
                                valid_targets = local_targets[valid_mask]
                                self.test_metrics[f"test/ood_acc_height{h}"].update_state(valid_logits,
                                                                                          valid_targets)

                    fused_p = fuse_predictions(softmax_preds,
                                               self.hierarchy,
                                               self.multi_classes,
                                               self.children_maps,
                                               self.group_sizes,
                                               self.path_indices,
                                               self.flat2node,
                                               self.uncertainty_method,
                                               self.uncertainty_args,
                                               enable_root=self.enable_root)

                    max_probs, pred_classes = torch.max(fused_p, dim=1)

                    expected_hdists = get_expected_hdist(fused_p, self.hdist_mat)
                    _, pred_minhdist = torch.min(expected_hdists, dim=1)

                    preds[index:(index+size_batch)] = pred_classes.to("cpu", non_blocking=True)
                    preds_minhdist[index:(index+size_batch)] = pred_minhdist.to("cpu", non_blocking=True)
                    labels[index:(index+size_batch)] = targets.to("cpu", non_blocking=True)

                    index += size_batch

                torch.cuda.synchronize()

            return preds, preds_minhdist, labels

        id_preds, id_preds_minhd, id_labels = eval_loop(self.id_val_loader, "id")
        ood_preds, ood_preds_minhd, ood_labels = eval_loop(self.ood_val_loader, "ood")

        train_preds, _, train_labels = eval_loop(self.labeled_eval_loader, "train")

        view2node_cpu = self.ds_view.view2node.detach().cpu()        
        id_node_labels = view2node_cpu[id_labels]
        ood_node_labels = view2node_cpu[ood_labels]
        train_node_labels = view2node_cpu[train_labels]

        dists_mats = (self.gt_dists_mat, self.pred_dists_mat)
        res_id = get_results(id_preds, id_node_labels, self.hierarchy, dists_mats=dists_mats)
        res_ood = get_results(ood_preds, ood_node_labels, self.hierarchy, dists_mats=dists_mats)

        avgdepths_id = self.depth_map[id_preds].float().mean()
        avgdepths_ood = self.depth_map[ood_preds].float().mean()
        avgdepths_id_minhd = self.depth_map[id_preds_minhd].float().mean()
        avgdepths_ood_minhd = self.depth_map[ood_preds_minhd].float().mean()

        res_minhd_id = get_results(id_preds_minhd, id_node_labels, self.hierarchy, dists_mats=dists_mats)
        res_minhd_ood = get_results(ood_preds_minhd, ood_node_labels, self.hierarchy, dists_mats=dists_mats)

        res_train = get_results(train_preds, train_node_labels, self.hierarchy, dists_mats=dists_mats)

        torch.save(res_id,
                   os.path.join(self.experiment_dir, f"id_results_epoch{epoch}.pth"))
        torch.save(res_ood,
                   os.path.join(self.experiment_dir, f"ood_results_epoch{epoch}.pth"))

        mix_bacc = 0.5*(res_id["balanced_acc"] + res_ood["balanced_acc"])
        mix_bmhd = 0.5*(res_id["balanced_hdist"] + res_ood["balanced_hdist"])

        mix_bacc_minhd = 0.5*(res_minhd_id["balanced_acc"] + res_minhd_ood["balanced_acc"])
        mix_bmhd_minhd = 0.5*(res_minhd_id["balanced_hdist"] + res_minhd_ood["balanced_hdist"])

        train_bacc = res_train["balanced_acc"]
        train_bmhd = res_train["balanced_hdist"]

        self.summary_writer.add_scalar("train/bacc", train_bacc, epoch)
        self.summary_writer.add_scalar("train/bmhd", train_bmhd, epoch)

        self.summary_writer.add_scalar("test/bacc_id", res_id["balanced_acc"], epoch)
        self.summary_writer.add_scalar("test/bacc_ood", res_ood["balanced_acc"], epoch)
        self.summary_writer.add_scalar("test/bmhd_id", res_id["balanced_hdist"], epoch)
        self.summary_writer.add_scalar("test/bmhd_ood", res_ood["balanced_hdist"], epoch)
        self.summary_writer.add_scalar("test/bacc_mix", mix_bacc, epoch)
        self.summary_writer.add_scalar("test/bmhd_mix", mix_bmhd, epoch)
        self.summary_writer.add_scalar("test/bacc_id_minhd", res_minhd_id["balanced_acc"], epoch)
        self.summary_writer.add_scalar("test/bacc_ood_minhd", res_minhd_ood["balanced_acc"], epoch)
        self.summary_writer.add_scalar("test/bmhd_id_minhd", res_minhd_id["balanced_hdist"], epoch)
        self.summary_writer.add_scalar("test/bmhd_ood_minhd", res_minhd_ood["balanced_hdist"], epoch)
        self.summary_writer.add_scalar("test/bacc_mix_minhd", mix_bacc_minhd, epoch)
        self.summary_writer.add_scalar("test/bmhd_mix_minhd", mix_bmhd_minhd, epoch)
        self.summary_writer.add_scalar("test/avgdepth_id", avgdepths_id.item(), epoch)
        self.summary_writer.add_scalar("test/avgdepth_ood", avgdepths_ood.item(), epoch)
        self.summary_writer.add_scalar("test/avgdepth_id_minhd", avgdepths_id_minhd.item(), epoch)
        self.summary_writer.add_scalar("test/avgdepth_ood_minhd", avgdepths_ood_minhd.item(), epoch)

        for key, val in self.test_metrics.items():
            res = val.result()
            print(f"{key} - {res}")
            if not torch.isnan(res):
                self.summary_writer.add_scalar(key, res.item(), epoch)
            self.summary_writer.add_scalar(key, res, epoch)
            val.reset_state()

        if self.checkpoint_fn is not None:
            print("Saving...")
            torch.save(self.ema.ema.state_dict(), self.checkpoint_fn)


def main(args):

    model = SSLBase(args)
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
