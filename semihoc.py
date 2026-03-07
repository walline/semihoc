import argparse
import os
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F

from lib.utils.prediction_util import fuse_predictions
from lib.utils.metric_util import AverageMetric

from lib.utils.dataset_util import IndexedDataset, repeated_dataloader

from ssl_base import SSLBase


class SemiHOC(SSLBase):

    
    def setup(self, args):

        super().setup(args)

        for h in range(self.nr_heights):
            self.train_metrics[f"train/height{h}_mask"] = AverageMetric()

        self.train_metrics["train/idset_purity"] = AverageMetric()
        self.train_metrics["train/oodset_purity"] = AverageMetric()
        self.train_metrics["train/idset_avgdepth"] = AverageMetric()
        self.train_metrics["train/oodset_avgdepth"] = AverageMetric()

        self.train_metrics["train/fallback_ratio_id"] = AverageMetric()
        self.train_metrics["train/fallback_ratio_ood"] = AverageMetric()

        self.train_metrics["train/fb_pre_desc_true_id"] = AverageMetric()
        self.train_metrics["train/fb_pre_desc_true_ood"] = AverageMetric()
        self.train_metrics["train/fb_true_not_desc_pre_id"] = AverageMetric()
        self.train_metrics["train/fb_true_not_desc_pre_ood"] = AverageMetric()

        self.depth_map_device = self.depth_map.to(self.device)

        n = self.descendant_mask.size(0)
        eye = torch.eye(n, dtype=self.descendant_mask.dtype, device=self.descendant_mask.device)
        self.descendant_mask_strict = (self.descendant_mask - eye).clamp_min(0)

        self.precompute_relations()
        
        self.assignment_epochs = dict()
        self.latest_assigned_class = dict()
        self.epoch_assigned_class = dict()
        
        self.epoch_cutoffs = dict()
        self.root_idx = self.hierarchy.id_node_list.index("root")
        self.epoch_cutoffs[self.root_idx] = float("inf")


    def precompute_relations(self):
        dm = self.descendant_mask.bool().cpu()   # inclusive: dm[a, d] = True iff d is descendant of a
        n = dm.shape[0]
        self.strict_anc, self.strict_desc, self.anc_incl = [], [], []
        for c in range(n):
            anc = torch.nonzero(dm[:, c], as_tuple=False).squeeze(1).tolist()   # inclusive ancestors of c
            desc = torch.nonzero(dm[c, :], as_tuple=False).squeeze(1).tolist()  # inclusive descendants of c
            self.anc_incl.append(anc)  # keep inclusive

            # strict versions
            if c in anc: anc.remove(c)
            if c in desc: desc.remove(c)
            self.strict_anc.append(anc)
            self.strict_desc.append(desc)


    def setup_data(self, args):

        super().setup_data(args)
        self.ul_train_ds = IndexedDataset(self.ul_train_ds)

    def train_step(self, inputs, targets, ul_inputs, ul_targets, ul_idxs, epoch, global_step, args):

        labeled_batch_size = inputs.size(0)
        ul_batch_size = ul_inputs.size(0)
        
        x = torch.concat([inputs, ul_inputs], dim=0)
        outputs = self.models(x, dropout=args.dropout)

        with torch.no_grad():
            teacher_outputs = self.ema.ema(ul_inputs)

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
                                       enable_root=self.enable_root,
                                       )

            cummulative_p = torch.matmul(fused_p, self.descendant_mask.T)
            cummulative_p[:, self.root_idx] = 1.0 # avoid numerical issues where root prob is not one

        element_mask = cummulative_p >= self.threshold
        cummulative_p[~element_mask] = float('inf')

        mask = element_mask.any(dim=1)

        confs, pred_classes = cummulative_p.min(dim=1)
        
        mask_float = mask.float()

        pseudo_classes = pred_classes[mask]

        ul_idxs_masked = ul_idxs[mask.cpu()]

        pseudo_classes_filtered = []
        fallback_mask = []

        for idx, cls in zip(ul_idxs_masked.tolist(), pseudo_classes.tolist()):

            final_class = None
            key = (idx, cls)
            cutoff = self.epoch_cutoffs.get(cls, float("inf"))
            assignment = self.assignment_epochs.get(key, epoch)

            if assignment <= cutoff or epoch < args.cutoff_warmup:
                # direct assignment
                final_class = cls
                fallback_mask.append(False)

            else:

                class_name = self.hierarchy.id_node_list[cls]
                fallback_mask.append(True)

                for ancestor_idx in reversed(self.hierarchy.node_ancestors[class_name]):
                    ancestor_key = (idx, ancestor_idx)
                    ancestor_cutoff = self.epoch_cutoffs.get(ancestor_idx, float("inf"))
                    ancestor_assignment = self.assignment_epochs.get(ancestor_key, epoch)

                    if ancestor_assignment <= ancestor_cutoff:
                        final_class = ancestor_idx
                        key = ancestor_key
                        break

            assert final_class is not None
            pseudo_classes_filtered.append(final_class)

        for idx, class_ in zip(ul_idxs_masked.to("cpu").tolist(),
                               pseudo_classes_filtered):
            self.epoch_assigned_class[idx] = class_

        pseudo_classes_prefilter = pseudo_classes
        pseudo_classes = torch.tensor(pseudo_classes_filtered, device=self.device)
        fallback_mask = torch.tensor(fallback_mask, dtype=torch.bool, device=self.device)

        ul_node_labels = self.ds_view.view2node[ul_targets]

        pseudo_node_labels = ul_node_labels[mask]

        pseudo_hits = self.descendant_mask[pseudo_classes, pseudo_node_labels] > 0
        pseudo_hits_float = pseudo_hits.float()

        ul_is_id = self.leaf_mask[ul_node_labels]
        ul_is_id_masked = ul_is_id[mask]

        # Booleans on the masked subset (same length as fallback_mask)
        pre_desc_of_true = (self.descendant_mask_strict[pseudo_node_labels, pseudo_classes_prefilter] > 0)
        true_desc_of_pre = (self.descendant_mask[pseudo_classes_prefilter, pseudo_node_labels] > 0)
        true_not_desc_of_pre = ~true_desc_of_pre

        fb = fallback_mask
        id_mask = ul_is_id_masked
        ood_mask = ~ul_is_id_masked

        def safe_mean_bool(x):
            # x is a boolean tensor
            if x.numel() == 0:
                return None
            return x.float().mean().cpu()        

        # 1) Rate: pre is descendant of true
        rate_pre_desc_true_id  = safe_mean_bool(pre_desc_of_true[fb & id_mask])
        rate_pre_desc_true_ood = safe_mean_bool(pre_desc_of_true[fb & ood_mask])

        # 2) Rate: true is NOT descendant of pre (i.e., pre is NOT ancestor of true)
        rate_true_not_desc_pre_id  = safe_mean_bool(true_not_desc_of_pre[fb & id_mask])
        rate_true_not_desc_pre_ood = safe_mean_bool(true_not_desc_of_pre[fb & ood_mask])

        # Log (skip None to avoid NaNs in your tracker)
        if rate_pre_desc_true_id is not None:
            self.train_metrics["train/fb_pre_desc_true_id"].update_state(rate_pre_desc_true_id, 1)
        if rate_pre_desc_true_ood is not None:
            self.train_metrics["train/fb_pre_desc_true_ood"].update_state(rate_pre_desc_true_ood, 1)

        if rate_true_not_desc_pre_id is not None:
            self.train_metrics["train/fb_true_not_desc_pre_id"].update_state(rate_true_not_desc_pre_id, 1)
        if rate_true_not_desc_pre_ood is not None:
            self.train_metrics["train/fb_true_not_desc_pre_ood"].update_state(rate_true_not_desc_pre_ood, 1)

        # --- Fallback ratios (split by ID vs OOD) ---
        id_fb_ratio  = fallback_mask[ul_is_id_masked].float().mean().cpu() if ul_is_id_masked.any() else None
        ood_fb_ratio = fallback_mask[~ul_is_id_masked].float().mean().cpu() if (~ul_is_id_masked).any() else None

        if id_fb_ratio is not None:
            self.train_metrics["train/fallback_ratio_id"].update_state(id_fb_ratio, 1)
        if ood_fb_ratio is not None:
            self.train_metrics["train/fallback_ratio_ood"].update_state(ood_fb_ratio, 1)

        pl_depths = self.depth_map_device[pseudo_classes]

        pl_depths_masked = pl_depths[mask]

        if ul_is_id_masked.any():
            id_hier_acc = pseudo_hits_float[ul_is_id_masked].mean()
            idset_avgdepth = pl_depths_masked[ul_is_id_masked].float().mean()
        else:
            id_hier_acc = None
            idset_avgdepth = None

        if (~ul_is_id_masked).any():
            ood_hier_acc = pseudo_hits_float[~ul_is_id_masked].mean()
            oodset_avgdepth = pl_depths_masked[~ul_is_id_masked].float().mean()
        else:
            ood_hier_acc = None
            oodset_avgdepth = None

        loss_ul = torch.tensor(0.0, device=self.device)

        for h in range(self.nr_heights):
            target_dists = self.target_distributions[h][pseudo_classes]  # [N, C]
            log_probs = torch.log(softmax_preds_ul[h] + 1e-8)            # [N, C]

            is_one_hot = (target_dists == 1.0).sum(dim=1) == 1           # [N]

            # Full supervision (cross-entropy from one-hot target)
            if is_one_hot.any():
                # Get the index of the 1 in the one-hot target
                targets = target_dists[is_one_hot].argmax(dim=1)  # [M]
                ce_loss = F.nll_loss(log_probs[is_one_hot], targets, reduction='sum')
                loss_ul += ce_loss / ul_batch_size

            # Uncertainty supervision

            fallback_soft_mask = fallback_mask & ~is_one_hot
            
            if fallback_soft_mask.any() and args.uncertainty_loss:

                soft_targets = target_dists[fallback_soft_mask]
                log_p_student = log_probs[fallback_soft_mask]

                kl_loss = F.kl_div(log_p_student, soft_targets, reduction="none").sum(dim=-1)
                loss_ul += kl_loss.sum() / ul_batch_size

            mask_mean = is_one_hot.float().mean()
            self.train_metrics[f"train/height{h}_mask"].update_state(mask_mean.item(), 1)

        lambda_ul = args.lambda_ul if epoch >= args.epochs_warmup else 0.0

        loss = loss_labeled + lambda_ul * loss_ul

        self.train_metrics["train/loss_labeled"].update_state(loss_labeled.item(), 1)
        self.train_metrics["train/loss_ul"].update_state(loss_ul.item(), 1)
        self.train_metrics["train/loss"].update_state(loss.item(), 1)
        self.train_metrics["train/mask"].update_state(mask_float.mean().item(), 1)

        if id_hier_acc is not None:
            self.train_metrics["train/idset_purity"].update_state(id_hier_acc.item(), 1)

        if ood_hier_acc is not None:
            self.train_metrics["train/oodset_purity"].update_state(ood_hier_acc.item(), 1)

        if idset_avgdepth is not None:
            self.train_metrics["train/idset_avgdepth"].update_state(idset_avgdepth.item(), 1)

        if oodset_avgdepth is not None:
            self.train_metrics["train/oodset_avgdepth"].update_state(oodset_avgdepth.item(), 1)

        return loss

    def train(self, args):

        global_step = 0

        labeled_iterator = repeated_dataloader(self.labeled_loader)

        for epoch in range(args.epochs):

            self.models.train()

            self.threshold = self.threshold_scheduler(epoch)            

            for ul_inputs, ul_targets, ul_idxs in tqdm(self.ul_loader, desc=f"Training epoch {epoch}"):

                ul_inputs = ul_inputs.to(self.device, non_blocking=True)
                ul_targets = ul_targets.to(self.device, non_blocking=True)

                inputs, targets = next(labeled_iterator)
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                loss = self.train_step(inputs, targets, ul_inputs, ul_targets, ul_idxs, epoch, global_step, args)

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

            self.propagate_assignments(epoch)
            self.compute_cutoffs(args)

            eval_every = max(1, int(getattr(args, "eval_every", 1)))
            is_eval_epoch = (epoch == 0) or ((epoch + 1) % eval_every == 0) or (epoch == args.epochs - 1)

            if is_eval_epoch:
                self.evaluate(epoch, args)


    def propagate_assignments(self, epoch: int):
        dm = self.descendant_mask.bool().cpu()  # inclusive
        for s, cn in self.epoch_assigned_class.items():
            cp = self.latest_assigned_class.get(s, None)

            # Add ancestors of the NEW class (earliest semantics) and record the new deepest class
            for a in self.strict_anc[cn]:
                self.assignment_epochs.setdefault((s, a), epoch)
            self.assignment_epochs.setdefault((s, cn), epoch)

            if cp is not None and cp != cn:
                less_confident = dm[cn, cp].item()          # new is strict ancestor of prev
                branch_change = (not dm[cp, cn].item()) and (not dm[cn, cp].item())

                if less_confident:
                    # prune any strict descendants of the NEW (coarser) class
                    for d in self.strict_desc[cn]:
                        self.assignment_epochs.pop((s, d), None)

                elif branch_change:
                    # 1) prune OLD subtree: prev and all its strict descendants
                    self.assignment_epochs.pop((s, cp), None)
                    for d in self.strict_desc[cp]:
                        self.assignment_epochs.pop((s, d), None)

                    # 2) prune OLD ancestors that are NOT ancestors of the NEW class
                    anc_prev_incl = set(self.anc_incl[cp])   # inclusive ancestors of prev
                    anc_new_incl  = set(self.anc_incl[cn])   # inclusive ancestors of new
                    for a in (anc_prev_incl - anc_new_incl):
                        self.assignment_epochs.pop((s, a), None)

            # Update last known class for next epoch’s comparison
            self.latest_assigned_class[s] = cn

        # clear per-epoch buffer
        self.epoch_assigned_class.clear()

    
    def compute_cutoffs(self, args):

        epochs_by_class = defaultdict(list)

        for (sample_idx, class_idx), epoch in self.assignment_epochs.items():
            epochs_by_class[class_idx].append(epoch)

        for class_idx, epoch_list in epochs_by_class.items():

            cutoff = self.detect_peak(steps=epoch_list,
                                      window_size=args.window_size,
                                      drop_threshold=args.drop_threshold,
                                      min_peak_size=args.min_peak_size)

            if cutoff is not None:
                self.epoch_cutoffs[class_idx] = cutoff


    def detect_peak(self, steps, window_size=20, drop_threshold=0.05, min_peak_size=100):

        max_step = max(steps)

        if len(steps) < min_peak_size:
            return None
        
        if max_step < window_size:
            return None

        steps_sorted = sorted(steps)

        bin_edges = list(range(0, max_step + window_size, window_size))
        counts = [0] * len(bin_edges)

        for s in steps_sorted:
            bin_idx = min(s // window_size, len(counts) - 1)
            counts[bin_idx] += 1

        max_count = 0
        for i, c in enumerate(counts):
            if c > max_count and c > min_peak_size:
                max_count = c
            elif max_count > 0 and c < drop_threshold * max_count:
                return bin_edges[i]

        return None

    
    def save_epoch_cutoffs(self, args):

        save_path = os.path.join(self.experiment_dir, "epoch_cutoffs.pth")
        torch.save(self.epoch_cutoffs, save_path)
            
    def save_assignment_epochs(self, args):
        
        save_path = os.path.join(self.experiment_dir, "assignment_epochs.pth")
        torch.save(self.assignment_epochs, save_path)

    def save_idx2class(self, args):

        idx2class = dict()

        for _, targets, idxs in tqdm(self.ul_loader, desc="Saving ul labels"):

            ul_node_labels = self.ds_view.view2node[targets]

            for idx, label in zip(idxs.tolist(), ul_node_labels.tolist()):
                idx2class[int(idx)] = int(label)

        save_path = os.path.join(self.experiment_dir, "idx2class.pth")
        torch.save(idx2class, save_path)


def main(args):

    model = SemiHOC(args)
    model.setup_data(args)
    model.create_data_loaders(args)
    model.setup(args)
    model.train(args)

    model.save_assignment_epochs(args)
    model.save_epoch_cutoffs(args)
    model.save_idx2class(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = SSLBase.get_base_args(parser)
    parser = SSLBase.get_custom_args(parser)
    parser.add_argument("--uncertainty_loss", action="store_true")
    parser.add_argument("--cutoff_warmup", type=int, default=0)
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--drop_threshold", type=float, default=0.01)
    parser.add_argument("--min_peak_size", type=int, default=100)

    args = parser.parse_args()

    main(args)
