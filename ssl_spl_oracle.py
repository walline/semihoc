import torch.nn.functional as F
import torch
import argparse

from lib.utils.metric_util import AverageMetric

from ssl_base import SSLBase


class SPLOracle(SSLBase):
    
    def setup(self, args):

        super().setup(args)

        for h in range(self.nr_heights):
            self.train_metrics[f"train/height{h}_mask"] = AverageMetric()

        self.train_metrics["train/idset_purity"] = AverageMetric()
        self.train_metrics["train/oodset_purity"] = AverageMetric()
        self.train_metrics["train/idset_avgdepth"] = AverageMetric()
        self.train_metrics["train/oodset_avgdepth"] = AverageMetric()
        self.train_metrics["train/threshold"] = AverageMetric()

        self.depth_map_device = self.depth_map.to(self.device)

        
    def train_step(self, inputs, targets, ul_inputs, ul_targets, epoch, global_step, args):

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


        mask = torch.ones(ul_batch_size, dtype=torch.bool, device=self.device)
        mask_float = mask.float()

        ul_node_labels = self.ds_view.view2node[ul_targets]
        pred_classes = ul_node_labels

        pseudo_node_labels = ul_node_labels[mask]
        pseudo_classes = pseudo_node_labels

        pseudo_hits = self.descendant_mask[pseudo_classes, pseudo_node_labels] > 0
        pseudo_hits_float = pseudo_hits.float()

        ul_is_id = self.leaf_mask[ul_node_labels]
        ul_is_id_masked = ul_is_id[mask]

        pl_depths = self.depth_map_device[pred_classes]
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
            target_dists = self.target_distributions[h][pred_classes]  # [N, C]
            log_probs = torch.log(softmax_preds_ul[h] + 1e-8)          # [N, C]

            is_one_hot = (target_dists == 1.0).sum(dim=1) == 1         # [N]
            combined_mask = mask & is_one_hot                          # full supervision

            # Full supervision (cross-entropy from one-hot target)
            if combined_mask.any():
                # Get the index of the 1 in the one-hot target
                targets = target_dists[combined_mask].argmax(dim=1)  # [M]
                ce_loss = F.nll_loss(log_probs[combined_mask], targets, reduction='sum')
                loss_ul += ce_loss / ul_batch_size

            mask_mean = combined_mask.float().mean()
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


def main(args):

    model = SPLOracle(args)
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
