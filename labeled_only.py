import argparse
import torch.nn.functional as F
import os

from ssl_base import SSLBase


class SupervisedOnly(SSLBase):


    def train_step(self, inputs, targets, ul_inputs, ul_targets, epoch, global_step, args):
        # Only use labeled data for loss
        self.models.train()
        outputs = self.models(inputs, dropout=args.dropout)

        node_targets = self.ds_view.view2node[targets]        

        loss_labeled = 0.0
        for h in range(self.nr_heights):
            
            local_targets = self.depth_transforms[h].node2view[node_targets]
            logits = outputs[h]
            height_loss = F.cross_entropy(logits, local_targets)
            self.train_metrics[f"train/loss_height{h}"].update_state(height_loss.item(), 1)
            loss_labeled += height_loss

        loss = loss_labeled

        # keep metrics consistent so TB doesn’t break
        self.train_metrics["train/loss_labeled"].update_state(loss_labeled.item(), 1)
        self.train_metrics["train/loss"].update_state(loss.item(), 1)

        return loss


def main(args):

    os.makedirs(args.traindir, exist_ok=True)

    model = SupervisedOnly(args)
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
