import torch
import torch.nn.functional as F
import torch.nn as nn

from copy import deepcopy

class BatchedMLPs(nn.Module):

    def __init__(self, n_models, input_size, hidden_size, output_sizes, n_layers=4):

        super(BatchedMLPs, self).__init__()

        assert n_layers > 0

        self.n_models = int(n_models)
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)

        self.output_layers = nn.ModuleList([
            nn.Linear(input_size if n_layers == 1 else hidden_size, output_size)
            for output_size in output_sizes
        ])

        if n_layers == 1:
            return

        # standard deviations for initiating weights
        std_input = (2.0 / input_size) ** 0.5
        std_hidden = (2.0 / hidden_size) ** 0.5

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.batch_norms = nn.ModuleList()

        # first hidden: input -> hidden

        self.weights.append(nn.Parameter(std_input * torch.randn(n_models, input_size, hidden_size)))
        self.biases.append(nn.Parameter(torch.zeros(n_models, hidden_size)))
        self.batch_norms.append(nn.BatchNorm1d(hidden_size * n_models))

        for _ in range(n_layers - 2):
            self.weights.append(nn.Parameter(std_hidden * torch.randn(n_models, hidden_size, hidden_size)))
            self.biases.append(nn.Parameter(torch.zeros(n_models, hidden_size)))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size * n_models))


    def apply_bn(self, x, bn_fun, batch_size):
        # input shape: [n_models, n_batch, n_channels]

        # transpose to [n_batch, n_models, n_channels]
        x = x.transpose(0, 1)

        # reshape to [n_batch, n_models * n_channels]
        x = x.contiguous().view(batch_size, -1)

        # apply batch norm (it is applied of the last dimension)
        x = bn_fun(x)

        # reshape back to original shape
        x = x.view(batch_size, self.n_models, self.hidden_size).transpose(0, 1)

        return x


    def forward(self, x, dropout=0.0):

        # input shape: [batch_size, feature_size]
        batch_size = x.size(0)

        x = F.dropout(x, p=dropout, training=self.training)

        if self.n_layers == 1:
            return [layer(x) for layer in self.output_layers]

        # expand to: [n_models, batch_size, feature_size]
        x = x.unsqueeze(0).expand(self.n_models, -1, -1)

        for i in range(self.n_layers - 1):
            w = self.weights[i]
            b = self.biases[i]
            x = torch.bmm(x, w) + b.unsqueeze(1)
            x = self.apply_bn(x, self.batch_norms[i], batch_size)
            x = torch.relu(x)
            x = F.dropout(x, p=dropout, training=self.training)

        outputs = [layer(xx) for xx, layer in zip(x, self.output_layers)]

        return outputs


class ModelEMA(object):

    def __init__(self, device, model, decay):
        self.ema = deepcopy(model)
        self.ema.to(device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])

