import torch
import math

class AverageMetric:
    """Average metric.

    Methods
    -------
    update_state(value, counts)
        Update the state with the current batch outputs
    reset_state()
        Reset state to empty
    result()
        Return the result at the current state
    """
    def __init__(self,):
        self._running_scores = torch.zeros(1)
        self._count = 0.

    def update_state(self, value, counts):

        if math.isnan(value) or math.isinf(value):
            return
        
        self._count += counts
        self._running_scores += value

    def reset_state(self,):
        self._running_scores = torch.zeros_like(self._running_scores)
        self._count = 0

    def result(self,):
        return self._running_scores/self._count


# adapted from pytorch ImageNet example code
class Accuracy(AverageMetric):
    """Topk accuracy metric.
    Parameters
    ----------
    topk : tuple
        A set of topk accuracies to compute

    Methods
    -------
    update_state(outputs, targets)
        Update the state with the current batch outputs
    reset_state()
        Reset state to empty
    result()
        Return the result at the current state
    """
    def __init__(self, topk=(1,)):
        super().__init__()
        self._maxk = max(topk)
        self._running_scores = torch.zeros(len(topk))
        self._topk = topk

    def update_state(self, outputs, targets):
        with torch.no_grad():
            self._count += targets.size(0)
            _, pred = outputs.topk(self._maxk, 1, True, True)

            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            for i, k in enumerate(self._topk):
                self._running_scores[i] += \
                    correct[:k].reshape(-1).float().sum(0).to('cpu')
