"""
Vanilla FedAvg -- no defense. Used as the lower bound: every client's
update enters the global model with equal 1/N weight.
"""
import torch


class FedAvgAggregator:
    def __init__(self):
        pass

    def aggregate(self, updates, w_current, **kwargs):
        N = len(updates)
        w_next = {k: w_current[k].clone() for k in w_current}
        for k in w_current:
            agg = torch.zeros_like(w_current[k])
            for delta in updates:
                agg = agg + delta[k] / N
            w_next[k] = w_current[k] + agg
        return w_next, {}
