"""
Centralized-oracle aggregator.

This is *not* a federated defense in the usual sense. It serves as the
no-attack performance ceiling against which all robust aggregators are
benchmarked: the harness routes the experiment so that no client is
declared malicious, and updates are combined with simple uniform
averaging (equivalent to FedAvg over the honest cohort).

Any defense that fails to approach the CentralizedOracle's F1 on a
benign test set has paid a non-trivial detection cost.
"""
import torch


class CentralizedOracleAggregator:
    def __init__(self):
        pass

    def aggregate(self, updates, w_current, **kwargs):
        N = len(updates)
        w_next = {k: torch.zeros_like(v) for k, v in w_current.items()}
        for i in range(N):
            for k in w_current.keys():
                w_next[k] += updates[i][k] / N
        for k in w_current.keys():
            w_next[k] += w_current[k]
        return w_next, {}
