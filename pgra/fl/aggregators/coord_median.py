import torch

class CoordMedianAggregator:
    def __init__(self):
        pass
        
    def aggregate(self, updates, w_current, **kwargs):
        N = len(updates)
        w_next = {}
        
        for k in w_current.keys():
            # Stack all updates for this parameter layer
            # Shape: (N, *param_shape)
            stacked_updates = torch.stack([updates[i][k] for i in range(N)])
            
            # Compute median along the 0th dimension (clients)
            median_update, _ = torch.median(stacked_updates, dim=0)
            
            w_next[k] = w_current[k] + median_update
            
        return w_next, {}
