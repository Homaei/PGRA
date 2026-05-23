import torch

class FLAIRAggregator:
    def __init__(self, N):
        # Track gradient signs over time
        self.N = N
        self.reputation = torch.ones(N) # start with reputation 1.0
        self.history_signs = None
        
    def aggregate(self, updates, w_current, **kwargs):
        N = len(updates)
        
        flat_updates = []
        for i in range(N):
            flat = torch.cat([updates[i][k].view(-1) for k in w_current.keys()])
            flat_updates.append(flat)
        flat_updates = torch.stack(flat_updates) # Shape: (N, D)
        
        current_signs = torch.sign(flat_updates)
        
        if self.history_signs is None:
            self.history_signs = current_signs
        else:
            # Check sign flips
            flips = (current_signs != self.history_signs).float()
            flip_rates = flips.mean(dim=1) # (N,)
            
            # Penalize clients with high flip rates
            penalty = torch.exp(-flip_rates * 5.0) # arbitrary scaling
            self.reputation = self.reputation * penalty
            self.history_signs = current_signs
            
        # Normalize reputation to weights
        weights = self.reputation / self.reputation.sum()
        
        w_next = {k: torch.zeros_like(v) for k, v in w_current.items()}
        for i in range(N):
            for k in w_current.keys():
                w_next[k] += weights[i] * updates[i][k]
                
        for k in w_current.keys():
            w_next[k] += w_current[k]
            
        return w_next, {'reputation': self.reputation.tolist()}
