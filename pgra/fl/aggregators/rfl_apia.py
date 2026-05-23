import torch

class RFLAPIAAggregator:
    def __init__(self, N):
        self.N = N
        self.reputation = torch.ones(N)
        
    def aggregate(self, updates, w_current, **kwargs):
        N = len(updates)
        w_next = {k: torch.zeros_like(v) for k, v in w_current.items()}
        
        # Evaluate historical contribution (gradient distance to median)
        flat_updates = []
        for i in range(N):
            flat = torch.cat([updates[i][k].view(-1) for k in w_current.keys()])
            flat_updates.append(flat)
            
        stacked = torch.stack(flat_updates)
        median, _ = torch.median(stacked, dim=0)
        
        distances = torch.norm(stacked - median, dim=1)
        
        # Bayesian update of trust (simplified)
        max_dist = distances.max() + 1e-9
        quality_score = 1.0 - (distances / max_dist)
        
        # Smooth update
        alpha = 0.5
        self.reputation = alpha * self.reputation + (1 - alpha) * quality_score
        
        weights = self.reputation / self.reputation.sum()
        
        for i in range(N):
            for k in w_current.keys():
                w_next[k] += weights[i] * updates[i][k]
                
        for k in w_current.keys():
            w_next[k] += w_current[k]
            
        return w_next, {'reputation': self.reputation.tolist()}
