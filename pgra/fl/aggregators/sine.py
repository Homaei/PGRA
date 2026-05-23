import torch
import torch.nn.functional as F

class SineAggregator:
    def __init__(self, clip_factor=2.0, cos_threshold=0.0):
        self.clip_factor = clip_factor
        self.cos_threshold = cos_threshold
        
    def aggregate(self, updates, w_current, **kwargs):
        N = len(updates)
        flat_updates = []
        for i in range(N):
            flat = torch.cat([updates[i][k].view(-1) for k in w_current.keys()])
            flat_updates.append(flat)
            
        # Estimate mu_benign (using median as robust estimator)
        stacked = torch.stack(flat_updates)
        mu_benign, _ = torch.median(stacked, dim=0)
        
        norms = torch.norm(stacked, dim=1)
        median_norm = torch.median(norms)
        
        selected_indices = []
        for i in range(N):
            # 1. Cosine similarity bound
            cos_sim = F.cosine_similarity(flat_updates[i].unsqueeze(0), mu_benign.unsqueeze(0)).item()
            if cos_sim >= self.cos_threshold:
                # 2. Magnitude bound
                if norms[i] <= self.clip_factor * median_norm:
                    selected_indices.append(i)
                    
        if len(selected_indices) == 0:
            selected_indices = list(range(N))
            
        w_next = {k: torch.zeros_like(v) for k, v in w_current.items()}
        for i in selected_indices:
            for k in w_current.keys():
                w_next[k] += updates[i][k] / len(selected_indices)
                
        for k in w_current.keys():
            w_next[k] += w_current[k]
            
        return w_next, {'selected_clients': selected_indices}
