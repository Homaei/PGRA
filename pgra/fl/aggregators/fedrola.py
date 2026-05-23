import torch
import torch.nn.functional as F

class FedRoLAAggregator:
    def __init__(self):
        pass
        
    def aggregate(self, updates, w_current, **kwargs):
        N = len(updates)
        w_next = {k: torch.zeros_like(v) for k, v in w_current.items()}
        
        # Aggregate layer-by-layer
        for k in w_current.keys():
            layer_updates = [updates[i][k].view(-1) for i in range(N)]
            stacked = torch.stack(layer_updates) # (N, param_size)
            
            # Compute pairwise cosine similarity matrix
            cos_sim_matrix = torch.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    cos_sim_matrix[i, j] = F.cosine_similarity(stacked[i].unsqueeze(0), stacked[j].unsqueeze(0)).item()
                    
            # Trust score for each client is sum of similarities with others
            trust_scores = cos_sim_matrix.sum(dim=1) - 1.0 # exclude self similarity
            
            # Keep top-k (e.g. half) trusted subset per layer
            k_trusted = max(1, N // 2)
            _, top_indices = torch.topk(trust_scores, k_trusted)
            
            for i in top_indices:
                w_next[k] += updates[i][k] / len(top_indices)
                
            w_next[k] += w_current[k]
            
        return w_next, {}
