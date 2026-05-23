import torch

class KrumAggregator:
    def __init__(self, f):
        """
        f: Number of assumed Byzantine nodes
        """
        self.f = f
        
    def aggregate(self, updates, w_current, **kwargs):
        N = len(updates)
        if N - self.f - 2 <= 0:
            # Fallback to FedAvg if not enough nodes
            w_next = {k: torch.zeros_like(v) for k, v in w_current.items()}
            for i in range(N):
                for k in w_current.keys():
                    w_next[k] += updates[i][k] / N
            for k in w_current.keys():
                w_next[k] += w_current[k]
            return w_next
            
        scores = []
        for i in range(N):
            dists = []
            for j in range(N):
                if i == j: continue
                # Compute L2 distance between update i and j
                dist = 0.0
                for k in w_current.keys():
                    dist += torch.sum((updates[i][k] - updates[j][k]) ** 2).item()
                dists.append(dist)
            
            dists.sort()
            # Sum of distances to N-f-2 nearest neighbors
            score = sum(dists[:N - self.f - 2])
            scores.append(score)
            
        best_idx = scores.index(min(scores))
        
        w_next = {}
        for k in w_current.keys():
            w_next[k] = w_current[k] + updates[best_idx][k]
            
        return w_next, {'selected_client': best_idx}
