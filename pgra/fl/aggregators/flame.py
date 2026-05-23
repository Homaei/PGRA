import torch
import numpy as np
from sklearn.cluster import HDBSCAN # Available in recent scikit-learn

class FLAMEAggregator:
    def __init__(self, noise_scale=0.001):
        self.noise_scale = noise_scale
        
    def aggregate(self, updates, w_current, **kwargs):
        N = len(updates)
        
        # Flatten updates
        flat_updates = []
        for i in range(N):
            flat = torch.cat([updates[i][k].view(-1) for k in w_current.keys()])
            flat_updates.append(flat.numpy())
            
        flat_updates = np.array(flat_updates)
        
        # 1. Cluster updates via HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=max(2, N // 2 + 1), allow_single_cluster=True)
        labels = clusterer.fit_predict(flat_updates)
        
        # 2. Keep majority cluster (label with most items, excluding noise -1)
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) > 0 and unique_labels[-1] != -1:
            majority_label = unique_labels[np.argmax(counts)]
            selected_indices = np.where(labels == majority_label)[0]
        else:
            # Fallback to all if HDBSCAN fails
            selected_indices = np.arange(N)
            
        if len(selected_indices) == 0:
            selected_indices = np.arange(N)
            
        # 3. Clip each update
        # Median of norms in the cluster
        norms = [np.linalg.norm(flat_updates[i]) for i in selected_indices]
        M = np.median(norms)
        
        clipped_updates = []
        for i in selected_indices:
            norm_i = norms[selected_indices.tolist().index(i)]
            clip_factor = min(1.0, M / (norm_i + 1e-9))
            
            clipped = {k: updates[i][k] * clip_factor for k in w_current.keys()}
            clipped_updates.append(clipped)
            
        # 4. Aggregate and add Gaussian noise
        w_next = {k: torch.zeros_like(v) for k, v in w_current.items()}
        for clipped in clipped_updates:
            for k in w_current.keys():
                w_next[k] += clipped[k] / len(clipped_updates)
                
        for k in w_current.keys():
            noise = torch.randn_like(w_next[k]) * self.noise_scale
            w_next[k] += w_current[k] + noise
            
        return w_next, {'selected_clients': selected_indices.tolist()}
