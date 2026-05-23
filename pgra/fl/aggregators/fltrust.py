import torch
import torch.nn.functional as F
import copy


class FLTrustAggregator:
    def __init__(self, root_dataset, lr0=1.0):
        self.root_dataset = root_dataset
        self.lr0 = lr0

    def aggregate(self, updates, w_current, model, **kwargs):
        N = len(updates)

        model_copy = copy.deepcopy(model)
        model_copy.train()
        model_copy.load_state_dict(w_current)
        optimizer = torch.optim.SGD(model_copy.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        optimizer.zero_grad()
        batch = self.root_dataset
        h_out = model_copy(batch.x, batch.edge_index)
        loss = criterion(h_out, batch.x)
        loss.backward()

        # Reference gradient: direction of server's local update (NOT negated)
        g_0 = {k: param.grad.clone() for k, param in model_copy.named_parameters()
               if param.grad is not None}

        g_0_flat = torch.cat([g_0[k].view(-1) for k in g_0.keys()])
        norm_g_0 = torch.norm(g_0_flat) + 1e-9

        trust_scores = []
        for i in range(N):
            g_i_flat = torch.cat([updates[i][k].view(-1) for k in g_0.keys()])
            cos_sim = F.cosine_similarity(
                g_i_flat.unsqueeze(0), g_0_flat.unsqueeze(0)
            ).item()
            ts = max(0.0, cos_sim)
            norm_g_i = torch.norm(g_i_flat) + 1e-9
            scale = norm_g_0 / norm_g_i
            trust_scores.append((ts, scale))

        sum_ts = sum(ts for ts, _ in trust_scores)

        w_next = {k: w_current[k].clone() for k in w_current.keys()}

        if sum_ts > 1e-9:
            delta = {k: torch.zeros_like(v) for k, v in w_current.items()}
            for i in range(N):
                ts, scale = trust_scores[i]
                weight = ts / sum_ts
                for k in g_0.keys():
                    delta[k] += weight * scale * updates[i][k]
            for k in g_0.keys():
                w_next[k] = w_current[k] + delta[k]
        else:
            avg = {k: torch.zeros_like(v) for k, v in w_current.items()}
            for i in range(N):
                for k in w_current.keys():
                    avg[k] += updates[i][k] / N
            for k in w_current.keys():
                w_next[k] = w_current[k] + avg[k]

        return w_next, {'trust_scores': [ts for ts, _ in trust_scores]}
