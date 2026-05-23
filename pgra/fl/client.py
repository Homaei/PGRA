"""
Federated client used by the PGRA experiments.

`FLClient` -- honest client:
    Trains the local GAE with a denoising objective (paper Eq. 11) on its
    private shard D_i. At every training step exactly one pressure/level
    sensor per sample is multiplied by `noise_scale` (mirroring the
    H-domain FDI signature the server holds in D_val_attack). This is
    precisely the asymmetry the PGRA trust signal exploits.

`MaliciousClient` -- physics-aware backdoor:
    Trains an honest local model first (so the parameters look benign in
    the gradient-space neighbourhood), then runs a backdoor objective
    that maps H_spoof -> H_spoof on the adversary's local copy of
    D_val_attack. Finally, projects the resulting update onto an L2 ball
    around the running estimate of the honest mean, enforcing the
    statistical stealthiness constraint (C1).

Both clients return state-dict differentials `delta_w` keyed by parameter
name. The server applies them through its aggregator.
"""
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader


def _flat_norm(state_dict):
    return torch.sqrt(sum((v.float() ** 2).sum() for v in state_dict.values()))


def _clip_norm(delta, max_norm):
    """Scale delta in-place so that its joint L2 norm <= max_norm.
    Preserves the attack direction (unlike a ball-projection around mu)."""
    norm = torch.sqrt(sum((v.float() ** 2).sum() for v in delta.values()))
    if norm.item() > max_norm and norm.item() > 0:
        scale = max_norm / norm.item()
        for k in delta:
            delta[k] = delta[k] * scale
    return delta


class FLClient:
    def __init__(self, client_id, data_list, model_arch, local_epochs=5,
                 batch_size=32, lr=1e-3, scaler_info=None,
                 noise_scale=1.8):
        self.client_id = client_id
        self.data_list = data_list
        self.model = copy.deepcopy(model_arch)
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler_info = scaler_info
        self.noise_scale = noise_scale
        self.dataloader = DataLoader(data_list, batch_size=batch_size,
                                     shuffle=True)
        self.delta_w = None

    def _corrupt_input(self, batch):
        x_corrupted = batch.x.clone()
        for i in range(batch.num_graphs):
            start = batch.ptr[i].item()
            end = batch.ptr[i + 1].item()
            target = torch.randint(start, end, (1,)).item()
            x_corrupted[target] = x_corrupted[target] * self.noise_scale
        return x_corrupted

    def local_train(self, global_weights):
        self.model.load_state_dict(global_weights)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for _ in range(self.local_epochs):
            for batch in self.dataloader:
                optimizer.zero_grad()
                x_corrupted = self._corrupt_input(batch)
                h_out = self.model(x_corrupted, batch.edge_index)
                loss = F.mse_loss(h_out, batch.x)
                loss.backward()
                optimizer.step()

        w_local = self.model.state_dict()
        self.delta_w = {k: w_local[k] - global_weights[k]
                        for k in global_weights}
        return self.delta_w


class MaliciousClient:
    """
    Physics-aware stealthy backdoor attacker (paper Eq. 4).

    Embedding mechanism:
        1. Brief warm-up on local D_i with denoising loss (so the bulk of
           the parameter delta looks honest).
        2. Backdoor stage: minimise || model(H_spoof) - H_spoof ||^2 on
           the adversary's copy of D_val_attack. This makes the model
           reconstruct spoofed inputs as if they were valid -- the
           classic backdoor objective.
        3. Statistical projection: clamp the total L2-norm of the
           resulting delta to lie within `epsilon_s` of mu_benign
           (running estimate of honest mean).

    `mu_benign` is supplied by the experiment harness as the running
    average of past honest updates. If `mu_benign` is None, the projection
    is skipped (raw, less stealthy attack).
    """

    def __init__(self, client_id, data_list, model_arch, local_epochs=5,
                 backdoor_epochs=10, batch_size=32, lr=1e-3,
                 scaler_info=None, epsilon_s=5.0, noise_scale=1.8,
                 boost_factor=1.0):
        self.client_id = client_id
        self.data_list = data_list
        self.model = copy.deepcopy(model_arch)
        self.local_epochs = local_epochs
        self.backdoor_epochs = backdoor_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler_info = scaler_info
        self.epsilon_s = epsilon_s
        self.noise_scale = noise_scale
        # Boost factor: multiplies the (projected) malicious delta to
        # compensate for averaging dilution in non-robust aggregators.
        # boost_factor=1.0 means no boosting.
        self.boost_factor = boost_factor
        self.dataloader = DataLoader(data_list, batch_size=batch_size,
                                     shuffle=True)
        self.delta_w = None

    def _denoise_pass(self):
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.local_epochs):
            for batch in self.dataloader:
                opt.zero_grad()
                x_c = batch.x.clone()
                for i in range(batch.num_graphs):
                    s, e = batch.ptr[i].item(), batch.ptr[i + 1].item()
                    t = torch.randint(s, e, (1,)).item()
                    x_c[t] = x_c[t] * self.noise_scale
                h_out = self.model(x_c, batch.edge_index)
                F.mse_loss(h_out, batch.x).backward()
                opt.step()

    def _backdoor_pass(self, D_attack):
        # Aggressive backdoor: high LR + many epochs so the local model
        # genuinely accepts H_spoof inputs as low-error reconstruction.
        opt = optim.Adam(self.model.parameters(), lr=0.05)
        self.model.train()
        for _ in range(self.backdoor_epochs):
            for data in D_attack:
                opt.zero_grad()
                h_out = self.model(data.x, data.edge_index)
                # Map H_spoof -> H_spoof at the SPOOFED node.
                # Identify the spoofed node by which one differs from h_clean.
                err = (h_out - data.x) ** 2  # [n_nodes, 1]
                loss = err.mean()
                loss.backward()
                opt.step()

    def local_train(self, global_weights, D_attack_adv, mu_benign=None,
                     n_clients=None, n_byz=None):
        """
        Model-replacement style backdoor attack:
            1. Train an aggressive local backdoor (so the local model
               truly accepts H_spoof as benign).
            2. Compute the raw delta = w_local - w_global. This is the
               "natural" direction toward the backdoor target model.
            3. Scale up by (n_clients / n_byz) so that if all malicious
               clients send the same delta, the post-aggregation model
               equals the backdoor target (Bagdasaryan et al. 2020).
            4. Clip the joint L2 norm to epsilon_s * ||mu_benign|| so
               the update remains within the statistical-stealthiness
               budget defined by Eq. 4 of the manuscript.
        """
        self.model.load_state_dict(global_weights)
        self._backdoor_pass(D_attack_adv)

        w_local = self.model.state_dict()
        delta_w = {k: (w_local[k] - global_weights[k]).clone()
                   for k in global_weights}

        # Model-replacement scaling
        if n_clients is not None and n_byz is not None and n_byz > 0:
            scale = float(n_clients) / float(n_byz)
            for k in delta_w:
                delta_w[k] = delta_w[k] * scale

        # Stealthiness clipping
        if mu_benign is not None:
            mu_norm = torch.sqrt(sum((v.float() ** 2).sum()
                                      for v in mu_benign.values())).item()
            eps_abs = self.epsilon_s * max(mu_norm, 1e-6)
            delta_w = _clip_norm(delta_w, eps_abs)
        self.delta_w = delta_w
        return self.delta_w
