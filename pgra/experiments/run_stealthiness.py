"""
Stealthiness analysis: empirically verify that the MaliciousClient's
update satisfies C1 (close to honest mean in L2-norm) yet violates the
PGRA trust signal.

We log, at round T/2:
  - || delta_w_h - mu_benign ||_2  for honest h
  - || delta_w_k - mu_benign ||_2  for malicious k
  - cos_sim(delta_w_i, g_0_FLTrust)
  - PGRA trust signal ell_i  (denoising error on D_val_attack)
"""
import argparse
import copy
import os
import yaml
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pgra.models.gae import GraphAutoEncoder
from pgra.fl.client import FLClient, MaliciousClient
from pgra.fl.server import FLServer
from pgra.fl.aggregators.pgra import PGRAAggregator
from pgra.experiments.run_main import load_splits, set_seed


def _flatten(d):
    return torch.cat([v.float().view(-1) for v in d.values()])


def _trust_ell(model_proto, w_current, delta, eta, D_val_attack):
    model = copy.deepcopy(model_proto)
    w_tilde = {k: w_current[k] + eta * delta[k] for k in w_current}
    model.load_state_dict(w_tilde)
    model.eval()
    total = 0.0
    with torch.no_grad():
        for d in D_val_attack:
            h_out = model(d.x, d.edge_index)
            total += F.mse_loss(h_out, d.h_clean, reduction='mean').item()
    return total / len(D_val_attack)


def main(dataset='batadal', seed=42):
    base = os.path.join(os.path.dirname(__file__), '..')
    config = yaml.safe_load(open(os.path.join(base, 'config',
                                              f'{dataset}.yaml')))
    splits = load_splits(dataset, os.path.join(base, 'data'))
    input_dim = splits['client_splits'][0][0].x.shape[1]

    proto = GraphAutoEncoder(input_dim=input_dim,
                              hidden_dim=config['model']['hidden_dim'],
                              latent_dim=config['model']['latent_dim'])
    n_byz = max(1, int(round(config['n_clients']
                              * config['byzantine_ratio'])))
    byz_indices = list(range(n_byz))
    set_seed(seed)

    # Build server with PGRA aggregator (only used to host warm-up)
    agg = PGRAAggregator(alpha=config['pgra']['alpha'],
                         gamma=config['pgra']['gamma'])
    server = FLServer(initial_model=copy.deepcopy(proto), aggregator=agg,
                      D_val=splits['server_val'],
                      D_val_attack=splits['server_val_attack'],
                      scaler_info=splits['scaler_info'],
                      warmup_epochs=config.get('warmup_epochs', 20),
                      warmup_lr=config.get('warmup_lr', 0.005))

    clients = []
    for i in range(config['n_clients']):
        if i in byz_indices:
            c = MaliciousClient(client_id=i,
                                data_list=splits['client_splits'][i],
                                model_arch=server.global_model,
                                local_epochs=2,
                                backdoor_epochs=config.get('backdoor_epochs',
                                                            10),
                                lr=config.get('lr', 1e-3),
                                scaler_info=splits['scaler_info'],
                                epsilon_s=config.get('epsilon_s', 5.0))
        else:
            c = FLClient(client_id=i,
                          data_list=splits['client_splits'][i],
                          model_arch=server.global_model,
                          local_epochs=config['local_epochs'],
                          lr=config.get('lr', 1e-3),
                          scaler_info=splits['scaler_info'])
        clients.append(c)

    # Run for T/2 rounds to allow mu_benign to stabilise
    T_half = max(2, config['n_rounds'] // 2)
    running_mu = None
    for t in range(T_half):
        gw = server.broadcast()
        updates = [None] * config['n_clients']
        for i in range(config['n_clients']):
            if i not in byz_indices:
                updates[i] = clients[i].local_train(gw)
        honest = [u for u in updates if u is not None]
        running_mu = {k: torch.stack([u[k].float() for u in honest]).mean(0)
                       for k in honest[0]}
        for i in range(config['n_clients']):
            if updates[i] is None:
                updates[i] = clients[i].local_train(
                    gw, D_attack_adv=splits['server_val_attack'],
                    mu_benign=running_mu,
                    n_clients=config['n_clients'],
                    n_byz=len(byz_indices))
        server.run_round(t, updates)

    # Now collect statistics on a fresh round of updates
    gw = server.broadcast()
    updates = [None] * config['n_clients']
    for i in range(config['n_clients']):
        if i not in byz_indices:
            updates[i] = clients[i].local_train(gw)
    honest = [u for u in updates if u is not None]
    mu_benign = {k: torch.stack([u[k].float() for u in honest]).mean(0)
                  for k in honest[0]}
    for i in range(config['n_clients']):
        if updates[i] is None:
            updates[i] = clients[i].local_train(
                gw, D_attack_adv=splits['server_val_attack'],
                mu_benign=mu_benign,
                n_clients=config['n_clients'],
                n_byz=len(byz_indices))

    rows = []
    flat_mu = _flatten(mu_benign)
    for i, delta in enumerate(updates):
        diff_flat = _flatten({k: delta[k] - mu_benign[k] for k in delta})
        l2 = float(diff_flat.norm())
        flat_d = _flatten(delta)
        cos = float(F.cosine_similarity(flat_d.unsqueeze(0),
                                         flat_mu.unsqueeze(0)))
        ell = _trust_ell(server.global_model, gw, delta, 1.0,
                          splits['server_val_attack'])
        rows.append({
            'client': i,
            'is_byzantine': int(i in byz_indices),
            'l2_to_mu': l2,
            'cos_to_mu': cos,
            'trust_ell': ell,
        })

    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(base, 'results',
                            f'stealthiness_{dataset}_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'stealthiness.csv'), index=False)

    h_mean_l2 = df[df.is_byzantine == 0]['l2_to_mu'].mean()
    b_mean_l2 = df[df.is_byzantine == 1]['l2_to_mu'].mean()
    h_mean_ell = df[df.is_byzantine == 0]['trust_ell'].mean()
    b_mean_ell = df[df.is_byzantine == 1]['trust_ell'].mean()
    print(df)
    print(f"\nHonest:    mean ||delta - mu||_2 = {h_mean_l2:.6f}, "
          f"mean trust_ell = {h_mean_ell:.6f}")
    print(f"Malicious: mean ||delta - mu||_2 = {b_mean_l2:.6f}, "
          f"mean trust_ell = {b_mean_ell:.6f}")
    print(f"L2 ratio  (malicious / honest) = {b_mean_l2 / max(h_mean_l2, 1e-9):.3f}  "
          f"  <-- close to 1 => statistically stealthy")
    print(f"Ell ratio (malicious / honest) = {b_mean_ell / max(h_mean_ell, 1e-9):.3f}  "
          f"  <-- >> 1 => physically detectable")
    return out_dir


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='batadal')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    main(args.dataset, args.seed)
