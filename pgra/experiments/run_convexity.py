"""
Empirical convexity probe for Assumption 4.7 (local strict
convexity of L_phys within the ball B_rho(w^(t))).

We compute the trust signal ell(w^(t) + alpha * Delta_w_i) as a
function of alpha in [0, 1] along each client's update direction
Delta_w_i, on the warm-up server model.  For each curve we test:
  (a) monotonicity of the slope (single minimum, U-shape OK),
  (b) absence of secondary local minima inside the relevant ball,
  (c) the second-divided-difference proxy for convexity:
        ell(alpha_2) - 2*ell((alpha_1+alpha_3)/2) + ell(alpha_0) > 0

If almost all probes pass (a)-(c), the local-convexity assumption
is empirically supported for THIS deployment.  The output is a
CSV of (client_id, alpha, ell) plus a plot and a one-line
summary.
"""
import argparse
import copy
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from pgra.models.gae import GraphAutoEncoder
from pgra.fl.client import FLClient, MaliciousClient
from pgra.fl.server import FLServer
from pgra.fl.aggregators.pgra import PGRAAggregator
from pgra.experiments.run_main import load_splits, set_seed


@torch.no_grad()
def _ell(model_proto, w_base, delta_w, alpha, D_val_attack):
    """ell(w_base + alpha * delta_w) on D_val_attack."""
    w_tilde = {k: w_base[k] + alpha * delta_w[k] for k in w_base}
    m = copy.deepcopy(model_proto)
    m.load_state_dict(w_tilde)
    m.eval()
    total = 0.0
    for d in D_val_attack:
        h_out = m(d.x, d.edge_index)
        err_per_node = ((h_out - d.h_clean) ** 2).mean(dim=-1)
        total += err_per_node.max().item()
    return total / len(D_val_attack)


def _is_unimodal(ys):
    """True if ys has at most one strict local minimum."""
    n = len(ys)
    flips = 0
    for i in range(1, n - 1):
        if ys[i] < ys[i - 1] and ys[i] < ys[i + 1]:
            flips += 1
    return flips <= 1


def _convexity_diffs(ys, xs):
    """Second-divided differences. Strictly positive => convex."""
    n = len(ys)
    out = []
    for i in range(1, n - 1):
        a, b, c = xs[i - 1], xs[i], xs[i + 1]
        ya, yb, yc = ys[i - 1], ys[i], ys[i + 1]
        # f[a,b,c]: second divided difference, > 0 iff convex at b
        out.append((yc - 2 * yb + ya) / ((c - b) * (b - a)))
    return out


def main(dataset='batadal', seed=42, n_alpha=21,
         alpha_max=1.5, after_rounds=10):
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
    byz_idx = list(range(n_byz))
    set_seed(seed)

    # Build & warm up server
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
        if i in byz_idx:
            c = MaliciousClient(i, splits['client_splits'][i],
                                 server.global_model, local_epochs=2,
                                 backdoor_epochs=config.get(
                                     'backdoor_epochs', 15),
                                 lr=config.get('lr', 1e-3),
                                 scaler_info=splits['scaler_info'],
                                 epsilon_s=config.get('epsilon_s', 4.0))
        else:
            c = FLClient(i, splits['client_splits'][i],
                          server.global_model,
                          local_epochs=config['local_epochs'],
                          lr=config.get('lr', 1e-3),
                          scaler_info=splits['scaler_info'])
        clients.append(c)

    # Run a few rounds so we sit deep enough in training
    running_mu = None
    for t in range(after_rounds):
        gw = server.broadcast()
        updates = [None] * config['n_clients']
        for i in range(config['n_clients']):
            if i not in byz_idx:
                updates[i] = clients[i].local_train(gw)
        honest = [u for u in updates if u is not None]
        running_mu = {k: torch.stack([u[k].float() for u in honest])
                       .mean(0) for k in honest[0]}
        for i in range(config['n_clients']):
            if updates[i] is None:
                updates[i] = clients[i].local_train(
                    gw, D_attack_adv=splits['server_val_attack'],
                    mu_benign=running_mu,
                    n_clients=config['n_clients'], n_byz=n_byz)
        server.run_round(t, updates)

    # Probe: ell(w + alpha * delta_w_i) for alpha in [0, alpha_max]
    w_base = server.broadcast()
    gw = server.broadcast()
    updates_now = [None] * config['n_clients']
    for i in range(config['n_clients']):
        if i not in byz_idx:
            updates_now[i] = clients[i].local_train(gw)
    honest = [u for u in updates_now if u is not None]
    running_mu = {k: torch.stack([u[k].float() for u in honest]).mean(0)
                   for k in honest[0]}
    for i in range(config['n_clients']):
        if updates_now[i] is None:
            updates_now[i] = clients[i].local_train(
                gw, D_attack_adv=splits['server_val_attack'],
                mu_benign=running_mu,
                n_clients=config['n_clients'], n_byz=n_byz)

    alphas = np.linspace(0.0, alpha_max, n_alpha)
    rows = []
    curves = {}
    for i, delta in enumerate(updates_now):
        ys = []
        for a in alphas:
            ys.append(_ell(server.global_model, w_base, delta,
                            float(a), splits['server_val_attack']))
        unim = _is_unimodal(ys)
        cdif = _convexity_diffs(ys, alphas)
        pct_positive = float(np.mean(np.array(cdif) > 0)) * 100.0
        rows.append({
            'client': i,
            'is_byzantine': int(i in byz_idx),
            'unimodal': int(unim),
            'pct_convex_diffs_positive': round(pct_positive, 1),
            'min_alpha': float(alphas[int(np.argmin(ys))]),
            'min_ell': float(min(ys)),
            'max_ell': float(max(ys)),
            'range': float(max(ys) - min(ys)),
        })
        curves[i] = ys
        print(f"  client {i:>2} {'BYZ' if i in byz_idx else 'hon'}: "
              f"unimodal={unim}  pct_convex={pct_positive:.1f}%  "
              f"min@alpha={alphas[int(np.argmin(ys))]:.2f}")

    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(base, 'results',
                            f'convexity_{dataset}_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'convexity.csv'), index=False)

    # Save the full per-alpha curves (long format) so the figure can
    # be regenerated independently in Colab/Jupyter.
    long_rows = []
    for client_id, ys in curves.items():
        for a_idx, a_val in enumerate(alphas):
            long_rows.append({
                'client': client_id,
                'is_byzantine': int(client_id in byz_idx),
                'alpha': float(a_val),
                'ell':   float(ys[a_idx]),
            })
    pd.DataFrame(long_rows).to_csv(
        os.path.join(out_dir, 'convexity_curves.csv'), index=False)

    # Plot all curves on one axis (log scale on ell since it can span
    # orders of magnitude for malicious deltas)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, ys in curves.items():
        ls = '-' if i in byz_idx else '--'
        col = '#c44e52' if i in byz_idx else '#888888'
        ax.plot(alphas, ys, ls=ls, color=col, alpha=0.85,
                 label=f"client {i} {'(BYZ)' if i in byz_idx else ''}")
    ax.axvline(1.0, ls=':', color='k', alpha=0.5, label=r'$\alpha=1$')
    ax.set_xlabel(r'step scale $\alpha$ along $\Delta w_i$')
    ax.set_ylabel(r'$\ell(w + \alpha \Delta w_i)$  (Eq.~\ref{eq:denoising_residual})')
    ax.set_yscale('symlog', linthresh=1e-2)
    ax.set_title(f'Empirical convexity probe of $\\ell$ along each '
                  f'client update ({dataset.upper()}, seed={seed})')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'convexity_probe.png'), dpi=130)
    plt.close()

    n_unim = int(df.unimodal.sum())
    pct = float(df.pct_convex_diffs_positive.mean())
    print(f"\nSummary: {n_unim}/{len(df)} clients have unimodal ell-curve;"
          f"  mean pct convex-positive diffs = {pct:.1f}%")
    print(f"Saved: {out_dir}")
    return out_dir, df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='batadal')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--after-rounds', type=int, default=10)
    args = ap.parse_args()
    main(args.dataset, seed=args.seed, after_rounds=args.after_rounds)
