"""
Ablation experiments for PGRA.

Three experiments are reported per dataset:

  A. alpha sweep — how the sensitivity constant alpha shapes the trust
     decay. Bigger alpha => sharper rejection of malicious updates.
  B. attacker epsilon_s sweep — how the statistical-stealthiness budget
     that the attacker uses affects PGRA's ASR.
  C. adaptive vs static beta — does the adaptive rule actually buy us
     anything compared to the best static beta?

Each variant is averaged over the dataset's configured seeds.
"""
import argparse
import copy
import os
import yaml
from datetime import datetime

import pandas as pd
import torch

from pgra.models.gae import GraphAutoEncoder
from pgra.fl.client import FLClient, MaliciousClient
from pgra.fl.server import FLServer
from pgra.fl.aggregators.pgra import PGRAAggregator
from pgra.experiments.run_main import _evaluate, load_splits, set_seed


def _run(config, seed, agg, splits, model_proto, byz_indices, epsilon_s=None):
    set_seed(seed)
    model = copy.deepcopy(model_proto)
    server = FLServer(
        initial_model=model, aggregator=agg,
        D_val=splits['server_val'],
        D_val_attack=splits['server_val_attack'],
        scaler_info=splits['scaler_info'],
        warmup_epochs=config.get('warmup_epochs', 20),
        warmup_lr=config.get('warmup_lr', 0.005),
        noise_scale=config.get('noise_scale', 1.8),
    )

    eps = epsilon_s if epsilon_s is not None else config.get('epsilon_s', 5.0)
    clients = []
    for i in range(config['n_clients']):
        if i in byz_indices:
            c = MaliciousClient(
                client_id=i, data_list=splits['client_splits'][i],
                model_arch=server.global_model, local_epochs=2,
                backdoor_epochs=config.get('backdoor_epochs', 10),
                lr=config.get('lr', 1e-3),
                scaler_info=splits['scaler_info'], epsilon_s=eps)
        else:
            c = FLClient(client_id=i, data_list=splits['client_splits'][i],
                          model_arch=server.global_model,
                          local_epochs=config['local_epochs'],
                          lr=config.get('lr', 1e-3),
                          scaler_info=splits['scaler_info'])
        clients.append(c)

    running_mu = None
    for t in range(config['n_rounds']):
        gw = server.broadcast()
        updates = [None] * config['n_clients']
        for i in range(config['n_clients']):
            if i not in byz_indices:
                updates[i] = clients[i].local_train(gw)
        honest = [u for u in updates if u is not None]
        if honest:
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

    f1, p, r, asr, thr = _evaluate(server.global_model,
                                    splits['test'],
                                    splits['server_val'],
                                    splits['D_target'])
    return {'F1': f1, 'Precision': p, 'Recall': r, 'ASR': asr,
            'Threshold': thr}


def main(dataset='batadal'):
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
    seeds = config['seeds']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(base, 'results',
                            f'ablation_{dataset}_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)

    print("\n=== Ablation A: alpha sweep ===")
    rows_a = []
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        for s in seeds:
            agg = PGRAAggregator(alpha=alpha, gamma=config['pgra']['gamma'])
            r = _run(config, s, agg, splits, proto, byz_indices)
            r.update({'alpha': alpha, 'seed': s})
            rows_a.append(r)
            print(f"  alpha={alpha:<6}  seed={s}  "
                  f"F1={r['F1']:.4f}  ASR={r['ASR']:.4f}")
    pd.DataFrame(rows_a).to_csv(
        os.path.join(out_dir, 'alpha_sweep.csv'), index=False)

    print("\n=== Ablation B: attacker epsilon_s sweep ===")
    rows_b = []
    for eps in [1.0, 2.0, 5.0, 10.0, 20.0]:
        for s in seeds:
            agg = PGRAAggregator(alpha=config['pgra']['alpha'],
                                  gamma=config['pgra']['gamma'])
            r = _run(config, s, agg, splits, proto, byz_indices,
                     epsilon_s=eps)
            r.update({'epsilon_s': eps, 'seed': s})
            rows_b.append(r)
            print(f"  eps_s={eps:<6}  seed={s}  "
                  f"F1={r['F1']:.4f}  ASR={r['ASR']:.4f}")
    pd.DataFrame(rows_b).to_csv(
        os.path.join(out_dir, 'epsilon_s_sweep.csv'), index=False)

    print("\n=== Ablation C: adaptive vs static beta ===")
    rows_c = []
    for spec in ['adaptive', 'static_10', 'static_50', 'static_200']:
        for s in seeds:
            if spec == 'adaptive':
                agg = PGRAAggregator(alpha=config['pgra']['alpha'],
                                      gamma=config['pgra']['gamma'])
            else:
                bv = float(spec.split('_')[1])
                agg = PGRAAggregator(use_static_beta=True, beta_value=bv)
            r = _run(config, s, agg, splits, proto, byz_indices)
            r.update({'beta_mode': spec, 'seed': s})
            rows_c.append(r)
            print(f"  {spec:<12}  seed={s}  "
                  f"F1={r['F1']:.4f}  ASR={r['ASR']:.4f}")
    pd.DataFrame(rows_c).to_csv(
        os.path.join(out_dir, 'beta_mode.csv'), index=False)

    print(f"\nAblation results saved to {out_dir}")
    return out_dir


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='batadal')
    args = ap.parse_args()
    main(args.dataset)
