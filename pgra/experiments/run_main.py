"""
Main comparison experiment for the PGRA framework.

For each (dataset, aggregator) combination we run `n_rounds` of federated
learning with `n_byz` malicious clients launching the physics-aware
backdoor described in `pgra/fl/client.py::MaliciousClient`. After
training we evaluate the global GAE on:

  - Test set (binary normal/attack):  F1, Precision, Recall
  - D_target backdoor samples:        ASR = fraction predicted as normal

Each (dataset, aggregator) combination is repeated across `seeds`. The
reported metric is mean +/- std.

Usage:
    python -m pgra.experiments.run_main --dataset batadal
    python -m pgra.experiments.run_main --dataset wadi
"""
import argparse
import copy
import os
import sys
import random
import yaml
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from pgra.models.gae import GraphAutoEncoder
from pgra.fl.client import FLClient, MaliciousClient
from pgra.fl.server import FLServer
from pgra.fl.aggregators.pgra import PGRAAggregator
from pgra.fl.aggregators.fedavg import FedAvgAggregator
from pgra.fl.aggregators.krum import KrumAggregator
from pgra.fl.aggregators.coord_median import CoordMedianAggregator
from pgra.fl.aggregators.fltrust import FLTrustAggregator
from pgra.fl.aggregators.flame import FLAMEAggregator
from pgra.fl.aggregators.flair import FLAIRAggregator
from pgra.fl.aggregators.sine import SineAggregator
from pgra.fl.aggregators.fedrola import FedRoLAAggregator
from pgra.fl.aggregators.rfl_apia import RFLAPIAAggregator
from pgra.fl.aggregators.centralized_oracle import CentralizedOracleAggregator
from pgra.evaluation.metrics import compute_f1_precision_recall, compute_asr


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _build_aggregators(config, n_byz, server_val_sample):
    return {
        'PGRA':              PGRAAggregator(alpha=config['pgra']['alpha'],
                                            gamma=config['pgra']['gamma']),
        'FedAvg':            FedAvgAggregator(),
        'Krum':              KrumAggregator(f=n_byz),
        'CoordMedian':       CoordMedianAggregator(),
        'FLTrust':           FLTrustAggregator(root_dataset=server_val_sample),
        'FLAME':             FLAMEAggregator(),
        'FLAIR':             FLAIRAggregator(N=config['n_clients']),
        'Sine':              SineAggregator(),
        'FedRoLA':           FedRoLAAggregator(),
        'RFL-APIA':          RFLAPIAAggregator(N=config['n_clients']),
        'CentralizedOracle': CentralizedOracleAggregator(),
    }


def _per_sample_score(model, data):
    """Hybrid anomaly score combining per-node MAX (sensitive to single-
    sensor FDI) and MEAN (sensitive to broad backdoor degradation).
    Weighted toward MAX since FDI affects only one node per sample."""
    h_out = model(data.x, data.edge_index)
    err = ((h_out - data.x) ** 2).mean(dim=-1)  # [n_nodes]
    return (0.7 * err.max() + 0.3 * err.mean()).item()


def _evaluate(model, test_data, server_val, D_target):
    """Returns (f1, precision, recall, asr, threshold)."""
    model.eval()
    with torch.no_grad():
        normal_errors = [_per_sample_score(model, d) for d in server_val]
        threshold = float(np.percentile(normal_errors, 95))

        test_scores = [_per_sample_score(model, d) for d in test_data]
        y_true = np.array([d.y.item() for d in test_data], dtype=np.float32)
        y_pred = (np.array(test_scores) > threshold).astype(np.float32)

    f1, prec, rec = compute_f1_precision_recall(y_true, y_pred)

    with torch.no_grad():
        target_scores = [_per_sample_score(model, d) for d in D_target]
    target_pred = (np.array(target_scores) > threshold).astype(np.float32)
    asr = float((target_pred == 0).mean())
    return f1, prec, rec, asr, threshold


def _run_one(dataset, config, seed, agg_name, splits, model_proto, byz_indices):
    set_seed(seed)

    aggregators = _build_aggregators(config, len(byz_indices),
                                     splits['server_val'][0])
    aggregator = aggregators[agg_name]

    model = copy.deepcopy(model_proto)
    server = FLServer(
        initial_model=model,
        aggregator=aggregator,
        D_val=splits['server_val'],
        D_val_attack=splits['server_val_attack'],
        scaler_info=splits['scaler_info'],
        warmup_epochs=config.get('warmup_epochs', 20),
        warmup_lr=config.get('warmup_lr', 0.005),
        noise_scale=config.get('noise_scale', 1.8),
    )

    clients = []
    for i in range(config['n_clients']):
        if i in byz_indices and agg_name != 'CentralizedOracle':
            c = MaliciousClient(
                client_id=i,
                data_list=splits['client_splits'][i],
                model_arch=server.global_model,
                local_epochs=2,
                backdoor_epochs=config.get('backdoor_epochs', 8),
                lr=config.get('lr', 1e-3),
                scaler_info=splits['scaler_info'],
                epsilon_s=config.get('epsilon_s', 5.0),
                boost_factor=config.get('boost_factor',
                                          float(config['n_clients']) / 2.0),
            )
        else:
            c = FLClient(
                client_id=i,
                data_list=splits['client_splits'][i],
                model_arch=server.global_model,
                local_epochs=config['local_epochs'],
                lr=config.get('lr', 1e-3),
                scaler_info=splits['scaler_info'],
            )
        clients.append(c)

    running_mu = None
    for t in range(config['n_rounds']):
        gw = server.broadcast()
        updates = [None] * config['n_clients']
        # Honest pass first to compute mu_benign for adversaries
        for i in range(config['n_clients']):
            if i not in byz_indices or agg_name == 'CentralizedOracle':
                updates[i] = clients[i].local_train(gw)
        honest_us = [u for u in updates if u is not None]
        if honest_us:
            mu = {k: torch.stack([u[k].float() for u in honest_us]).mean(dim=0)
                  for k in honest_us[0]}
            running_mu = mu

        # Malicious updates with stealthiness projection against mu_benign
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


def load_splits(dataset, root):
    splits = {}
    splits['server_val'] = torch.load(
        os.path.join(root, 'splits', f'{dataset}_server_val.pt'),
        weights_only=False)
    splits['server_val_attack'] = torch.load(
        os.path.join(root, 'splits', f'{dataset}_server_val_attack.pt'),
        weights_only=False)
    splits['D_target'] = torch.load(
        os.path.join(root, 'splits', f'{dataset}_D_target.pt'),
        weights_only=False)
    splits['client_splits'] = torch.load(
        os.path.join(root, 'splits', f'{dataset}_client_splits.pt'),
        weights_only=False)
    splits['test'] = torch.load(
        os.path.join(root, 'processed', f'{dataset}_test.pt'),
        weights_only=False)
    splits['scaler_info'] = torch.load(
        os.path.join(root, 'processed', f'{dataset}_scaler.pt'),
        weights_only=False)
    return splits


def main(dataset='batadal', aggregators=None, seeds=None):
    base = os.path.join(os.path.dirname(__file__), '..')
    config = yaml.safe_load(open(os.path.join(base, 'config',
                                              f'{dataset}.yaml')))
    if seeds is None:
        seeds = config['seeds']
    splits = load_splits(dataset, os.path.join(base, 'data'))

    input_dim = splits['client_splits'][0][0].x.shape[1]
    proto = GraphAutoEncoder(input_dim=input_dim,
                              hidden_dim=config['model']['hidden_dim'],
                              latent_dim=config['model']['latent_dim'])

    n_byz = max(1, int(round(config['n_clients']
                              * config['byzantine_ratio'])))
    byz_indices = list(range(n_byz))
    print(f"[{dataset}] n_clients={config['n_clients']} n_byz={n_byz}  "
          f"byz_indices={byz_indices}  seeds={seeds}  "
          f"rounds={config['n_rounds']}")

    if aggregators is None:
        aggregators = ['PGRA', 'FedAvg', 'Krum', 'CoordMedian', 'FLTrust',
                       'FLAME', 'FLAIR', 'Sine', 'FedRoLA', 'RFL-APIA',
                       'CentralizedOracle']

    rows = []
    for agg in aggregators:
        seed_rows = []
        for s in seeds:
            print(f"  -> {agg} | seed={s}")
            r = _run_one(dataset, config, s, agg, splits, proto, byz_indices)
            r.update({'Aggregator': agg, 'Seed': s, 'Dataset': dataset})
            print(f"     F1={r['F1']:.4f}  P={r['Precision']:.4f}  "
                  f"R={r['Recall']:.4f}  ASR={r['ASR']:.4f}  "
                  f"thr={r['Threshold']:.5f}")
            seed_rows.append(r)
            rows.append(r)

    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(base, 'results',
                            f'main_{dataset}_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'all_runs.csv'), index=False)

    agg_df = (df.groupby('Aggregator')[['F1', 'Precision', 'Recall', 'ASR']]
                .agg(['mean', 'std'])
                .round(4))
    agg_df.to_csv(os.path.join(out_dir, 'summary.csv'))
    print("\n=== SUMMARY ===")
    print(agg_df)
    print(f"\nSaved to {out_dir}")
    return df, agg_df, out_dir


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='batadal',
                    choices=['batadal', 'wadi'])
    ap.add_argument('--aggregators', nargs='+', default=None)
    ap.add_argument('--seeds', nargs='+', type=int, default=None)
    args = ap.parse_args()
    main(args.dataset, args.aggregators, args.seeds)
