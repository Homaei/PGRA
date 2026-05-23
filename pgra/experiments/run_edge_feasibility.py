"""
Edge-device feasibility benchmark for the per-client local
training step of the PGRA GAE.

For each (dataset, client) we measure on the host CPU:
  - wall-clock time per local epoch (median of 5 runs)
  - peak resident memory (psutil), as a proxy for the working
    set of the edge process
  - parameter footprint (number of trainable scalars and
    bytes of float32 storage), which equals the communication
    cost per round per client
  - estimated energy on a representative edge platform
    (NVIDIA Jetson Nano-class profile, 10 W TDP), under the
    standard E = P_avg * t approximation; we cite this profile
    rather than measuring it directly.

Outputs a CSV and a Markdown summary suitable for the
manuscript's Scalability subsection.
"""
import argparse
import copy
import gc
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
import yaml

from pgra.models.gae import GraphAutoEncoder
from pgra.experiments.run_main import load_splits, set_seed


# Conservative edge-device TDPs (W) reported by the vendors for
# anomaly-detection-class workloads. We use the geometric mean
# for the per-row "estimated energy" column.
EDGE_PROFILES = {
    'RaspberryPi4_4GB':       3.0,   # 5V * 0.6A typical inference
    'Jetson_Nano_4GB':       10.0,
    'Jetson_Xavier_NX':      15.0,
    'CoralDevBoard':          4.5,
}
EDGE_GEOMEAN_W = float(np.exp(np.mean(np.log(list(EDGE_PROFILES.values())))))


def _flat_param_bytes(model):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n, n * 4  # float32


def _bench_client(model_proto, data_list, local_epochs, lr, n_repeats=5):
    """Median per-epoch wall-time and peak RSS for one client shard."""
    from torch_geometric.loader import DataLoader
    proc = psutil.Process(os.getpid())
    dataloader = DataLoader(data_list, batch_size=32, shuffle=True)
    times = []
    peak_rss_mb = proc.memory_info().rss / (1024 ** 2)
    for _ in range(n_repeats):
        m = copy.deepcopy(model_proto)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        m.train()
        t0 = time.perf_counter()
        for _e in range(local_epochs):
            for batch in dataloader:
                opt.zero_grad()
                x_c = batch.x.clone()
                for j in range(batch.num_graphs):
                    s = batch.ptr[j].item(); e = batch.ptr[j + 1].item()
                    t_idx = torch.randint(s, e, (1,)).item()
                    x_c[t_idx] = x_c[t_idx] * 1.8
                h_out = m(x_c, batch.edge_index)
                loss = F.mse_loss(h_out, batch.x)
                loss.backward()
                opt.step()
        t1 = time.perf_counter()
        times.append((t1 - t0) / local_epochs)
        peak_rss_mb = max(peak_rss_mb,
                           proc.memory_info().rss / (1024 ** 2))
        del m, opt
        gc.collect()
    return float(np.median(times)), float(peak_rss_mb)


def main(dataset='batadal', n_repeats=5):
    base = os.path.join(os.path.dirname(__file__), '..')
    config = yaml.safe_load(open(os.path.join(base, 'config',
                                              f'{dataset}.yaml')))
    splits = load_splits(dataset, os.path.join(base, 'data'))
    input_dim = splits['client_splits'][0][0].x.shape[1]
    proto = GraphAutoEncoder(input_dim=input_dim,
                              hidden_dim=config['model']['hidden_dim'],
                              latent_dim=config['model']['latent_dim'])
    n_params, n_bytes = _flat_param_bytes(proto)

    set_seed(42)
    rows = []
    print(f"=== {dataset.upper()} edge benchmark ===")
    print(f"Param count = {n_params:,d}  ({n_bytes/1024:.1f} KB per client per round)\n")
    for i, shard in enumerate(splits['client_splits']):
        t_med, rss = _bench_client(proto, shard,
                                     local_epochs=config['local_epochs'],
                                     lr=config.get('lr', 1e-3),
                                     n_repeats=n_repeats)
        # Estimated energy on the geomean edge profile (E = P * t).
        e_joules_geomean = EDGE_GEOMEAN_W * t_med
        e_joules_jetson = EDGE_PROFILES['Jetson_Nano_4GB'] * t_med
        rows.append({
            'client': i,
            'shard_size': len(shard),
            'host_cpu_epoch_s': round(t_med, 3),
            'peak_rss_MB': round(rss, 1),
            'param_count': n_params,
            'param_bytes_per_round': n_bytes,
            'est_energy_J_per_epoch_geomean': round(e_joules_geomean, 2),
            'est_energy_J_per_epoch_jetson_nano':
                round(e_joules_jetson, 2),
        })
        print(f"  client {i:>2}  shard={len(shard):>5}  "
              f"epoch={t_med*1000:6.1f} ms  RSS={rss:6.1f} MB  "
              f"E_Jetson={e_joules_jetson*1000:.0f} mJ/epoch")

    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(base, 'results',
                            f'edge_bench_{dataset}_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'edge_bench.csv'), index=False)

    # Print summary
    tot_round_jetson = (df['host_cpu_epoch_s'].sum()
                        * config['local_epochs']
                        * EDGE_PROFILES['Jetson_Nano_4GB'])
    print(f"\nSummary ({dataset.upper()}, {config['n_clients']} clients):")
    print(f"  median per-client per-epoch time = "
          f"{df.host_cpu_epoch_s.median()*1000:.1f} ms  "
          f"(min {df.host_cpu_epoch_s.min()*1000:.1f} ms, "
          f"max {df.host_cpu_epoch_s.max()*1000:.1f} ms)")
    print(f"  per-client comm cost            = "
          f"{n_bytes/1024:.1f} KB / round")
    print(f"  est. round energy on N Jetson Nano = "
          f"{tot_round_jetson:.2f} J/round (sum of clients)")
    print(f"  saved: {out_dir}")
    return out_dir, df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='batadal')
    ap.add_argument('--n_repeats', type=int, default=5)
    args = ap.parse_args()
    main(args.dataset, n_repeats=args.n_repeats)
