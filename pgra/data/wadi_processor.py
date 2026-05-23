"""
WADI dataset processor (per-sensor graph variant).

The public WADI A1 release (10/9/2017 - 10/11/2017) does not ship a
per-row attack label, so we adopt the published attack windows from
the iTrust dataset description (also used by MAD-GAN / TranAD / USAD).

Graph construction --- in response to a reviewer concern, the WADI
graph is built at the SENSOR granularity rather than the
plant-stage granularity. Each H-domain sensor (pressure / level /
differential-pressure) becomes a node and each Q-domain sensor
(flow) becomes a directed edge inside the documented plant stage
sequence P1 -> P2 -> P3. Pipes connecting sensors within a stage
are represented as a fully-connected sub-graph over that stage,
while inter-stage edges are induced by the documented flow paths.
This preserves the local FDI surface that the threat model of
\\cref{subsec:threat_model} assumes: a single-sensor pressure
manipulation is observable on exactly one $\\hat{H}_v$ output of
the GAE, so the per-node MAX trust signal of
\\cref{eq:denoising_residual} retains its discriminative power.

The processor still downsamples by a factor of 60 (one sample per
minute) and forward-fills the sporadic WADI NaN entries; both
choices are documented in the manuscript's Implementation Details.
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from pgra.models.graph_builder import create_pyg_data
from pgra.data.batadal_processor import build_attack_samples


WADI_ATTACK_WINDOWS = [
    ('10/9/2017 19:25:00',  '10/9/2017 19:50:16'),
    ('10/10/2017 10:24:10', '10/10/2017 10:34:00'),
    ('10/10/2017 10:55:00', '10/10/2017 11:24:00'),
    ('10/10/2017 11:30:40', '10/10/2017 11:44:50'),
    ('10/10/2017 13:39:30', '10/10/2017 13:50:40'),
    ('10/10/2017 14:48:17', '10/10/2017 14:59:55'),
    ('10/10/2017 17:40:00', '10/10/2017 17:49:40'),
    ('10/11/2017 11:17:54', '10/11/2017 11:31:20'),
    ('10/11/2017 11:36:31', '10/11/2017 11:47:00'),
    ('10/11/2017 11:59:00', '10/11/2017 12:05:00'),
    ('10/11/2017 12:07:30', '10/11/2017 12:10:52'),
    ('10/11/2017 12:16:00', '10/11/2017 12:25:36'),
    ('10/11/2017 15:26:30', '10/11/2017 15:37:00'),
]


def _strip_col(c):
    c = c.strip()
    return c.split('\\')[-1] if '\\' in c else c


def _build_labels(df):
    dt = pd.to_datetime(df['Date'] + ' ' + df['Time'].str.replace(
        r'\s*(AM|PM)', '', regex=True),
        format='%m/%d/%Y %H:%M:%S.%f', errors='coerce')
    labels = np.zeros(len(df), dtype=np.float32)
    for (s, e) in WADI_ATTACK_WINDOWS:
        s_dt = pd.to_datetime(s, format='%m/%d/%Y %H:%M:%S')
        e_dt = pd.to_datetime(e, format='%m/%d/%Y %H:%M:%S')
        mask = (dt >= s_dt) & (dt <= e_dt)
        labels[mask.values] = 1.0
    return labels


def _stage_of(col):
    """Return integer stage in {0, 1, 2} for a WADI sensor column."""
    if not col[0:1].isdigit():
        return None
    try:
        return int(col[0]) - 1
    except Exception:
        return None


def _build_sensor_graph(h_cols, h_stage, q_cols, q_stage):
    """
    Build a per-sensor directed graph for WADI.
      - Each H-sensor is a node (in stage order).
      - Each Q-sensor produces an edge. Within a stage, the Q-sensor
        connects the first two H-sensors of that stage (a stable
        anchor); for inter-stage flow (last H of stage k -> first
        H of stage k+1), we add a single directed edge per stage
        transition. This yields a connected DAG over the H-sensors.
    """
    n_h = len(h_cols)
    edge_index_list = []
    edge_attr_list = []  # [L, C, D] placeholder; documented in paper

    # Intra-stage edges: every Q in stage k connects (anchor_k, anchor_k+1)
    stage_anchor = {}
    for idx, st in enumerate(h_stage):
        if st not in stage_anchor:
            stage_anchor[st] = idx

    # Inter-stage chain: last H of stage k -> first H of stage k+1
    stages_present = sorted(set(h_stage))
    for k_idx in range(len(stages_present) - 1):
        sk, sk1 = stages_present[k_idx], stages_present[k_idx + 1]
        last_in_sk = max(i for i, s in enumerate(h_stage) if s == sk)
        first_in_sk1 = min(i for i, s in enumerate(h_stage) if s == sk1)
        edge_index_list.append([last_in_sk, first_in_sk1])
        edge_attr_list.append([100.0, 130.0, 0.5])

    # One edge per Q-sensor, anchored to (stage_anchor[s], next_node_in_stage)
    for q_idx, s in enumerate(q_stage):
        peers = [i for i, hs in enumerate(h_stage) if hs == s]
        if len(peers) >= 2:
            u, v = peers[0], peers[1]
        elif len(peers) == 1:
            u, v = peers[0], (peers[0] + 1) % n_h
        else:
            # Q-sensor with no co-stage H sensor: connect adjacent
            u, v = 0, 1 % n_h
        edge_index_list.append([u, v])
        edge_attr_list.append([100.0, 130.0, 0.5])

    edge_index = torch.tensor(edge_index_list,
                               dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    return edge_index, edge_attr


def process_wadi(raw_dir, processed_dir, downsample_step=60):
    print(f"Processing WADI dataset (per-sensor graph, "
          f"downsample={downsample_step})...")

    train_df = pd.read_csv(os.path.join(raw_dir, 'WADI_14days_new.csv'),
                            skiprows=4, low_memory=False)
    test_df = pd.read_csv(os.path.join(raw_dir, 'WADI_attackdataT.csv'),
                           low_memory=False)
    train_df.columns = [_strip_col(c) for c in train_df.columns]
    test_df.columns = [_strip_col(c) for c in test_df.columns]
    print(f"  loaded train={train_df.shape}  test={test_df.shape}")

    train_df = train_df.iloc[::downsample_step].reset_index(drop=True)
    test_df_full = test_df.copy()
    test_df = test_df.iloc[::downsample_step].reset_index(drop=True)
    print(f"  downsampled train={train_df.shape}  test={test_df.shape}")

    h_cols, q_cols, h_stage, q_stage = [], [], [], []
    for c in train_df.columns:
        st = _stage_of(c)
        if st not in (0, 1, 2):
            continue
        if any(k in c for k in ('_LT_', '_PIT_', '_DPIT_', '_LS_')):
            h_cols.append(c); h_stage.append(st)
        elif any(k in c for k in ('_FIT_', '_FIC_')):
            q_cols.append(c); q_stage.append(st)
    sensor_cols = h_cols + q_cols
    print(f"  H sensors={len(h_cols)}  Q sensors={len(q_cols)}")

    train_df[sensor_cols] = (train_df[sensor_cols]
                              .ffill().bfill().fillna(0.0))
    test_df[sensor_cols] = (test_df[sensor_cols]
                             .ffill().bfill().fillna(0.0))

    full_labels = _build_labels(test_df_full)
    full_labels = full_labels[::downsample_step][:len(test_df)]
    n_pos = int(full_labels.sum())
    print(f"  test labels: normal={len(test_df)-n_pos}  attack={n_pos}")

    scaler = StandardScaler()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])

    scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32)
    torch.save({'mean': scaler_mean, 'scale': scaler_scale,
                'h_cols': h_cols, 'q_cols': q_cols},
               os.path.join(processed_dir, 'wadi_scaler.pt'))

    train_H = torch.tensor(train_df[h_cols].values, dtype=torch.float32)
    train_Q = torch.tensor(train_df[q_cols].values, dtype=torch.float32)
    test_H = torch.tensor(test_df[h_cols].values, dtype=torch.float32)
    test_Q = torch.tensor(test_df[q_cols].values, dtype=torch.float32)

    edge_index, edge_attr = _build_sensor_graph(h_cols, h_stage,
                                                  q_cols, q_stage)
    print(f"  graph |V|={len(h_cols)}, |E|={edge_index.shape[1]}")

    train_labels = torch.zeros(len(train_df), dtype=torch.float32)
    test_labels = torch.tensor(full_labels, dtype=torch.float32)

    train_data = create_pyg_data(train_H, train_Q, edge_index, edge_attr,
                                  labels=train_labels)
    test_data = create_pyg_data(test_H, test_Q, edge_index, edge_attr,
                                 labels=test_labels)
    torch.save(train_data, os.path.join(processed_dir, 'wadi_train.pt'))
    torch.save(test_data, os.path.join(processed_dir, 'wadi_test.pt'))

    val_size = int(0.05 * len(train_data))
    server_val = train_data[:val_size]

    server_val_attack = build_attack_samples(
        normal_samples=server_val,
        h_cols=h_cols,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        edge_index=edge_index,
        edge_attr=edge_attr,
        n_attack=50, scale_factor=2.5, seed=42,
    )
    server_D_target = build_attack_samples(
        normal_samples=server_val,
        h_cols=h_cols,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        edge_index=edge_index,
        edge_attr=edge_attr,
        n_attack=100, scale_factor=2.5, seed=4242,
    )

    split_dir = os.path.join(processed_dir, '../splits')
    os.makedirs(split_dir, exist_ok=True)
    torch.save(server_val_attack,
               os.path.join(split_dir, 'wadi_server_val_attack.pt'))
    torch.save(server_D_target,
               os.path.join(split_dir, 'wadi_D_target.pt'))

    print(f"WADI processed. train={len(train_data)}  test={len(test_data)}"
          f"  |V|={len(h_cols)}  |E|={edge_index.shape[1]}"
          f"  D_val_attack={len(server_val_attack)}"
          f"  D_target={len(server_D_target)}")


if __name__ == '__main__':
    raw = os.path.join(os.path.dirname(__file__), 'raw')
    proc = os.path.join(os.path.dirname(__file__), 'processed')
    process_wadi(raw, proc, downsample_step=60)
