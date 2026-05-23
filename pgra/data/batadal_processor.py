import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from pgra.models.graph_builder import build_batadal_graph, create_pyg_data


def build_attack_samples(normal_samples, h_cols, scaler_mean, scaler_scale,
                         edge_index, edge_attr, n_attack=50, scale_factor=5.0,
                         seed=42):
    """
    Builds (H_spoofed, Q_real) pairs by scaling ONE pressure/level sensor
    in physical units. Carries h_clean (normalized clean target) for the
    physics-aware trust signal.
    """
    n_h = len(h_cols)
    H_mean = scaler_mean[:n_h]
    H_scale = scaler_scale[:n_h]

    H_normal_all = torch.stack([d.x.squeeze(1) for d in normal_samples])
    Q_real_all = torch.stack([d.q for d in normal_samples])

    H_phys = H_normal_all * H_scale.unsqueeze(0) + H_mean.unsqueeze(0)

    gen = torch.Generator(); gen.manual_seed(seed)
    idx = torch.randperm(len(normal_samples), generator=gen)[:n_attack]
    H_phys_sample = H_phys[idx].clone()
    Q_real_sample = Q_real_all[idx]
    H_clean_sample = H_normal_all[idx].clone()

    gen2 = torch.Generator(); gen2.manual_seed(seed + 1)
    for i in range(n_attack):
        target = torch.randint(0, n_h, (1,), generator=gen2).item()
        H_phys_sample[i, target] = H_phys_sample[i, target] * scale_factor

    H_spoofed_norm = (H_phys_sample - H_mean.unsqueeze(0)) / H_scale.unsqueeze(0)

    dataset = create_pyg_data(
        H_spoofed_norm, Q_real_sample,
        edge_index, edge_attr, labels=None
    )

    for i in range(n_attack):
        dataset[i].h_clean = H_clean_sample[i].unsqueeze(1)
        dataset[i].y = torch.tensor([1.0], dtype=torch.float32)

    return dataset


def process_batadal(raw_dir, processed_dir,
                    inp_file='BATADAL_network.inp',
                    train_file='BATADAL_dataset03.csv',
                    test_file='BATADAL_dataset04.csv'):
    print("Processing BATADAL dataset...")

    inp_path = os.path.join(raw_dir, inp_file)
    train_df = pd.read_csv(os.path.join(raw_dir, train_file))
    test_df = pd.read_csv(os.path.join(raw_dir, test_file))

    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    h_cols = [c for c in train_df.columns
              if c.startswith('L_') or c.startswith('P_')]
    q_cols = [c for c in train_df.columns if c.startswith('F_')]

    sensor_cols = h_cols + q_cols

    edge_index, edge_attr, node_names, link_names = build_batadal_graph(
        inp_path, h_cols, q_cols)

    scaler = StandardScaler()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])

    scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32)
    torch.save({'mean': scaler_mean, 'scale': scaler_scale,
                'h_cols': h_cols, 'q_cols': q_cols},
               os.path.join(processed_dir, 'batadal_scaler.pt'))

    train_H = torch.tensor(train_df[h_cols].values, dtype=torch.float32)
    train_Q = torch.tensor(train_df[q_cols].values, dtype=torch.float32)
    train_labels = torch.zeros(len(train_df), dtype=torch.float32)

    test_H = torch.tensor(test_df[h_cols].values, dtype=torch.float32)
    test_Q = torch.tensor(test_df[q_cols].values, dtype=torch.float32)

    if 'ATT_FLAG' in test_df.columns:
        raw_labels = test_df['ATT_FLAG'].values.astype(np.float32)
        # -999 = unlabeled portion of challenge release => treated as NORMAL (0)
        # 1 = attack window
        clean_labels = np.where(raw_labels == 1.0, 1.0, 0.0)
        test_labels = torch.tensor(clean_labels, dtype=torch.float32)
    else:
        test_labels = torch.zeros(len(test_df), dtype=torch.float32)

    n_attack_test = int(test_labels.sum().item())
    n_normal_test = int((test_labels == 0).sum().item())
    print(f"  Test labels: normal={n_normal_test}, attack={n_attack_test}")

    train_data = create_pyg_data(train_H, train_Q, edge_index, edge_attr,
                                 labels=train_labels)
    test_data = create_pyg_data(test_H, test_Q, edge_index, edge_attr,
                                labels=test_labels)

    torch.save(train_data, os.path.join(processed_dir, 'batadal_train.pt'))
    torch.save(test_data, os.path.join(processed_dir, 'batadal_test.pt'))

    val_size = int(0.05 * len(train_data))
    server_val = train_data[:val_size]

    server_val_attack = build_attack_samples(
        normal_samples=server_val,
        h_cols=h_cols,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        edge_index=edge_index,
        edge_attr=edge_attr,
        n_attack=50,
        scale_factor=1.8,
        seed=42,
    )

    server_D_target = build_attack_samples(
        normal_samples=server_val,
        h_cols=h_cols,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        edge_index=edge_index,
        edge_attr=edge_attr,
        n_attack=100,
        scale_factor=1.8,
        seed=4242,
    )

    split_dir = os.path.join(processed_dir, '../splits')
    os.makedirs(split_dir, exist_ok=True)
    torch.save(server_val_attack,
               os.path.join(split_dir, 'batadal_server_val_attack.pt'))
    torch.save(server_D_target,
               os.path.join(split_dir, 'batadal_D_target.pt'))

    print(f"BATADAL processed. Train samples: {len(train_data)}, "
          f"Test samples: {len(test_data)}")


if __name__ == '__main__':
    raw = os.path.join(os.path.dirname(__file__), 'raw')
    proc = os.path.join(os.path.dirname(__file__), 'processed')
    process_batadal(raw, proc)
