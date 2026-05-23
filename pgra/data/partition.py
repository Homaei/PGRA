import torch
import numpy as np
import os


def partition_data(data_list, n_clients, alpha_dir=0.5, seed=42, val_size=50):
    """
    Non-IID Dirichlet partition. Reserves `val_size` normal samples for D_val_server.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    d_val_size = min(val_size, int(len(data_list) * 0.05))
    server_val = data_list[:d_val_size]
    client_data = data_list[d_val_size:]

    proportions = np.random.dirichlet(np.repeat(alpha_dir, n_clients))
    proportions = np.maximum(proportions, 0.01)
    proportions = proportions / proportions.sum()

    num_samples = len(client_data)
    client_sizes = (proportions * num_samples).astype(int)
    client_sizes[-1] = num_samples - sum(client_sizes[:-1])

    indices = np.random.permutation(num_samples)

    client_splits = []
    start = 0
    for size in client_sizes:
        end = start + size
        client_indices = indices[start:end]
        split = [client_data[i] for i in client_indices]
        client_splits.append(split)
        start = end

    return server_val, client_splits


if __name__ == '__main__':
    proc_dir = os.path.join(os.path.dirname(__file__), 'processed')
    split_dir = os.path.join(os.path.dirname(__file__), 'splits')
    os.makedirs(split_dir, exist_ok=True)

    # ---- BATADAL ----
    batadal_train_path = os.path.join(proc_dir, 'batadal_train.pt')
    if os.path.exists(batadal_train_path):
        batadal_train = torch.load(batadal_train_path, weights_only=False)
        server_val, client_splits = partition_data(batadal_train, n_clients=9,
                                                    val_size=50)
        torch.save(server_val,
                   os.path.join(split_dir, 'batadal_server_val.pt'))
        torch.save(client_splits,
                   os.path.join(split_dir, 'batadal_client_splits.pt'))
        print(f"BATADAL partitioned: server_val={len(server_val)} "
              f"clients={[len(s) for s in client_splits]}")

    # ---- WADI ----
    wadi_train_path = os.path.join(proc_dir, 'wadi_train.pt')
    if os.path.exists(wadi_train_path):
        wadi_train = torch.load(wadi_train_path, weights_only=False)
        server_val, client_splits = partition_data(wadi_train, n_clients=10,
                                                    val_size=50, seed=42)
        torch.save(server_val,
                   os.path.join(split_dir, 'wadi_server_val.pt'))
        torch.save(client_splits,
                   os.path.join(split_dir, 'wadi_client_splits.pt'))
        print(f"WADI partitioned: server_val={len(server_val)} "
              f"clients={[len(s) for s in client_splits]}")
