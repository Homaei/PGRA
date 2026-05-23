"""
Physics-residual based loss for the PGRA framework.

Two complementary trust signals are exposed:

  - `denoising_residual(model, D_val_attack)`:
        L_dn(w) = mean_{(H_spoof, H_clean) in D_val_attack}
                    || model(H_spoof, E) - H_clean ||^2 / |V|
        Pure data-driven physics signal. An honest model trained as a
        denoiser maps H_spoof -> H_clean (low residual); a malicious model
        trained on the backdoor objective maps H_spoof -> H_spoof
        (high residual). This is the primary trust signal of PGRA.

  - `hazen_williams_residual(model, D_val_attack, edge_index, edge_attr,
                              scaler_info)`:
        L_hw(w) = (1 / (|D| * |E|)) sum_s sum_e R_hw(e; w)^2
        where R_hw uses model's reconstructed H_hat (unnormalised) and
        the observed Q (unnormalised). Used for ablation / theoretical
        consistency reporting.

Both functions are PyTorch-autograd compatible.
"""
import torch
import torch.nn.functional as F


def hazen_williams(L, Q, C, D, eps=1e-8):
    """
    h_f = 10.67 * L * |Q|^1.852 / (C^1.852 * D^4.87) * sgn(Q)
    All inputs broadcast-compatible. Returns h_f tensor.
    """
    abs_Q = torch.clamp(Q.abs(), min=eps)
    num = 10.67 * L * (abs_Q ** 1.852)
    den = (C.clamp(min=eps) ** 1.852) * (D.clamp(min=eps) ** 4.87)
    return (num / den) * torch.sign(Q)


def R_mass(Q_hat, edge_index, num_nodes):
    """Mass-conservation residual at every node (vectorised)."""
    if Q_hat.dim() == 1:
        Q_hat = Q_hat.unsqueeze(0)
    num_samples = Q_hat.shape[0]
    res = torch.zeros(num_samples, num_nodes, device=Q_hat.device)
    u, v = edge_index[0], edge_index[1]
    # flow leaves u, enters v
    res.index_add_(1, u, -Q_hat)
    res.index_add_(1, v,  Q_hat)
    return res


def R_hw(H_hat, Q, edge_index, edge_attr):
    """Hazen-Williams residual per edge using model's H_hat and observed Q."""
    u, v = edge_index[0], edge_index[1]
    h_f_pred = H_hat[:, u] - H_hat[:, v]
    L = edge_attr[:, 0].unsqueeze(0)
    C = edge_attr[:, 1].unsqueeze(0)
    D = edge_attr[:, 2].unsqueeze(0)
    h_f_hw = hazen_williams(L, Q, C, D)
    return h_f_pred - h_f_hw


@torch.no_grad()
def denoising_residual(model, D_val_attack):
    """
    Mean per-node squared error between model(H_spoof, E) and H_clean.

    This is the core PGRA trust signal: it captures whether the model
    correctly projects spoofed pressure inputs back onto the clean
    hydraulic manifold. Higher value => model accepts spoofed inputs
    (malicious behaviour). Lower value => model denoises spoofs
    (honest behaviour).
    """
    model.eval()
    if D_val_attack is None or len(D_val_attack) == 0:
        return 0.0
    total = 0.0
    for data in D_val_attack:
        h_out = model(data.x, data.edge_index)
        target = data.h_clean
        total += F.mse_loss(h_out, target, reduction='mean').item()
    return total / len(D_val_attack)


def L_phys(model, D_val, scaler_info, edge_index=None, edge_attr=None,
           lambda1=0.0, lambda2=1.0):
    """
    Hydraulic inconsistency loss (Eq. 13 of the manuscript). Reported for
    theoretical consistency and ablation; not the primary trust signal.

    Implementation note: pump links in BATADAL have placeholder pipe
    parameters, so the magnitude of L_phys here is a relative -- not
    absolute -- measure of hydraulic compliance.
    """
    model.eval()
    if D_val is None or len(D_val) == 0:
        return torch.tensor(0.0)

    n_samples = len(D_val)
    if edge_index is None:
        edge_index = D_val[0].edge_index
    if edge_attr is None:
        edge_attr = D_val[0].edge_attr
    num_nodes = D_val[0].x.shape[0]
    num_edges = edge_attr.shape[0]

    if scaler_info is not None:
        n_h = len(scaler_info['h_cols'])
        n_q = len(scaler_info['q_cols'])
        H_mean = scaler_info['mean'][:n_h]
        H_scale = scaler_info['scale'][:n_h]
        Q_mean = scaler_info['mean'][n_h:n_h + n_q]
        Q_scale = scaler_info['scale'][n_h:n_h + n_q]
    else:
        H_mean = H_scale = Q_mean = Q_scale = None

    total_mass = torch.tensor(0.0)
    total_hw = torch.tensor(0.0)

    for data in D_val:
        h_out = model(data.x, data.edge_index)
        H_hat = h_out.squeeze(-1).unsqueeze(0)
        Q_obs = data.q.unsqueeze(0)
        if H_mean is not None:
            H_hat = H_hat * H_scale.unsqueeze(0) + H_mean.unsqueeze(0)
            Q_obs = Q_obs * Q_scale.unsqueeze(0) + Q_mean.unsqueeze(0)
            Q_obs = Q_obs / 1000.0
        if lambda1 > 0.0:
            rm = R_mass(Q_obs, edge_index, num_nodes)
            total_mass = total_mass + (rm ** 2).sum()
        if lambda2 > 0.0:
            rh = R_hw(H_hat, Q_obs, edge_index, edge_attr)
            total_hw = total_hw + (rh ** 2).sum()

    L_mass = (lambda1 / (n_samples * num_nodes)) * total_mass
    L_hw = (lambda2 / (n_samples * num_edges)) * total_hw
    total = L_mass + L_hw
    if torch.isnan(total) or torch.isinf(total):
        return torch.tensor(1e6)
    return total
