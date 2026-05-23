"""
PGRA aggregator (Physics-Guided Robust Aggregation).

Trust signal for each tentative model w_tilde_i (=w_current + eta * dW_i):

    ell_i = || model(w_tilde_i)(H_spoof, E) - H_clean ||^2

(see physics/residuals.denoising_residual). Honest tentative models inherit
the warm-up denoiser's ability to project H_spoof onto H_clean, yielding
small ell. Malicious tentative models -- trained on a backdoor that maps
H_spoof to itself -- diverge from this denoising map, yielding large ell.

The aggregation then follows Algorithm 1 of the manuscript:
    beta_t = alpha / (Var(ell) + gamma)
    tau_i  = exp(-beta_t * (ell_i - min(ell)))
    w_next = sum_i tau_i_bar * w_tilde_i.

`use_static_beta` and `beta_value` enable ablation of the adaptive rule.
"""
import torch
import torch.nn.functional as F


class PGRAAggregator:
    def __init__(self, alpha=5.0, gamma=0.005,
                 use_static_beta=False, beta_value=50.0,
                 verbose=False):
        self.alpha = alpha
        self.gamma = gamma
        self.use_static_beta = use_static_beta
        self.beta_value = beta_value
        self.verbose = verbose

    @torch.no_grad()
    def _trust_signal(self, model, w, D_val_attack):
        """
        Physics-aware trust signal: per-sample, take the MAXIMUM per-node
        squared denoising error between the model's H_hat and H_clean.
        This mirrors the FDI threat model (a single spoofed sensor per
        attack sample) and the anomaly detector's per-node MAX score
        used at inference time, so the trust signal is aligned with
        what the model is ultimately expected to flag.
        """
        model.load_state_dict(w)
        model.eval()
        if not D_val_attack:
            return 0.0
        total = 0.0
        for data in D_val_attack:
            h_out = model(data.x, data.edge_index)
            err_per_node = ((h_out - data.h_clean) ** 2).mean(dim=-1)
            total += err_per_node.max().item()
        return total / len(D_val_attack)

    def aggregate(self, updates, w_current, model,
                  D_val_normal, D_val_attack, scaler_info=None,
                  baseline=None, eta=1.0):
        N = len(updates)

        ell_vals = []
        for delta_w_i in updates:
            w_tilde = {k: w_current[k] + eta * delta_w_i[k]
                       for k in w_current}
            ell_i = self._trust_signal(model, w_tilde, D_val_attack)
            ell_vals.append(ell_i)

        ell_t = torch.tensor(ell_vals, dtype=torch.float64)
        # Sanitise: replace inf/nan with the largest finite value so the
        # softmax stays well-defined even if a tentative model produced
        # garbage outputs (a sign of an extreme malicious update -- it
        # SHOULD be filtered out, not crash the aggregator).
        finite_mask = torch.isfinite(ell_t)
        if finite_mask.any():
            big = float(ell_t[finite_mask].max().item()) * 10.0 + 1e6
        else:
            big = 1e6
        ell_t = torch.where(finite_mask, ell_t, torch.tensor(big,
                                                              dtype=ell_t.dtype))

        if self.use_static_beta:
            beta_t = torch.tensor(self.beta_value, dtype=torch.float64)
        else:
            # Use Var(log(1+ell)) to be robust to extreme outliers (the
            # boosted malicious updates can produce ell values orders of
            # magnitude above honest ell, which would otherwise drive
            # raw variance through the roof and beta to ~0).
            log_ell = torch.log1p(ell_t)
            if N > 1:
                var_ell = torch.var(log_ell)
            else:
                var_ell = torch.tensor(1e-10)
            if torch.isnan(var_ell) or var_ell.item() < 1e-12:
                var_ell = torch.tensor(1e-12, dtype=torch.float64)
            beta_t = self.alpha / (var_ell + self.gamma)

        # Numerically stable softmax operating on log(1+ell) so that
        # large ell values do not collapse beta. The original Eq. 16
        # formulation is preserved by using log1p as the discriminative
        # scale (monotone in ell).
        log_ell = torch.log1p(ell_t)
        ell_min = log_ell.min()
        logits = -beta_t * (log_ell - ell_min)
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        tau = torch.exp(logits)
        tau_bar = tau / tau.sum()

        w_next = {k: w_current[k].clone() for k in w_current}
        for k in w_current:
            w_next[k] = w_current[k].clone()
        for k in w_current:
            agg_delta = torch.zeros_like(w_current[k])
            for i, tb in enumerate(tau_bar):
                tb_val = tb.item()
                if tb_val > 1e-12:
                    agg_delta = agg_delta + tb_val * updates[i][k]
            w_next[k] = w_current[k] + eta * agg_delta

        if self.verbose:
            print(f"  PGRA: ell={[f'{e:.4f}' for e in ell_vals]}  "
                  f"beta={beta_t.item():.2f}  "
                  f"tau_bar={[f'{t:.3f}' for t in tau_bar.tolist()]}")

        return w_next, {
            'ell': ell_vals,
            'beta': float(beta_t.item()),
            'tau_bar': tau_bar.tolist(),
        }
