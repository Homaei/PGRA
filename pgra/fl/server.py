"""
Federated server with warm-up denoising and pluggable aggregator.
"""
import copy
import torch
import torch.nn.functional as F


class FLServer:
    def __init__(self, initial_model, aggregator, D_val, D_val_attack=None,
                 pipe_params=None, scaler_info=None,
                 warmup_epochs=20, warmup_lr=0.005, noise_scale=1.8,
                 do_warmup=True):
        self.global_model = initial_model
        self.global_weights = initial_model.state_dict()
        self.aggregator = aggregator
        self.D_val = D_val
        self.D_val_attack = D_val_attack
        self.pipe_params = pipe_params
        self.scaler_info = scaler_info
        self.history = []
        self._baseline = 1.0

        if do_warmup and self.D_val is not None and len(self.D_val) > 0:
            self.global_model = self._warmup(
                self.global_model, self.D_val,
                warmup_epochs=warmup_epochs, lr=warmup_lr,
                noise_scale=noise_scale)
            self.global_weights = self.global_model.state_dict()

    def _warmup(self, model, D_val_normal, warmup_epochs, lr, noise_scale):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in range(warmup_epochs):
            for data in D_val_normal:
                x_c = data.x.clone()
                target = torch.randint(0, x_c.shape[0], (1,)).item()
                x_c[target] = x_c[target] * noise_scale
                h_out = model(x_c, data.edge_index)
                loss = F.mse_loss(h_out, data.x)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            total = 0.0
            for data in D_val_normal:
                h_out = model(data.x, data.edge_index)
                total += F.mse_loss(h_out, data.x, reduction='mean').item()
        self._baseline = total / len(D_val_normal)
        return model

    def run_round(self, t, client_updates):
        agg_name = self.aggregator.__class__.__name__
        if agg_name == 'PGRAAggregator':
            agg_result = self.aggregator.aggregate(
                updates=client_updates,
                w_current=self.global_weights,
                model=self.global_model,
                D_val_normal=self.D_val,
                D_val_attack=self.D_val_attack,
                scaler_info=self.scaler_info,
                baseline=self._baseline,
                eta=1.0,
            )
        else:
            agg_result = self.aggregator.aggregate(
                updates=client_updates,
                w_current=self.global_weights,
                model=self.global_model,
                D_val=self.D_val,
                scaler_info=self.scaler_info,
                eta=1.0,
            )

        if isinstance(agg_result, tuple):
            w_next, extra_logs = agg_result
        else:
            w_next, extra_logs = agg_result, {}

        self.global_weights = w_next
        self.global_model.load_state_dict(self.global_weights)
        self.history.append(extra_logs)
        return extra_logs

    def broadcast(self):
        return copy.deepcopy(self.global_weights)
