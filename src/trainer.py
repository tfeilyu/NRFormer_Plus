import torch.optim as optim
import math
import torch
from src import utils


class Trainer():
    def __init__(self, model, config, scaler, device):
        self.model = model
        self.model.to(device)
        self.scaler = scaler
        self.config = config
        self.use_log_space = config.get('use_log_space', False)

        self.iter = 0
        self.task_level = 1
        self.seq_out_len = config['out_length']
        self.max_value = scaler.max_value

        self.loss = utils.masked_mae

        # Iter5: horizon-dependent loss weighting
        hw_strategy = config.get('horizon_weight', 'none')
        if hw_strategy != 'none':
            H = self.seq_out_len
            h = torch.arange(1, H + 1, dtype=torch.float32)
            if hw_strategy == 'linear':
                w = 1.0 + h / H  # 1.04 → 2.0
            elif hw_strategy == 'inverse_acf':
                w = 1.0 / (0.946 ** h + 0.1)  # weight inversely to autocorrelation
            else:
                w = torch.ones(H)
            w = w / w.mean()
            self.horizon_weights = w.view(1, 1, 1, H)
        else:
            self.horizon_weights = None

        self.weight_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config['weight_lr']),
            weight_decay=float(config['weight_decay'])
        )

        # Iteration 2: support cosine annealing with warmup
        scheduler_type = config.get('scheduler', 'multistep')
        if scheduler_type == 'cosine':
            warmup_epochs = config.get('warmup_epochs', 5)
            total_epochs = config.get('epochs', 200)
            # Cosine annealing after warmup
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.weight_optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
            )
            # Linear warmup
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.weight_optimizer, start_factor=0.1, total_iters=warmup_epochs
            )
            self.weight_scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.weight_optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            self.weight_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.weight_optimizer,
                milestones=list(config['weight_lr_decay_milestones']),
                gamma=float(config['weight_lr_decay_ratio'])
            )

    def _get_raw_model(self):
        """Unwrap DataParallel if needed."""
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model

    def _to_original_space(self, output):
        """Inverse-transform model output to original radiation space."""
        predict = self.scaler.inverse_transform(output)
        if self.use_log_space:
            predict = torch.expm1(predict)  # exp(x) - 1, inverse of log1p
        predict = torch.clamp(predict, min=0.)
        return predict

    def train_weight(self, inputs, loc_feature, real_val):

        self.weight_optimizer.zero_grad()
        self.model.train()

        output = self.model(inputs, loc_feature)

        real = torch.unsqueeze(real_val, dim=1)
        predict = self._to_original_space(output)

        if self.horizon_weights is not None:
            hw = self.horizon_weights.to(predict.device)
            mask = (real != 0.0).float()
            mask = mask / torch.mean(mask)
            mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
            loss = torch.mean(torch.abs(predict - real) * mask * hw)
        else:
            loss = self.loss(predict, real, 0.0)
        loss.backward(retain_graph=False)

        mae = utils.masked_mae(predict, real, 0.0).item()
        mape = utils.masked_mape(predict, real, 0.0).item()
        rmse = utils.masked_rmse(predict, real, 0.0).item()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['weight_clip_gradient'])
        self.weight_optimizer.step()
        self.weight_optimizer.zero_grad()

        return loss.item(), mae, mape, rmse, grad_norm.item()

    def get_lr(self):
        return self.weight_optimizer.param_groups[0]['lr']

    def get_physics_diagnostics(self):
        """Get diagnostics from physics module if available."""
        raw_model = self._get_raw_model()
        if hasattr(raw_model, 'physics_module') and hasattr(raw_model.physics_module, '_last_diagnostics'):
            return raw_model.physics_module._last_diagnostics
        return None

    def eval(self, inputs, loc_feature, real_val):
        self.model.eval()
        with torch.no_grad():
            output = self.model(inputs, loc_feature)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self._to_original_space(output)

        predict = torch.clamp(predict, max=self.max_value)

        loss = self.loss(predict, real, 0.0)
        mae = utils.masked_mae(predict, real, 0.0).item()
        mape = utils.masked_mape(predict, real, 0.0).item()
        rmse = utils.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def load(self, model_path):
        states = torch.load(model_path)
        raw_model = self._get_raw_model()
        raw_model.load_state_dict(states['net'])
        self.weight_optimizer.load_state_dict(states['weight_optimizer'])
        self.weight_scheduler.load_state_dict(states['weight_scheduler'])
        return states['best_epoch']

    def save(self, model_path, best_epoch):
        raw_model = self._get_raw_model()
        states = {
            'net': raw_model.state_dict(),
            'weight_optimizer': self.weight_optimizer.state_dict(),
            'weight_scheduler': self.weight_scheduler.state_dict(),
            'best_epoch': best_epoch
        }
        torch.save(obj=states, f=model_path)
