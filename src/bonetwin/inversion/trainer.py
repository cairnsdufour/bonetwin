"""ELBO trainer for the latent-SDE digital twin.

Ref: Sec. III-C (Adam, FP16, cosine, <=100 epochs, early stopping on
validation ELBO, KL annealing; gradients via direct backprop through the
Euler-Maruyama unroll).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from bonetwin.inversion.ema import ParameterEMA
from bonetwin.inversion.schedule import cosine_warmup
from bonetwin.misfit.objective import elbo, kl_weight
from bonetwin.propagation.station import BoneTwin
from bonetwin.recorder.schema import MisfitConfig, OptimConfig
from bonetwin.recorder.telemetry import get_logger
from bonetwin.sensors.traces import Batch

_LOG = get_logger(__name__)


@dataclass
class TrainState:
    epoch: int
    step: int
    best_val: float


def build_targets(batch: Batch, grid: torch.Tensor) -> torch.Tensor:
    baseline = batch["baseline"].unsqueeze(-1)
    slope = (batch["delta"] / grid[-1]).unsqueeze(-1)
    return baseline + slope * grid.unsqueeze(0)


class Trainer:
    def __init__(
        self,
        model: BoneTwin,
        optim_cfg: OptimConfig,
        loss_cfg: MisfitConfig,
        device: torch.device,
        total_steps: int,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optim_cfg = optim_cfg
        self.loss_cfg = loss_cfg
        self.optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=optim_cfg.lr,
            betas=optim_cfg.betas,
            weight_decay=optim_cfg.weight_decay,
        )
        warmup = max(1, int(optim_cfg.warmup_epochs))
        self.scheduler = cosine_warmup(self.optimizer, warmup, max(total_steps, warmup + 1))
        self.ema = ParameterEMA(model, optim_cfg.ema_decay)
        self.use_amp = optim_cfg.precision == "fp16" and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.state = TrainState(epoch=0, step=0, best_val=float("inf"))

    def _loss(self, batch: Batch, beta: float) -> torch.Tensor:
        grid = self.model.grid.to(self.device)
        volume = batch["volume"].to(self.device)
        context = batch["context"].to(self.device)
        targets = build_targets(batch, grid).to(self.device)
        out = self.model(volume, context)
        terms = elbo(
            out["tscore_path"],
            targets,
            out["mean"],
            out["logvar"],
            beta,
            self.loss_cfg.free_bits,
        )
        return terms["loss"]

    def train_step(self, batch: Batch, beta: float) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=self.use_amp):
            loss = self._loss(batch, beta)
        scaled = self.scaler.scale(loss)
        assert isinstance(scaled, torch.Tensor)
        torch.autograd.backward(scaled)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.optim_cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.ema.update(self.model)
        self.state.step += 1
        return float(loss.detach().cpu())

    @torch.no_grad()
    def validate(self, loader: DataLoader[Batch], beta: float) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        for batch in loader:
            total += self._loss(batch, beta).item() * batch["volume"].shape[0]
            count += batch["volume"].shape[0]
        return total / max(count, 1)

    def fit(self, train_loader: DataLoader[Batch], val_loader: DataLoader[Batch]) -> TrainState:
        patience = 0
        for epoch in range(self.optim_cfg.epochs):
            self.state.epoch = epoch
            beta = kl_weight(epoch, self.loss_cfg.kl_warmup_epochs, self.loss_cfg.beta_kl_max)
            for batch in train_loader:
                loss = self.train_step(batch, beta)
                if self.optim_cfg.max_steps and self.state.step >= self.optim_cfg.max_steps:
                    _LOG.info("step %d loss %.4f", self.state.step, loss)
                    return self.state
            val = self.validate(val_loader, beta)
            _LOG.info("epoch %d val_elbo %.4f beta %.3f", epoch, val, beta)
            if val < self.state.best_val - 1.0e-4:
                self.state.best_val = val
                patience = 0
            else:
                patience += 1
                if patience >= self.optim_cfg.early_stop_patience:
                    _LOG.info("early stop at epoch %d", epoch)
                    break
        return self.state
