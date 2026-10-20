import numpy as np
import torch
from torch import nn

from .minimal_base import BaseRegressionModel

class FieldMLP(BaseRegressionModel):
    """Feed-forward network on the whole flattened snapshot (works on any mesh, it ignores connectivity).

    Inputs and outputs are normalised per variable; the network predicts the increment to the next
    snapshot (residual update), trained full-batch with Gaussian input noise for stable rollouts.
    """
    def __init__(self, shape, n_scalars=2, hidden=(512, 256, 512), epochs=3000, lr=1e-3, noise=0.01):
        super().__init__()
        self.shape = tuple(shape)  # (channels, *spatial)
        self.epochs, self.lr, self.noise = epochs, lr, noise
        size = int(np.prod(shape))

        layers, width = [], size + n_scalars
        for h in hidden:
            layers += [nn.Linear(width, h), nn.GELU()]
            width = h
        layers.append(nn.Linear(width, size))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)  # start as persistence
        nn.init.zeros_(self.net[-1].bias)

        channels = shape[0]
        for name in ("mean", "std", "delta_std"):
            self.register_buffer(name, torch.ones(channels))
        self.register_buffer("scalar_mean", torch.zeros(n_scalars))
        self.register_buffer("scalar_std", torch.ones(n_scalars))

    def _view(self, stat):
        return stat.view(1, -1, *[1] * (len(self.shape) - 1))

    def _increment(self, scalars, fields, noise=0.0):
        """Normalised increment predicted for raw inputs."""
        x = (fields - self._view(self.mean)) / self._view(self.std)
        if noise:
            x = x + noise * torch.randn_like(x)
        s = (scalars - self.scalar_mean) / self.scalar_std
        return self.net(torch.cat([x.flatten(1), s], dim=1)).view(len(fields), *self.shape)

    def forward(self, X):
        scalars, fields = X
        return fields + self._increment(scalars, fields) * self._view(self.delta_std)

    def fit(self, train_dataloader, val_dataloader):
        def gather(loader):
            scalars, fields, targets = zip(*[(X[0], X[1], y) for X, y in loader])
            return torch.cat(scalars).float().to(self.device), torch.cat(fields).float().to(self.device), torch.cat(targets).float().to(self.device)
        train, val = gather(train_dataloader), gather(val_dataloader)

        # Per-variable statistics of the training snapshots, their increments and the scalars (constant ones keep std 1).
        scalars, fields, targets = train
        dims = [0] + list(range(2, fields.ndim))
        self.mean, self.std = fields.mean(dim=dims), fields.std(dim=dims)
        self.delta_std = (targets - fields).std(dim=dims)
        self.scalar_mean = scalars.mean(dim=0)
        self.scalar_std = torch.where(scalars.std(dim=0) > 1e-6, scalars.std(dim=0), torch.ones_like(self.scalar_mean))

        def loss_on(data, noise=0.0):
            scalars, fields, targets = data
            target = (targets - fields) / self._view(self.delta_std)
            return self.loss(self._increment(scalars, fields, noise), target)

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.epochs)
        history = {"train_losses": [], "val_losses": [], "train_epochs": [], "loss_name": "MSELoss (normalised increment)"}
        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_on(train, self.noise)
            loss.backward()
            optimizer.step()
            scheduler.step()

            self.eval()
            with torch.no_grad():
                val_loss = loss_on(val).item()
            history["train_losses"].append(loss.item())
            history["val_losses"].append(val_loss)
            history["train_epochs"].append(epoch)
        return history
