import numpy as np
from tqdm import tqdm
import torch
from torch import nn

class BaseRegressionModel(nn.Module):
    def __init__(self, loss=torch.nn.MSELoss):
        super().__init__()
        
        self.loss = loss()
        self.loss_unreduced = loss(reduction="none")
        
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    @property
    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def _to_device(self, X):
        if isinstance(X, tuple) or isinstance(X, list):
            return tuple(x.to(self.device) for x in X)
        else:
            return X.to(self.device)

    # Train the model
    def fit(self,
        train_dataloader,
        val_dataloader=None
    ):
        EPOCHS = 15
        optimizer = torch.optim.AdamW(self.parameters())
        
        # Avoid leaving an initial 0% bar above the live plot in notebooks.
        history = {"train_losses": [], "train_epochs": [], "loss_name": self.loss.__class__.__name__}
        for epoch in range(EPOCHS):
            # Training
            self.train()
            train_loss = 0

            for batch, (X, y) in enumerate(train_dataloader):
                X, y = self._to_device(X), self._to_device(y)

                # Zero parameter gradients
                optimizer.zero_grad(set_to_none=True)

                pred = self(X)
                loss = self.loss(pred, y)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            train_loss /= len(train_dataloader)
            history["train_losses"].append(train_loss)
            history["train_epochs"].append(epoch)

        return history
    
    # Validate the model
    def validate(self, val_dataloader, reduced_error=True):
        self.eval()
        if reduced_error: val_loss = 0
        else: val_loss = np.zeros_like(next(iter(val_dataloader))[1].numpy())
        with torch.no_grad():
            for X, y in val_dataloader:
                X, y = self._to_device(X), self._to_device(y)

                if self._is_rollout_batch(X, y):
                    scalars_seq, q0 = X
                    pred = self.rollout(scalars_seq, q0)
                else:
                    pred = self._predict_step(X)
                
                if reduced_error: val_loss += self.loss(pred, y).item()
                else: val_loss += self.loss_unreduced(pred, y).cpu().numpy()
        val_loss /= len(val_dataloader)
        return val_loss

    # Make predictions    
    def predict(self, test_dataloader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for X in test_dataloader:
                X = self._to_device(X)
                pred = self._predict_step(X)

                predictions.append(pred.cpu())
        predictions = torch.cat(predictions, dim=0)
        return predictions

    def predict_autoregressive(self, test_dataloader, n_snaps, progress_bar=True):
        self.eval()
        predictions_snaps = []

        with torch.no_grad():
            X = next(iter(test_dataloader)) # Only the first snapshot is used
            scalars, fields = self._to_device(X)

            progress_bar = tqdm(range(n_snaps), desc="Autoregressive prediction", unit="Snapshot", disable=not progress_bar)
            for _ in progress_bar:
                X_snap = (scalars, fields)
                pred = self._predict_step(X_snap)
                
                fields = pred
                predictions_snaps.append(pred.cpu())

        return predictions_snaps