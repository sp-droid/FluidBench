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

    def _predict_step(self, X):
        return self(X)

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

    def predict(self, test_dataloader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for X, _ in test_dataloader:
                X = self._to_device(X)
                pred = self._predict_step(X)

                predictions.append(pred.cpu())
        predictions = torch.cat(predictions, dim=0)
        return predictions

    def predict_rollout(self, test_dataloader):
        self.eval()
        predictions_snaps = []

        with torch.no_grad():
            # Get scalars from all snapshots in the test dataloader
            scalars = []
            for inputs, _ in test_dataloader:
                scalars.append(inputs[0])
            N = len(scalars)

            inputs, _ = next(iter(test_dataloader)) # Only the first snapshot is used
            fields = self._to_device(inputs[1])

            for i in range(N):
                scalars_i = self._to_device(scalars[i])
                X_snap = (scalars_i, fields)
                pred = self._predict_step(X_snap)
                
                fields = pred
                predictions_snaps.append(pred.cpu())

        return predictions_snaps
