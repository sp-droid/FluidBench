from tqdm import tqdm
import torch
from torch import nn

class BaseRegressionModel(nn.Module):
    def __init__(self,
        loss=torch.nn.MSELoss
    ):
        super().__init__()

        self.loss = loss()
        self.loss_unreduced = loss(reduction="none")

        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def _to_device(self, X):
        if isinstance(X, tuple) or isinstance(X, list):
            return tuple(x.to(self.device) for x in X)
        else:
            return X.to(self.device)

    # Train the model: each sample maps snapshot i to snapshot i + 1
    def fit(self,
        train_dataloader,
        val_dataloader,
        optimizer=lambda model: torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-3,
            betas=(0.90, 0.999),
            eps=1e-8
        ),
        epochs: int = 300,
    ):
        optimizer = optimizer(self)

        history = {"train_losses": [], "val_losses": [], "train_epochs": [], "loss_name": self.loss.__class__.__name__}
        progress = tqdm(range(epochs), desc="Training")
        for epoch in progress:
            # Training
            self.train()
            train_loss = 0

            for X, y in train_dataloader:
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

            # Validation
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for X, y in val_dataloader:
                    X, y = self._to_device(X), self._to_device(y)
                    val_loss += self.loss(self(X), y).item()
            val_loss /= len(val_dataloader)
            history["val_losses"].append(val_loss)

            progress.set_postfix(train=f"{train_loss:.3e}", val=f"{val_loss:.3e}")

        return history

    # Predict the next snapshot from each true snapshot
    def predict(self, test_dataloader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for X, _ in test_dataloader:
                X = self._to_device(X)
                pred = self(X)

                predictions.append(pred.cpu())
        predictions = torch.cat(predictions, dim=0)
        return predictions

    # Same as predict, but starting from the first snapshot and feeding each prediction back in
    def predict_rollout(self, test_dataloader):
        self.eval()
        predictions_snaps = []
        fields = None

        with torch.no_grad():
            for (scalars, inputs), _ in test_dataloader:
                if fields is None:
                    fields = self._to_device(inputs) # Only the first snapshot is used
                fields = self((self._to_device(scalars), fields))
                predictions_snaps.append(fields.cpu())

        return predictions_snaps
