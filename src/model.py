import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from tqdm.notebook import tqdm


class RNNBinaryClassifier(nn.Module):
    def __init__(self, criterion, optimizer, lr, hidden_size=128, num_layers=2, tol=1e-4, max_epochs=1000, device=None):
        super(RNNBinaryClassifier, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.criterion = criterion
        self.tol = tol
        self.max_epochs = max_epochs
        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.sigmoid = nn.Sigmoid()

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.to(self.device)

    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        # RNN outputs: (batch, seq_len, hidden_size)
        x, _ = self.rnn(x)
        # Take the last timestep
        x = x[:, -1, :]  # shape: (batch, hidden_size)
        x = self.fc(x)  # shape: (batch, 1)
        x = self.sigmoid(x)
        return x

    def fit(self, X, y):
        # Convert data to tensors if needed and move to device
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        X, y = X.to(self.device), y.to(self.device)

        loss_history = []
        epoch = 0

        # Use tqdm for a progress bar
        with tqdm(total=self.max_epochs, desc="Training") as pbar:
            while epoch < self.max_epochs:
                self.optimizer.zero_grad()
                outputs = self(X)
                loss = self.criterion(outputs.squeeze(), y)
                loss.backward()
                self.optimizer.step()

                current_loss = loss.item()
                loss_history.append(current_loss)

                # Update progress bar description
                pbar.set_description(f"Epoch {epoch + 1}, Loss: {current_loss:.6f}")
                pbar.update(1)

                # Check convergence if we have enough history
                if len(loss_history) > 5:
                    recent_losses = loss_history[-5:]
                    # Check if changes in recent losses are below tolerance
                    if max(recent_losses) - min(recent_losses) < self.tol:
                        print("Convergence reached; stopping training.")
                        break

                epoch += 1

        if epoch >= self.max_epochs:
            print("Max epochs reached; stopping training.")

        # Free up memory
        del X, y, outputs, loss, loss_history, recent_losses, current_loss
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def get_params(self):
        crit_name = self.criterion.__class__.__name__
        opt_name = self.optimizer.__class__.__name__
        lr = self.optimizer.param_groups[0]['lr']
        hl, nl = self.rnn.hidden_size, self.rnn.num_layers
        return f"crit={crit_name},opt={opt_name},lr={lr},tol={self.tol},hl={hl},nl={nl}"

    @torch.no_grad()
    def predict(self, X):
        # Convert data to tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        X = X.to(self.device)

        # Forward pass
        outputs = self(X).cpu().squeeze()  # shape: (batch,)
        # Apply threshold
        predictions = (outputs > 0.5).int().numpy()
        return predictions

    def evaluate(self, X, y):
        # Simple evaluation to return accuracy
        # You can also provide your own evaluation function
        preds = self.predict(X)
        return np.mean(preds == y)
