import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import numpy as np
import random 
from datasets.arithmetic import generate_arithmetic_dataset

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# -------------------------------
# TRM Model (PyTorch)
# -------------------------------

class TRM(nn.Module):
    def __init__(self, D: int, K: int):
        super().__init__()

        self.D = D
        self.K = K

        # Shared first layer
        self.fc1 = nn.Linear(2 * D + K, D)

        # Output heads
        self.fc_z = nn.Linear(D, D)
        self.fc_y = nn.Linear(D, K)

    def forward_step(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One TRM step:
          z_next = net(x, y, z)
          y_next = net(0, y, z_next)
        """

        # ----- latent reasoning -----
        inp_z = torch.cat([x, y, z], dim=-1)
        h = torch.tanh(self.fc1(inp_z))
        z_next = self.fc_z(h)

        # ----- answer refinement -----
        x_zero = torch.zeros_like(x)
        inp_y = torch.cat([x_zero, y, z_next], dim=-1)
        h_y = torch.tanh(self.fc1(inp_y))
        y_next = self.fc_y(h_y)

        return z_next, y_next

    def forward(
        self, x: torch.Tensor, y0: torch.Tensor, z0: torch.Tensor, n: int, T: int
    ) -> List[torch.Tensor]:
        """
        Deep supervision forward:
        returns [y1, y2, ..., yT]
        """

        y = y0
        z = z0
        ys = []

        for _ in range(T):
            # latent reasoning
            for _ in range(n):
                z, _ = self.forward_step(x, y, z)

            # answer refinement
            _, y = self.forward_step(torch.zeros_like(x), y, z)
            ys.append(y)

        return ys

def train_arithmetic():
    # -------------------------------
    # Config
    # -------------------------------
    D = 32
    K = 19          # sums 0..18
    n = 6           # latent steps
    T = 3           # supervision steps
    epochs = 20
    lr = 1e-3

    device = torch.device("cpu")

    # -------------------------------
    # Data
    # -------------------------------
    dataset = generate_arithmetic_dataset(
        n_samples=500, dim=D, seed=0
    )

    # Convert dataset to tensors
    xs = torch.from_numpy(
        np.stack([x for x, _, _ in dataset], axis=0)
    ).float()
    ys_true = torch.from_numpy(
        np.stack([y[:K] for _, y, _ in dataset], axis=0)
    ).float()

    # -------------------------------
    # Model + Optimizer
    # -------------------------------
    model = TRM(D, K).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # -------------------------------
    # Training
    # -------------------------------
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0

        for i in range(len(xs)):
            x = xs[i].to(device)
            y_true = ys_true[i].argmax().to(device)

            y0 = torch.zeros(K, device=device)
            z0 = torch.zeros(D, device=device)

            y_preds = model(x, y0, z0, n=n, T=T)

            # Deep supervision loss
            loss = 0.0
            for y_pred in y_preds:
                loss = loss + criterion(y_pred.unsqueeze(0), y_true.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # accuracy from final prediction
            if torch.argmax(y_preds[-1]) == y_true:
                correct += 1

        acc = correct / len(xs)
        print(
            f"Epoch {epoch:02d} | "
            f"loss={total_loss:.3f} | acc={acc:.3f}"
        )

    # -------------------------------
    # Save trained model
    # -------------------------------
    torch.save(model.state_dict(), "checkpoints/trm_arithmetic.pt")
    print("✅ Model saved to checkpoints/trm_arithmetic.pt")


if __name__ == "__main__":
    train_arithmetic()
