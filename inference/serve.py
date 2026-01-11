import torch
from train.train_arithmetic import TRM


def build_input(a: int, b: int, D: int) -> torch.Tensor:
    """
    Builds x = one_hot(a) + one_hot(b)
    """
    x = torch.zeros(D)
    x[a] = 1.0
    x[b] += 1.0
    return x


def run_inference(a: int, b: int) -> int:
    """
    Runs trained TRM model in inference-only mode.
    """
    # --- config (must match training) ---
    D = 32
    K = 19
    n = 6
    T = 3
    checkpoint_path = "checkpoints/trm_arithmetic.pt"

    # --- load model ---
    model = TRM(D, K)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()  # IMPORTANT: inference mode

    # --- build inputs ---
    x = build_input(a, b, D)
    y0 = torch.zeros(K)
    z0 = torch.zeros(D)

    # --- inference ---
    with torch.no_grad():  # no gradients, faster + safer
        ys = model(x, y0, z0, n=n, T=T)

    # final answer
    pred = torch.argmax(ys[-1]).item()
    return pred


if __name__ == "__main__":
    for a, b in [(3, 7), (5, 9), (2, 4), (8, 1)]:
        print(f"{a} + {b} = {run_inference(a, b)}")
