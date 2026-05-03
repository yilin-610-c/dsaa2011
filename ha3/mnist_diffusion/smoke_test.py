import torch

from .diffusion import build_schedule, ddim_sample, ddpm_sample, q_sample
from .models import MNISTClassifier, MNISTUNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTUNet(base_channels=16).to(device)
    sched = build_schedule("scaled_linear", 20, device=device)
    x0 = torch.randn(4, 1, 32, 32, device=device).clamp(-1, 1)
    noise = torch.randn_like(x0)
    t = torch.randint(0, 20, (4,), device=device)
    xt = q_sample(x0, t, noise, sched)
    pred = model(xt, t)
    assert pred.shape == x0.shape
    ddpm = ddpm_sample(model, sched, (2, 1, 32, 32), device)
    ddim = ddim_sample(model, sched, (2, 1, 32, 32), device, sampling_steps=5)
    assert ddpm.shape == (2, 1, 32, 32)
    assert ddim.shape == (2, 1, 32, 32)
    clf = MNISTClassifier().to(device)
    logits, feats = clf(x0, return_features=True)
    assert logits.shape == (4, 10)
    assert feats.shape[0] == 4
    print("smoke test passed")


if __name__ == "__main__":
    main()

