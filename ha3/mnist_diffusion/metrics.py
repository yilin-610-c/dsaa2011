from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

from .models import MNISTClassifier


@torch.no_grad()
def classifier_accuracy(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def load_classifier(path: str | Path, device) -> Tuple[MNISTClassifier, dict]:
    payload = torch.load(path, map_location=device)
    model = MNISTClassifier(feature_dim=payload.get("feature_dim", 256)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


@torch.no_grad()
def classifier_outputs(model, images: torch.Tensor, device):
    logits, feats = model(images.to(device), return_features=True)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy(), feats.detach().cpu().numpy()


@torch.no_grad()
def collect_real_features(model, loader, device):
    feats = []
    probs = []
    for x, _ in loader:
        p, f = classifier_outputs(model, x, device)
        probs.append(p)
        feats.append(f)
    return np.concatenate(probs, axis=0), np.concatenate(feats, axis=0)


def mnist_is(probs: np.ndarray, splits: int = 10) -> Tuple[float, float]:
    probs = np.clip(probs, 1e-12, 1.0)
    scores = []
    n = probs.shape[0]
    splits = min(splits, n)
    for part in np.array_split(probs, splits):
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part) - np.log(py))
        scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))
    return float(np.mean(scores)), float(np.std(scores))


def classifier_feature_fid(real_features: np.ndarray, gen_features: np.ndarray) -> float:
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(gen_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(gen_features, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)

