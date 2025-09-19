import torch

from rejuvbridge.metrics import retrieval_metrics


def test_retrieval_beats_random() -> None:
    sim = torch.eye(5) * 2.0 + torch.ones(5, 5) * 0.1
    metrics = retrieval_metrics(sim, ks=(1, 5))
    assert metrics["R@1"] > 0.5
    assert metrics["R@5"] == 1.0
