import torch


def sanitize_theta(theta: torch.Tensor) -> torch.Tensor:
    """Numerically stabilizes CRF emissions."""
    theta_max = torch.max(theta, dim=1, keepdim=True).values
    theta_stable = theta - theta_max
    return torch.clamp(theta_stable, min=-30.0)