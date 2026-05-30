from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


TASKS = ("incidence", "progression")


def masked_bce_with_logits_loss(
    logits: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    pos_weights: Optional[Dict[str, torch.Tensor]] = None,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute mean BCE across available task-specific masked losses."""
    losses = []
    task_losses: Dict[str, float] = {}
    for task in TASKS:
        label = batch[f"{task}_label"]
        mask = batch[f"{task}_mask"]
        active = mask > 0
        if active.any():
            active_label = label[active]
            if label_smoothing > 0:
                active_label = active_label * (1.0 - label_smoothing) + 0.5 * label_smoothing
            pos_weight = None if pos_weights is None else pos_weights.get(task)
            loss = F.binary_cross_entropy_with_logits(
                logits[task][active],
                active_label,
                pos_weight=pos_weight,
            )
            losses.append(loss)
            task_losses[f"{task}_loss"] = float(loss.detach().cpu())
        else:
            task_losses[f"{task}_loss"] = float("nan")

    if not losses:
        device = next(iter(logits.values())).device
        return torch.zeros((), dtype=torch.float32, device=device, requires_grad=True), task_losses
    return torch.stack(losses).mean(), task_losses
