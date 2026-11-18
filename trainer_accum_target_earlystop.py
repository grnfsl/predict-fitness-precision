from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from graph import collate_alignment_pairs


@dataclass
class PredictorTrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 20
    fitness_loss_weight: float = 1.0
    precision_loss_weight: float = 1.0
    grad_clip_norm: float = 1.0
    pin_memory: bool = True

    # Which prediction task to train.
    # Options: "both", "fitness", "precision".
    target: str = "both"

    # Bin-weighted loss.
    # If True, training batches use sample_weight from the dataset.
    # Validation remains unweighted.
    use_bin_weighted_loss: bool = False

    # Metric used for early stopping / best-checkpoint selection.
    # Options:
    #   "loss"          -> validation training objective (MSE-based)
    #   "mean_mae"      -> mean of validation fitness MAE and precision MAE
    #   "fitness_mae"   -> validation fitness MAE
    #   "precision_mae" -> validation precision MAE
    early_stopping_metric: str = "loss"
    early_stopping_patience: Optional[int] = 20
    early_stopping_min_delta: float = 0.0
    restore_best_weights: bool = True

    grad_accum_steps: int = 1


class AlignmentPredictorTrainer:
    '''
    Batched trainer for the alignment surrogate.

    Supports three training modes:
      - target="both": fitness + precision
      - target="fitness": fitness only
      - target="precision": precision only

    Uses MSE loss for the active target(s).
    '''

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        config: Optional[PredictorTrainConfig] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.cfg = config or PredictorTrainConfig()

        self.cfg.target = str(self.cfg.target).lower()
        if self.cfg.target not in {"both", "fitness", "precision"}:
            raise ValueError(
                "target must be 'both', 'fitness', or 'precision'"
            )

        self.cfg.early_stopping_metric = str(
            self.cfg.early_stopping_metric
        ).lower()

        valid_early_stopping_metrics = {
            "loss",
            "mean_mae",
            "fitness_mae",
            "precision_mae",
        }

        if self.cfg.early_stopping_metric not in valid_early_stopping_metrics:
            raise ValueError(
                "early_stopping_metric must be one of: "
                "'loss', 'mean_mae', 'fitness_mae', 'precision_mae'"
            )

        if (
            self.cfg.early_stopping_metric == "mean_mae"
            and self.cfg.target != "both"
        ):
            raise ValueError(
                "early_stopping_metric='mean_mae' requires target='both'."
            )

        if (
            self.cfg.early_stopping_metric == "fitness_mae"
            and self.cfg.target == "precision"
        ):
            raise ValueError(
                "early_stopping_metric='fitness_mae' requires fitness "
                "to be an active target."
            )

        if (
            self.cfg.early_stopping_metric == "precision_mae"
            and self.cfg.target == "fitness"
        ):
            raise ValueError(
                "early_stopping_metric='precision_mae' requires precision "
                "to be an active target."
            )

        # Prevent accidentally training a model/head configuration that does not
        # match the trainer configuration.
        if hasattr(self.model, "target") and self.model.target != self.cfg.target:
            raise ValueError(
                f"Model target={self.model.target!r} but "
                f"trainer target={self.cfg.target!r}"
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=int(self.cfg.batch_size),
            shuffle=shuffle,
            num_workers=int(self.cfg.num_workers),
            pin_memory=bool(self.cfg.pin_memory),
            collate_fn=collate_alignment_pairs,
        )

    def _unpack_batch(self, batch):
        '''
        Supports both old batches:
            (g_batch, y_batch)
        and weighted-loss batches:
            (g_batch, y_batch, sample_weight)
        '''
        if len(batch) == 2:
            g_batch, y_batch = batch
            sample_weight = None
        elif len(batch) == 3:
            g_batch, y_batch, sample_weight = batch
        else:
            raise ValueError("Expected batch to contain 2 or 3 items.")

        return g_batch, y_batch, sample_weight

    def _run_one_epoch(self, loader: DataLoader, train: bool) -> Dict[str, Optional[float]]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_fit_loss = 0.0
        total_prec_loss = 0.0
        total_abs_fit = 0.0
        total_abs_prec = 0.0
        total_count = 0

        accum_steps = max(1, int(self.cfg.grad_accum_steps))

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        num_batches = len(loader)

        for batch_idx, batch in enumerate(loader):
            g_batch, y_batch, sample_weight = self._unpack_batch(batch)

            g_batch = g_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            if sample_weight is not None:
                sample_weight = sample_weight.to(self.device).view(-1)

            with torch.set_grad_enabled(train):
                out = self.model(g_batch)

                fit_true = y_batch[:, 0]
                prec_true = y_batch[:, 1]

                fit_pred = out.get("fitness")
                prec_pred = out.get("precision")

                fit_loss_each = None
                prec_loss_each = None

                # Per-sample MSE for the active target(s).
                if self.cfg.target in {"both", "fitness"}:
                    fit_loss_each = (fit_pred - fit_true) ** 2

                if self.cfg.target in {"both", "precision"}:
                    prec_loss_each = (prec_pred - prec_true) ** 2

                if self.cfg.target == "both":
                    sample_loss_each = (
                        float(self.cfg.fitness_loss_weight) * fit_loss_each
                        + float(self.cfg.precision_loss_weight) * prec_loss_each
                    )
                elif self.cfg.target == "fitness":
                    sample_loss_each = fit_loss_each
                else:
                    sample_loss_each = prec_loss_each

                # Weighted loss is used only for training.
                # Validation remains normal unweighted MSE.
                if train and bool(self.cfg.use_bin_weighted_loss):
                    if sample_weight is None:
                        sample_weight = torch.ones_like(sample_loss_each)

                    loss = (
                        sample_weight * sample_loss_each
                    ).sum() / sample_weight.sum().clamp_min(1e-8)
                else:
                    loss = sample_loss_each.mean()

                # Keep per-target statistics unweighted so train/valid numbers
                # remain easy to compare.
                fit_loss = (
                    fit_loss_each.mean()
                    if fit_loss_each is not None
                    else None
                )
                prec_loss = (
                    prec_loss_each.mean()
                    if prec_loss_each is not None
                    else None
                )

                if train:
                    loss_to_backprop = loss / accum_steps
                    loss_to_backprop.backward()

                    should_step = (
                        ((batch_idx + 1) % accum_steps == 0)
                        or (batch_idx + 1 == num_batches)
                    )

                    if should_step:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=float(self.cfg.grad_clip_norm),
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

            bsz = int(y_batch.size(0))
            total_count += bsz
            total_loss += float(loss.item()) * bsz

            if fit_loss is not None:
                total_fit_loss += float(fit_loss.item()) * bsz
                total_abs_fit += (
                    float((fit_pred - fit_true).abs().mean().item()) * bsz
                )

            if prec_loss is not None:
                total_prec_loss += float(prec_loss.item()) * bsz
                total_abs_prec += (
                    float((prec_pred - prec_true).abs().mean().item()) * bsz
                )

        denom = max(1, total_count)

        return {
            "loss": total_loss / denom,
            "fit_loss": (
                total_fit_loss / denom
                if self.cfg.target in {"both", "fitness"}
                else None
            ),
            "prec_loss": (
                total_prec_loss / denom
                if self.cfg.target in {"both", "precision"}
                else None
            ),
            "fit_mae": (
                total_abs_fit / denom
                if self.cfg.target in {"both", "fitness"}
                else None
            ),
            "prec_mae": (
                total_abs_prec / denom
                if self.cfg.target in {"both", "precision"}
                else None
            ),
        }

    def _early_stopping_value(
        self,
        valid_stats: Dict[str, Optional[float]],
    ) -> float:
        metric = self.cfg.early_stopping_metric

        if metric == "loss":
            return float(valid_stats["loss"])

        if metric == "mean_mae":
            return 0.5 * (
                float(valid_stats["fit_mae"])
                + float(valid_stats["prec_mae"])
            )

        if metric == "fitness_mae":
            return float(valid_stats["fit_mae"])

        if metric == "precision_mae":
            return float(valid_stats["prec_mae"])

        raise RuntimeError(
            f"Unsupported early_stopping_metric: {metric}"
        )

    def fit(self, train_dataset, valid_dataset=None) -> Dict[str, list]:
        train_loader = self._make_loader(train_dataset, shuffle=True)
        valid_loader = (
            self._make_loader(valid_dataset, shuffle=False)
            if valid_dataset is not None
            else None
        )

        hist = {
            "train_loss": [],
            "train_fit_loss": [],
            "train_prec_loss": [],
            "train_fit_mae": [],
            "train_prec_mae": [],
            "valid_loss": [],
            "valid_fit_loss": [],
            "valid_prec_loss": [],
            "valid_fit_mae": [],
            "valid_prec_mae": [],
            "models": [],
            "best_epoch": None,
            "best_valid_loss": None,
            "best_early_stopping_metric": self.cfg.early_stopping_metric,
            "best_early_stopping_value": None,
            "best_model": None,
            "stopped_early": False,
        }

        best_valid_loss = float("inf")
        best_early_stopping_value = float("inf")
        best_epoch = None
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(int(self.cfg.epochs)):
            train_stats = self._run_one_epoch(train_loader, train=True)
            hist["train_loss"].append(train_stats["loss"])

            if self.cfg.target in {"both", "fitness"}:
                hist["train_fit_loss"].append(train_stats["fit_loss"])
                hist["train_fit_mae"].append(train_stats["fit_mae"])

            if self.cfg.target in {"both", "precision"}:
                hist["train_prec_loss"].append(train_stats["prec_loss"])
                hist["train_prec_mae"].append(train_stats["prec_mae"])

            if valid_loader is not None:
                valid_stats = self._run_one_epoch(valid_loader, train=False)
                hist["valid_loss"].append(valid_stats["loss"])

                if self.cfg.target in {"both", "fitness"}:
                    hist["valid_fit_loss"].append(valid_stats["fit_loss"])
                    hist["valid_fit_mae"].append(valid_stats["fit_mae"])

                if self.cfg.target in {"both", "precision"}:
                    hist["valid_prec_loss"].append(valid_stats["prec_loss"])
                    hist["valid_prec_mae"].append(valid_stats["prec_mae"])

                current_valid_loss = float(valid_stats["loss"])
                current_early_stopping_value = self._early_stopping_value(
                    valid_stats
                )

                improved = current_early_stopping_value < (
                    best_early_stopping_value
                    - float(self.cfg.early_stopping_min_delta)
                )

                if improved:
                    best_early_stopping_value = current_early_stopping_value
                    best_valid_loss = current_valid_loss
                    best_epoch = epoch + 1

                    best_model_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }

                    hist["best_epoch"] = best_epoch
                    hist["best_valid_loss"] = best_valid_loss
                    hist["best_early_stopping_value"] = (
                        best_early_stopping_value
                    )
                    hist["best_model"] = {
                        "epoch": best_epoch,
                        "state_dict": best_model_state,
                    }

                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            msg = (
                f"Epoch {epoch + 1}/{self.cfg.epochs} | "
                f"train loss={train_stats['loss']:.4f}"
            )

            if self.cfg.target in {"both", "fitness"}:
                msg += (
                    f", fit_loss={train_stats['fit_loss']:.4f}, "
                    f"fit_mae={train_stats['fit_mae']:.4f}"
                )

            if self.cfg.target in {"both", "precision"}:
                msg += (
                    f", prec_loss={train_stats['prec_loss']:.4f}, "
                    f"prec_mae={train_stats['prec_mae']:.4f}"
                )

            if valid_loader is not None:
                msg += f" | valid loss={valid_stats['loss']:.4f}"

                if self.cfg.target in {"both", "fitness"}:
                    msg += (
                        f", fit_loss={valid_stats['fit_loss']:.4f}, "
                        f"fit_mae={valid_stats['fit_mae']:.4f}"
                    )

                if self.cfg.target in {"both", "precision"}:
                    msg += (
                        f", prec_loss={valid_stats['prec_loss']:.4f}, "
                        f"prec_mae={valid_stats['prec_mae']:.4f}"
                    )

            # Keep your existing per-epoch model snapshots.
            hist["models"].append({
                "epoch": epoch + 1,
                "state_dict": {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            })

            print(msg)

            if (
                valid_loader is not None
                and self.cfg.early_stopping_patience is not None
                and epochs_no_improve >= int(self.cfg.early_stopping_patience)
            ):
                hist["stopped_early"] = True
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"Best {self.cfg.early_stopping_metric}="
                    f"{best_early_stopping_value:.4f} "
                    f"at epoch {best_epoch} "
                    f"(valid loss={best_valid_loss:.4f})."
                )
                break

        # Restore best validation model weights.
        if (
            valid_loader is not None
            and self.cfg.restore_best_weights
            and best_model_state is not None
        ):
            self.model.load_state_dict(best_model_state, strict=True)

        return hist

    @torch.no_grad()
    def predict_loader(self, dataset):
        loader = self._make_loader(dataset, shuffle=False)
        self.model.eval()

        preds = []
        tgts = []

        for batch in loader:
            g_batch, y_batch, _sample_weight = self._unpack_batch(batch)

            g_batch = g_batch.to(self.device)
            out = self.model(g_batch)

            pred = out["pred"].detach().cpu()

            if self.cfg.target == "both":
                target = y_batch
            elif self.cfg.target == "fitness":
                target = y_batch[:, 0:1]
            else:
                target = y_batch[:, 1:2]

            preds.append(pred)
            tgts.append(target.detach().cpu())

        return torch.cat(preds, dim=0), torch.cat(tgts, dim=0)

    def save_checkpoint(self, path, epoch: int, history=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "epoch": int(epoch),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": history,
                "config": self.cfg,
            },
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.to(self.device)
        return checkpoint