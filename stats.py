from __future__ import annotations

from typing import Dict, Iterable, Optional
import math
import matplotlib.pyplot as plt


def _history_len(history: Dict[str, list]) -> int:
    for v in history.values():
        if isinstance(v, list):
            return len(v)
    return 0


def _has_valid(history: Dict[str, list]) -> bool:
    return len(history.get("valid_loss", [])) > 0


def print_history_text(
    history: Dict[str, list],
    every: int = 1,
    start_epoch: int = 1,
    last_n: Optional[int] = None,
) -> None:
    """
    Print history in text form.

    Parameters
    ----------
    history:
        Trainer history dict.
    every:
        Print every `every` epochs.
    start_epoch:
        First epoch number shown in text.
    last_n:
        If set, only print the last `last_n` epochs.
    """
    n = _history_len(history)
    if n == 0:
        print("History is empty.")
        return

    has_valid = _has_valid(history)

    start_idx = 0
    if last_n is not None:
        start_idx = max(0, n - int(last_n))

    for i in range(start_idx, n):
        epoch_num = start_epoch + i
        if (epoch_num - start_epoch) % every != 0:
            continue

        msg = (
            f"Epoch {epoch_num:>4} | "
            f"train loss={history['train_loss'][i]:.4f}, "
            f"fit_loss={history['train_fit_loss'][i]:.4f}, "
            f"prec_loss={history['train_prec_loss'][i]:.4f}, "
            f"fit_mae={history['train_fit_mae'][i]:.4f}, "
            f"prec_mae={history['train_prec_mae'][i]:.4f}"
        )

        if has_valid:
            msg += (
                f" | valid loss={history['valid_loss'][i]:.4f}, "
                f"fit_loss={history['valid_fit_loss'][i]:.4f}, "
                f"prec_loss={history['valid_prec_loss'][i]:.4f}, "
                f"fit_mae={history['valid_fit_mae'][i]:.4f}, "
                f"prec_mae={history['valid_prec_mae'][i]:.4f}"
            )

        print(msg)


def print_history_summary(history: Dict[str, list]) -> None:
    """
    Print a compact summary:
      - number of epochs
      - final train metrics
      - final valid metrics (if present)
      - best valid loss epoch (if present)
    """
    n = _history_len(history)
    if n == 0:
        print("History is empty.")
        return

    has_valid = _has_valid(history)
    last = n - 1

    print(f"Epochs: {n}")
    print(
        f"Final train | "
        f"loss={history['train_loss'][last]:.4f}, "
        f"fit_loss={history['train_fit_loss'][last]:.4f}, "
        f"prec_loss={history['train_prec_loss'][last]:.4f}, "
        f"fit_mae={history['train_fit_mae'][last]:.4f}, "
        f"prec_mae={history['train_prec_mae'][last]:.4f}"
    )

    if has_valid:
        print(
            f"Final valid | "
            f"loss={history['valid_loss'][last]:.4f}, "
            f"fit_loss={history['valid_fit_loss'][last]:.4f}, "
            f"prec_loss={history['valid_prec_loss'][last]:.4f}, "
            f"fit_mae={history['valid_fit_mae'][last]:.4f}, "
            f"prec_mae={history['valid_prec_mae'][last]:.4f}"
        )

        best_idx = min(range(n), key=lambda i: history["valid_loss"][i])
        print(
            f"Best valid loss at epoch {best_idx + 1}: "
            f"{history['valid_loss'][best_idx]:.4f}"
        )


def plot_history_metric(
    history: Dict[str, list],
    train_key: str,
    valid_key: Optional[str] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    start_epoch: int = 1,
    logy: bool = False,
) -> None:
    """
    Plot one metric from history.

    Example:
        plot_history_metric(history, "train_loss", "valid_loss", title="Total Loss")
    """
    n = _history_len(history)
    if n == 0:
        raise ValueError("History is empty.")

    epochs = list(range(start_epoch, start_epoch + n))

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, history[train_key], label=train_key)

    if valid_key is not None and valid_key in history and len(history[valid_key]) > 0:
        plt.plot(epochs, history[valid_key], label=valid_key)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel or train_key)
    plt.title(title or train_key)
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_history_all(history: Dict[str, list], start_epoch: int = 1) -> None:
    """
    Plot the main 3 panels:
      1) total loss
      2) fitness / precision losses
      3) fitness / precision MAE
    """
    n = _history_len(history)
    if n == 0:
        raise ValueError("History is empty.")

    epochs = list(range(start_epoch, start_epoch + n))
    has_valid = _has_valid(history)

    plt.figure(figsize=(9, 4.8))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    if has_valid:
        plt.plot(epochs, history["valid_loss"], label="valid_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4.8))
    plt.plot(epochs, history["train_fit_loss"], label="train_fit_loss")
    plt.plot(epochs, history["train_prec_loss"], label="train_prec_loss")
    if has_valid:
        plt.plot(epochs, history["valid_fit_loss"], label="valid_fit_loss")
        plt.plot(epochs, history["valid_prec_loss"], label="valid_prec_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fitness / Precision Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4.8))
    plt.plot(epochs, history["train_fit_mae"], label="train_fit_mae")
    plt.plot(epochs, history["train_prec_mae"], label="train_prec_mae")
    if has_valid:
        plt.plot(epochs, history["valid_fit_mae"], label="valid_fit_mae")
        plt.plot(epochs, history["valid_prec_mae"], label="valid_prec_mae")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Fitness / Precision MAE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_histories(
    histories: Dict[str, Dict[str, list]],
    metric_key: str = "train_loss",
    title: Optional[str] = None,
    start_epoch: int = 1,
) -> None:
    """
    Compare the same metric across multiple runs.

    Example:
        compare_histories(
            {"A": hist_a, "C": hist_c},
            metric_key="valid_loss",
            title="Variant Comparison"
        )
    """
    plt.figure(figsize=(8, 4.5))

    for name, hist in histories.items():
        if metric_key not in hist:
            continue
        if not isinstance(hist[metric_key], list):
            continue
        if len(hist[metric_key]) == 0:
            continue

        n = len(hist[metric_key])
        epochs = list(range(start_epoch, start_epoch + n))
        plt.plot(epochs, hist[metric_key], label=name)

    plt.xlabel("Epoch")
    plt.ylabel(metric_key)
    plt.title(title or metric_key)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def best_epoch_info(
    history: Dict[str, list],
    key: str = "valid_loss",
    mode: str = "min",
) -> Dict[str, float]:
    """
    Return metrics at the best epoch according to `key`.

    Parameters
    ----------
    history:
        Trainer history dict.
    key:
        Metric used to choose the best epoch.
        Common choices: "valid_loss", "train_loss", "valid_fit_mae", ...
    mode:
        "min" for metrics where lower is better,
        "max" for metrics where higher is better.

    Returns
    -------
    dict with:
        - epoch
        - selector_key
        - selector_value
        - all available metrics at that epoch
    """
    n = _history_len(history)
    if n == 0:
        raise ValueError("History is empty.")
    if key not in history or len(history[key]) == 0:
        raise ValueError(f"History does not contain non-empty key: {key!r}")

    values = history[key]
    if mode == "min":
        best_idx = min(range(n), key=lambda i: values[i])
    elif mode == "max":
        best_idx = max(range(n), key=lambda i: values[i])
    else:
        raise ValueError("mode must be 'min' or 'max'")

    out = {
        "epoch": best_idx + 1,
        "selector_key": key,
        "selector_value": history[key][best_idx],
    }

    for k, v in history.items():
        if isinstance(v, list) and len(v) == n:
            out[k] = v[best_idx]

    return out


def print_best_epoch(
    history: Dict[str, list],
    key: str = "valid_loss",
    mode: str = "min",
) -> None:
    """
    Print a compact summary of the best epoch according to `key`.
    """
    info = best_epoch_info(history, key=key, mode=mode)

    epoch = info["epoch"]
    selector_key = info["selector_key"]
    selector_value = info["selector_value"]

    print(f"Best epoch by {selector_key} ({mode}) = epoch {epoch}")
    print(f"  {selector_key} = {selector_value:.4f}")

    keys_in_order = [
        "train_loss",
        "train_fit_loss",
        "train_prec_loss",
        "train_fit_mae",
        "train_prec_mae",
        "valid_loss",
        "valid_fit_loss",
        "valid_prec_loss",
        "valid_fit_mae",
        "valid_prec_mae",
    ]

    for k in keys_in_order:
        if k in info:
            print(f"  {k} = {info[k]:.4f}")


def plot_train_valid_loss_with_best_epoch(
    history: Dict[str, list],
    start_epoch: int = 1,
    key: str = "valid_loss",
) -> None:
    """
    Plot train/valid loss and mark the best validation epoch.
    """
    if "train_loss" not in history or len(history["train_loss"]) == 0:
        raise ValueError("History does not contain train_loss.")

    n = len(history["train_loss"])
    epochs = list(range(start_epoch, start_epoch + n))

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")

    has_valid = "valid_loss" in history and len(history["valid_loss"]) == n
    if has_valid:
        plt.plot(epochs, history["valid_loss"], label="valid_loss")

        best_idx = min(range(n), key=lambda i: history[key][i])
        best_epoch = start_epoch + best_idx
        best_value = history[key][best_idx]

        plt.axvline(best_epoch, linestyle="--", label=f"best epoch = {best_epoch}")
        plt.scatter([best_epoch], [best_value], zorder=5, label=f"best {key} = {best_value:.4f}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Valid Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()