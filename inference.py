import importlib
from pathlib import Path
from collections import Counter

import pandas as pd
import torch
import dgl

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer

import gc
import statistics
import time

# import graph as graph_module
#
# from graph import (
#     prepare_pair_specs,
#     prepare_pair_spec,
#     AlignmentPairDataset,
#     collate_alignment_pairs,
#     build_alignment_pair_graph
# )

GRAPH_MODULE = "graph_pgnn_attention"   # <-- change only this

graph_module = importlib.import_module(GRAPH_MODULE)

prepare_pair_specs = graph_module.prepare_pair_specs
prepare_pair_spec = graph_module.prepare_pair_spec
AlignmentPairDataset = graph_module.AlignmentPairDataset
collate_alignment_pairs = graph_module.collate_alignment_pairs
build_alignment_pair_graph = graph_module.build_alignment_pair_graph

from torch.utils.data import DataLoader

from lazy_dataset import LazyAlignmentPairDataset


def read_log_as_variants_no_cache(log_path, activity_key="concept:name"):
    log = xes_importer.apply(
        str(log_path),
        parameters={
            xes_importer.Variants.ITERPARSE.value.Parameters.SHOW_PROGRESS_BAR: False
        },
    )

    counter = Counter()

    for trace in log:
        seq = tuple(str(event[activity_key]) for event in trace)
        counter[seq] += 1

    return [(list(seq), count) for seq, count in counter.items()]


def load_model_generic(model_path, FitnessPrecisionPredictor, model_config = None, target: str = 'both', ):

    if model_config is None:
        raise ValueError('model_config is missing')

    model = FitnessPrecisionPredictor(**model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    model.load_state_dict(
        # checkpoint["policy_state_dict"].state_dict(),
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


def load_pair_for_inference(log_path, model_path):
    log_variants = read_log_as_variants_no_cache(log_path)

    net, im, fm = pnml_importer.apply(str(model_path))

    return log_variants, net, im, fm


def extract_prediction_tensor(model_output):
    """
    Your model may return either:
        tensor of shape [B, 2]
    or:
        dict containing the prediction tensor.

    This function extracts the final [B, 2] tensor.
    """

    if torch.is_tensor(model_output):
        pred = model_output

    elif isinstance(model_output, dict):
        # First try common prediction keys
        possible_keys = [
            "pred",
            "prediction",
            "y_pred",
            "y_hat",
            "out",
            "output",
            "scores",
            "logits",
        ]

        pred = None

        for key in possible_keys:
            if key in model_output and torch.is_tensor(model_output[key]):
                pred = model_output[key]
                break

        # If not found, try to find any tensor whose last dimension is 2
        if pred is None:
            for key, value in model_output.items():
                if torch.is_tensor(value) and value.ndim >= 2 and value.shape[-1] == 2:
                    pred = value
                    break

        # If model returns separate fitness and precision tensors
        if pred is None:
            fitness_keys = ["fitness", "fit", "fitness_pred", "pred_fitness"]
            precision_keys = ["precision", "prec", "precision_pred", "pred_precision"]

            fitness = None
            precision = None

            for key in fitness_keys:
                if key in model_output and torch.is_tensor(model_output[key]):
                    fitness = model_output[key]
                    break

            for key in precision_keys:
                if key in model_output and torch.is_tensor(model_output[key]):
                    precision = model_output[key]
                    break

            if fitness is not None and precision is not None:
                fitness = fitness.view(fitness.shape[0], -1) if fitness.ndim > 0 else fitness.view(1, 1)
                precision = precision.view(precision.shape[0], -1) if precision.ndim > 0 else precision.view(1, 1)

                fitness = fitness[:, 0]
                precision = precision[:, 0]

                pred = torch.stack([fitness, precision], dim=-1)

        if pred is None:
            raise KeyError(
                "Could not find prediction tensor in model output dict. "
                f"Available keys: {list(model_output.keys())}"
            )

    else:
        raise TypeError(f"Unexpected model output type: {type(model_output)}")

    # Normalize shape to [B, 2]
    if pred.ndim == 1:
        if pred.numel() == 2:
            pred = pred.unsqueeze(0)
        else:
            raise ValueError(f"Prediction tensor has unexpected shape: {tuple(pred.shape)}")

    if pred.ndim > 2:
        pred = pred.view(pred.shape[0], -1)

    if pred.shape[-1] != 2:
        raise ValueError(
            f"Expected prediction tensor with last dimension 2, got shape {tuple(pred.shape)}"
        )

    return pred


def model_output_to_tensor(output):
    """
    Converts model output to a [batch_size, 2] tensor.

    Supports:
      - direct tensor output
      - dict output with common prediction keys
      - dict output with separate fitness/precision keys
    """

    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        # Case 1: model returns {"pred": tensor} or similar
        for key in ["pred", "prediction", "predictions", "y_pred", "out", "output"]:
            if key in output and torch.is_tensor(output[key]):
                return output[key]

        # Case 2: model returns {"fitness": tensor, "precision": tensor}
        fitness_keys = ["fitness", "fit", "fitness_pred", "pred_fitness"]
        precision_keys = ["precision", "prec", "precision_pred", "pred_precision"]

        fit_key = next((k for k in fitness_keys if k in output), None)
        prec_key = next((k for k in precision_keys if k in output), None)

        if fit_key is not None and prec_key is not None:
            fit = output[fit_key]
            prec = output[prec_key]

            if not torch.is_tensor(fit):
                fit = torch.tensor(fit)

            if not torch.is_tensor(prec):
                prec = torch.tensor(prec)

            fit = fit.reshape(-1, 1)
            prec = prec.reshape(-1, 1)

            return torch.cat([fit, prec], dim=1)

        raise KeyError(
            f"Could not find prediction tensor in model output dict. "
            f"Available keys: {list(output.keys())}"
        )

    if isinstance(output, (tuple, list)):
        if len(output) > 0 and torch.is_tensor(output[0]):
            return output[0]

    raise TypeError(f"Unsupported model output type: {type(output)}")


def predict_single_pair(
    model,
    log_path,
    model_path,
    variant="A",
    max_local_onehot_dim=None,
    trace_freq_mode="normalized",
):
    raw_pair = load_pair_for_inference(log_path, model_path)

    # Dummy target. Needed only because AlignmentPairDataset returns (graph, y).
    dummy_target = (0.0, 0.0)

    specs = prepare_pair_specs(
        [raw_pair],
        precomputed_targets=[dummy_target],
        compute_targets_if_missing=False,
    )

    ds = AlignmentPairDataset(
        specs,
        variant=variant,
        max_local_onehot_dim=max_local_onehot_dim,
        random_permute_onehot=False,
        trace_freq_mode=trace_freq_mode,
        prebuild_graphs=True,
    )

    g, _ = ds[0]

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        model_output = model(dgl.batch([g]).to(device))

    pred = extract_prediction_tensor(model_output)
    pred = pred.cpu().squeeze(0)

    return {
        "fitness": float(pred[0]),
        "precision": float(pred[1]),
    }


def predict_many_pairs(
    model,
    pairs,
    variant="A",
    max_local_onehot_dim=None,
    trace_freq_mode="normalized",
    batch_size=32,
):
    """
    pairs should be:
        [
            ("path/to/log1.xes", "path/to/model1.pnml"),
            ("path/to/log2.xes", "path/to/model2.pnml"),
            ...
        ]
    """

    raw_pairs = []
    names = []

    for log_path, model_path in pairs:
        raw_pair = load_pair_for_inference(log_path, model_path)
        raw_pairs.append(raw_pair)
        names.append(Path(model_path).stem)

    dummy_targets = [(0.0, 0.0) for _ in raw_pairs]

    specs = prepare_pair_specs(
        raw_pairs,
        precomputed_targets=dummy_targets,
        compute_targets_if_missing=False,
    )

    ds = AlignmentPairDataset(
        specs,
        variant=variant,
        max_local_onehot_dim=max_local_onehot_dim,
        random_permute_onehot=False,
        trace_freq_mode=trace_freq_mode,
        prebuild_graphs=True,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_alignment_pairs,
    )

    device = next(model.parameters()).device
    model.eval()

    predictions = []

    with torch.no_grad():
        start = 0

        for g_batch, _ in loader:
            g_batch = g_batch.to(device)

            model_output = model(g_batch)
            y_pred = extract_prediction_tensor(model_output).cpu()

            for j in range(y_pred.shape[0]):
                predictions.append({
                    "name": names[start + j],
                    "fitness": float(y_pred[j, 0]),
                    "precision": float(y_pred[j, 1]),
                })

            start += y_pred.shape[0]

    return predictions


def predict_records(
    model,
    records,
    variant="A",
    max_local_onehot_dim=None,
    trace_freq_mode="normalized",
    batch_size=32,
    num_workers=0,
    target: str='both'
):
    """
    records format:
        [
            {
                "name": "...",
                "log_path": Path("...xes"),
                "model_path": Path("...pnml"),
                "fitness": 0.98,
                "precision": 0.21,
            },
            ...
        ]

    Returns a pandas DataFrame with:
        name, log_path, model_path,
        predicted_fitness, true_fitness, fitness_error,
        predicted_precision, true_precision, precision_error
    """

    ds = LazyAlignmentPairDataset(
        records,
        variant=variant,
        max_local_onehot_dim=max_local_onehot_dim,
        random_permute_onehot=False,
        trace_freq_mode=trace_freq_mode,
        variant_cache_dir=".cache_variants",
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,          # important: keeps same order as records
        num_workers=num_workers,
        collate_fn=collate_alignment_pairs,
    )

    device = next(model.parameters()).device
    model.eval()

    rows = []
    start_idx = 0

    with torch.no_grad():
        for g_batch, y_batch, weight in loader:
            batch_size_actual = y_batch.shape[0]
            batch_records = records[start_idx:start_idx + batch_size_actual]

            g_batch = g_batch.to(device)

            output = model(g_batch)
            preds = model_output_to_tensor(output).cpu()

            y_batch = y_batch.cpu()

            for i, record in enumerate(batch_records):
                if target == 'fitness':
                    pred_fitness = float(preds[i, 0])
                    true_fitness = float(y_batch[i, 0])
                    rows.append({
                        "name": record["name"],
                        "log_path": str(record["log_path"]),
                        "model_path": str(record["model_path"]),

                        "predicted_fitness": pred_fitness,
                        "true_fitness": true_fitness,
                        "fitness_error": pred_fitness - true_fitness,
                        "fitness_abs_error": abs(pred_fitness - true_fitness),
                    })
                elif target == 'precision':
                    pred_precision = float(preds[i, 0])
                    true_precision = float(y_batch[i, 0])

                    rows.append({
                        "name": record["name"],
                        "log_path": str(record["log_path"]),
                        "model_path": str(record["model_path"]),

                        "predicted_precision": pred_precision,
                        "true_precision": true_precision,
                        "precision_error": pred_precision - true_precision,
                        "precision_abs_error": abs(pred_precision - true_precision),
                    })
                else:
                    pred_fitness = float(preds[i, 0])
                    pred_precision = float(preds[i, 1])

                    true_fitness = float(y_batch[i, 0])
                    true_precision = float(y_batch[i, 1])

                    rows.append({
                        "name": record["name"],
                        "log_path": str(record["log_path"]),
                        "model_path": str(record["model_path"]),

                        "predicted_fitness": pred_fitness,
                        "true_fitness": true_fitness,
                        "fitness_error": pred_fitness - true_fitness,
                        "fitness_abs_error": abs(pred_fitness - true_fitness),

                        "predicted_precision": pred_precision,
                        "true_precision": true_precision,
                        "precision_error": pred_precision - true_precision,
                        "precision_abs_error": abs(pred_precision - true_precision),
                    })

            start_idx += batch_size_actual

    return pd.DataFrame(rows)


def load_pair_to_memory(
    log_path,
    model_path,
):
    """
    Load one event log and one Petri net into memory.

    IMPORTANT:
    This loading time is NOT part of inference timing.
    """
    log = xes_importer.apply(
        str(log_path),
        parameters={
            xes_importer.Variants.ITERPARSE.value.Parameters.SHOW_PROGRESS_BAR: False
        },
    )

    net, im, fm = pnml_importer.apply(str(model_path))

    return log, net, im, fm


def _loaded_log_to_variants(
    log,
    activity_key="concept:name",
):
    """
    Convert an already-loaded PM4Py EventLog to trace variants.

    This IS part of inference preprocessing and therefore is timed.
    """
    counter = Counter()

    for trace in log:
        seq = tuple(
            str(event[activity_key])
            for event in trace
        )
        counter[seq] += 1

    return [
        (list(seq), count)
        for seq, count in counter.items()
    ]


def _synchronize_device(device):
    """
    Required for correct CUDA timing because GPU operations are asynchronous.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear_behavior_cache(graph_module):
    """
    Clear model-behavior caches before a timed repetition.
    """
    for name, value in vars(graph_module).items():
        if (
            name.startswith("_MODEL_")
            and name.endswith("_CACHE")
            and hasattr(value, "clear")
        ):
            value.clear()


def _build_loaded_pair_graph(
    log,
    net,
    im,
    fm,
    graph_module,
    variant="A",
    max_local_onehot_dim=None,
    trace_freq_mode="normalized",
):
    """
    Full preprocessing from already-loaded log/model objects to DGL graph.
    """

    # --------------------------------------------------------
    # 1. Compress loaded event log into trace variants
    # --------------------------------------------------------
    variants = _loaded_log_to_variants(log)

    # --------------------------------------------------------
    # 2. Create pair specification
    # --------------------------------------------------------
    spec = graph_module.prepare_pair_spec(
        log_df=variants,
        net=net,
        im=im,
        fm=fm,
        fitness=0.0,
        precision=0.0,
        compute_targets_if_missing=False,
    )

    # --------------------------------------------------------
    # 3. Build DGL graph
    # --------------------------------------------------------
    g = graph_module.build_alignment_pair_graph(
        spec,
        variant=variant,
        max_local_onehot_dim=max_local_onehot_dim,
        random_permute_onehot=False,
        trace_freq_mode=trace_freq_mode,
    )

    return g


# def predict_loaded_pair_timed(
#     model,
#     log,
#     net,
#     im,
#     fm,
#     graph_module,
#     variant="A",
#     max_local_onehot_dim=None,
#     trace_freq_mode="normalized",
#     repeats=1,
#     warmup=False,
#     clear_behavior_cache_each_repeat=True,
# ):
#     """
#     Predict fitness and precision for ONE log/model pair already in memory.
#
#     Disk loading is excluded.
#
#     Timed pipeline:
#         loaded log + loaded Petri net
#             -> trace variant extraction
#             -> behavioral preprocessing
#             -> graph construction
#             -> device transfer
#             -> neural-network forward
#             -> prediction copied back to CPU
#     """
#
#     repeats = int(repeats)
#
#     if repeats < 1:
#         raise ValueError("repeats must be >= 1")
#
#     device = next(model.parameters()).device
#     model.eval()
#
#     # ========================================================
#     # Optional warm-up
#     # ========================================================
#
#     if warmup:
#         _clear_behavior_cache(graph_module)
#
#         g = _build_loaded_pair_graph(
#             log=log,
#             net=net,
#             im=im,
#             fm=fm,
#             graph_module=graph_module,
#             variant=variant,
#             max_local_onehot_dim=max_local_onehot_dim,
#             trace_freq_mode=trace_freq_mode,
#         )
#
#         g_batch = dgl.batch([g]).to(device)
#
#         with torch.inference_mode():
#             _ = model(g_batch)
#
#         _synchronize_device(device)
#
#         del g
#         del g_batch
#
#         # Do not allow warm-up preprocessing to populate
#         # the cache used by the first timed repetition.
#         _clear_behavior_cache(graph_module)
#
#     # ========================================================
#     # Timed repetitions
#     # ========================================================
#
#     total_times = []
#     preprocessing_times = []
#     forward_times = []
#
#     fitness_predictions = []
#     precision_predictions = []
#
#     with torch.inference_mode():
#
#         for _ in range(repeats):
#
#             if clear_behavior_cache_each_repeat:
#                 _clear_behavior_cache(graph_module)
#
#             # Make sure previous GPU work is complete.
#             _synchronize_device(device)
#
#             total_start_ns = time.perf_counter_ns()
#
#             # ------------------------------------------------
#             # Preprocessing + graph creation
#             # ------------------------------------------------
#             preprocessing_start_ns = time.perf_counter_ns()
#
#             g = _build_loaded_pair_graph(
#                 log=log,
#                 net=net,
#                 im=im,
#                 fm=fm,
#                 graph_module=graph_module,
#                 variant=variant,
#                 max_local_onehot_dim=max_local_onehot_dim,
#                 trace_freq_mode=trace_freq_mode,
#             )
#
#             g_batch = dgl.batch([g]).to(device)
#
#             _synchronize_device(device)
#
#             preprocessing_end_ns = time.perf_counter_ns()
#
#             # ------------------------------------------------
#             # Neural-network forward
#             # ------------------------------------------------
#             forward_start_ns = time.perf_counter_ns()
#
#             output = model(g_batch)
#
#             _synchronize_device(device)
#
#             forward_end_ns = time.perf_counter_ns()
#
#             preds = model_output_to_tensor(output)
#             preds = preds.detach().cpu()
#
#             total_end_ns = time.perf_counter_ns()
#
#             # ------------------------------------------------
#             # Store predictions
#             # ------------------------------------------------
#
#             fitness_predictions.append(
#                 float(preds[0, 0])
#             )
#
#             precision_predictions.append(
#                 float(preds[0, 1])
#             )
#
#             # ------------------------------------------------
#             # Store timing
#             # ------------------------------------------------
#
#             preprocessing_times.append(
#                 (
#                     preprocessing_end_ns
#                     - preprocessing_start_ns
#                 ) / 1e9
#             )
#
#             forward_times.append(
#                 (
#                     forward_end_ns
#                     - forward_start_ns
#                 ) / 1e9
#             )
#
#             total_times.append(
#                 (
#                     total_end_ns
#                     - total_start_ns
#                 ) / 1e9
#             )
#
#             del g
#             del g_batch
#             del output
#             del preds
#
#     # ========================================================
#     # Timing statistics
#     # ========================================================
#
#     def _mean(values):
#         return float(statistics.mean(values))
#
#     def _std(values):
#         if len(values) <= 1:
#             return 0.0
#         return float(statistics.stdev(values))
#
#     return {
#         # Prediction from first complete inference
#         "fitness": fitness_predictions[0],
#         "precision": precision_predictions[0],
#
#         # Prediction variability
#         "fitness_prediction_mean":
#             _mean(fitness_predictions),
#
#         "fitness_prediction_std":
#             _std(fitness_predictions),
#
#         "precision_prediction_mean":
#             _mean(precision_predictions),
#
#         "precision_prediction_std":
#             _std(precision_predictions),
#
#         # Full inference latency
#         "time_mean":
#             _mean(total_times),
#
#         "time_std":
#             _std(total_times),
#
#         "time_min":
#             min(total_times),
#
#         "time_max":
#             max(total_times),
#
#         # Breakdown
#         "preprocessing_time_mean":
#             _mean(preprocessing_times),
#
#         "forward_time_mean":
#             _mean(forward_times),
#
#         # Raw timing measurements
#         "repeat_times":
#             total_times,
#
#         "repeats":
#             repeats,
#     }

# def predict_loaded_pair_timed(
#     model,
#     log,
#     net,
#     im,
#     fm,
#     graph_module,
#     variant="A",
#     max_local_onehot_dim=None,
#     trace_freq_mode="normalized",
#     repeats=1,
#     warmup=False,
#     clear_behavior_cache_each_repeat=True,
# ):
#     """
#     Predict fitness and precision for ONE log/model pair already in memory.
#
#     Disk loading is excluded.
#
#     Timed pipeline:
#         loaded log + loaded Petri net
#             -> trace variant extraction
#             -> behavioral preprocessing
#             -> graph construction
#             -> device transfer
#             -> neural-network forward
#             -> prediction copied back to CPU
#     """
#
#     repeats = int(repeats)
#
#     if repeats < 1:
#         raise ValueError("repeats must be >= 1")
#
#     device = next(model.parameters()).device
#     model.eval()
#
#     # ========================================================
#     # Optional warm-up
#     # ========================================================
#
#     if warmup:
#         _clear_behavior_cache(graph_module)
#
#         g = _build_loaded_pair_graph(
#             log=log,
#             net=net,
#             im=im,
#             fm=fm,
#             graph_module=graph_module,
#             variant=variant,
#             max_local_onehot_dim=max_local_onehot_dim,
#             trace_freq_mode=trace_freq_mode,
#         )
#
#         g_batch = dgl.batch([g]).to(device)
#
#         with torch.inference_mode():
#             _ = model(g_batch)
#
#         _synchronize_device(device)
#
#         del g
#         del g_batch
#
#         # Do not allow warm-up preprocessing to populate
#         # the cache used by the first timed repetition.
#         _clear_behavior_cache(graph_module)
#
#     # ========================================================
#     # Timed repetitions
#     # ========================================================
#
#     total_times = []
#     preprocessing_times = []
#     forward_times = []
#
#     fitness_predictions = []
#     precision_predictions = []
#
#     with torch.inference_mode():
#
#         for _ in range(repeats):
#
#             if clear_behavior_cache_each_repeat:
#                 _clear_behavior_cache(graph_module)
#
#             # Make sure previous GPU work is complete.
#             _synchronize_device(device)
#
#             total_start_ns = time.perf_counter_ns()
#
#             # ------------------------------------------------
#             # Preprocessing + graph creation
#             # ------------------------------------------------
#
#             preprocessing_start_ns = time.perf_counter_ns()
#
#             g = _build_loaded_pair_graph(
#                 log=log,
#                 net=net,
#                 im=im,
#                 fm=fm,
#                 graph_module=graph_module,
#                 variant=variant,
#                 max_local_onehot_dim=max_local_onehot_dim,
#                 trace_freq_mode=trace_freq_mode,
#             )
#
#             g_batch = dgl.batch([g]).to(device)
#
#             _synchronize_device(device)
#
#             preprocessing_end_ns = time.perf_counter_ns()
#
#             # ------------------------------------------------
#             # Neural-network forward
#             # ------------------------------------------------
#
#             forward_start_ns = time.perf_counter_ns()
#
#             output = model(g_batch)
#
#             _synchronize_device(device)
#
#             forward_end_ns = time.perf_counter_ns()
#
#             preds = model_output_to_tensor(output)
#             preds = preds.detach().cpu()
#
#             total_end_ns = time.perf_counter_ns()
#
#             # ------------------------------------------------
#             # Store predictions
#             # ------------------------------------------------
#
#             fitness_predictions.append(
#                 float(preds[0, 0])
#             )
#
#             precision_predictions.append(
#                 float(preds[0, 1])
#             )
#
#             # ------------------------------------------------
#             # Store timing
#             # ------------------------------------------------
#
#             preprocessing_times.append(
#                 (
#                     preprocessing_end_ns
#                     - preprocessing_start_ns
#                 ) / 1e9
#             )
#
#             forward_times.append(
#                 (
#                     forward_end_ns
#                     - forward_start_ns
#                 ) / 1e9
#             )
#
#             total_times.append(
#                 (
#                     total_end_ns
#                     - total_start_ns
#                 ) / 1e9
#             )
#
#             del g
#             del g_batch
#             del output
#             del preds
#
#     # ========================================================
#     # Timing statistics
#     # ========================================================
#
#     def _mean(values):
#         return float(statistics.mean(values))
#
#     def _std(values):
#         if len(values) <= 1:
#             return 0.0
#
#         return float(statistics.stdev(values))
#
#     return {
#         # Prediction from first complete inference
#         "fitness":
#             fitness_predictions[0],
#
#         "precision":
#             precision_predictions[0],
#
#         # Prediction variability
#         "fitness_prediction_mean":
#             _mean(fitness_predictions),
#
#         "fitness_prediction_std":
#             _std(fitness_predictions),
#
#         "precision_prediction_mean":
#             _mean(precision_predictions),
#
#         "precision_prediction_std":
#             _std(precision_predictions),
#
#         # Full inference latency
#         "time_mean":
#             _mean(total_times),
#
#         "time_std":
#             _std(total_times),
#
#         "time_min":
#             min(total_times),
#
#         "time_max":
#             max(total_times),
#
#         # Mean breakdown
#         "preprocessing_time_mean":
#             _mean(preprocessing_times),
#
#         "forward_time_mean":
#             _mean(forward_times),
#
#         # ====================================================
#         # Raw per-repeat timing measurements
#         # ====================================================
#
#         "preprocessing_repeat_times":
#             preprocessing_times,
#
#         "forward_repeat_times":
#             forward_times,
#
#         "repeat_times":
#             total_times,
#
#         "repeats":
#             repeats,
#     }

def predict_loaded_pair_timed(
    model,
    log,
    net,
    im,
    fm,
    graph_module,
    variant="A",
    max_local_onehot_dim=None,
    trace_freq_mode="normalized",
    repeats=1,
    warmup=False,
    clear_behavior_cache_each_repeat=True,
):
    """
    Predict fitness and precision for ONE log/model pair already in memory.

    Disk loading is excluded.

    Timed pipeline:
        loaded log + loaded Petri net
            -> trace variant extraction
            -> behavioral preprocessing
            -> graph construction
            -> device transfer
            -> neural-network forward
            -> prediction copied back to CPU

    Both mean and median timing statistics are returned.

    Predictions are deterministic, so only the prediction from
    the first timed repetition is retained.
    """

    repeats = int(repeats)

    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    device = next(model.parameters()).device
    model.eval()

    # ========================================================
    # Optional warm-up
    # ========================================================

    if warmup:
        _clear_behavior_cache(graph_module)

        g = _build_loaded_pair_graph(
            log=log,
            net=net,
            im=im,
            fm=fm,
            graph_module=graph_module,
            variant=variant,
            max_local_onehot_dim=max_local_onehot_dim,
            trace_freq_mode=trace_freq_mode,
        )

        g_batch = dgl.batch([g]).to(device)

        with torch.inference_mode():
            _ = model(g_batch)

        _synchronize_device(device)

        del g
        del g_batch

        # Do not allow warm-up preprocessing to populate
        # the cache used by the first timed repetition.
        _clear_behavior_cache(graph_module)

    # ========================================================
    # Timed repetitions
    # ========================================================

    total_times = []
    preprocessing_times = []
    forward_times = []

    fitness_prediction = None
    precision_prediction = None

    with torch.inference_mode():

        for repeat_idx in range(repeats):

            if clear_behavior_cache_each_repeat:
                _clear_behavior_cache(graph_module)

            # Make sure previous GPU work is complete.
            # On CPU this does nothing.
            _synchronize_device(device)

            total_start_ns = time.perf_counter_ns()

            # ------------------------------------------------
            # Preprocessing + graph creation
            # ------------------------------------------------

            preprocessing_start_ns = time.perf_counter_ns()

            g = _build_loaded_pair_graph(
                log=log,
                net=net,
                im=im,
                fm=fm,
                graph_module=graph_module,
                variant=variant,
                max_local_onehot_dim=max_local_onehot_dim,
                trace_freq_mode=trace_freq_mode,
            )

            g_batch = dgl.batch([g]).to(device)

            _synchronize_device(device)

            preprocessing_end_ns = time.perf_counter_ns()

            # ------------------------------------------------
            # Neural-network forward
            # ------------------------------------------------

            forward_start_ns = time.perf_counter_ns()

            output = model(g_batch)

            _synchronize_device(device)

            forward_end_ns = time.perf_counter_ns()

            preds = model_output_to_tensor(output)
            preds = preds.detach().cpu()

            total_end_ns = time.perf_counter_ns()

            # ------------------------------------------------
            # Store prediction once
            # ------------------------------------------------

            if repeat_idx == 0:
                fitness_prediction = float(
                    preds[0, 0]
                )

                precision_prediction = float(
                    preds[0, 1]
                )

            # ------------------------------------------------
            # Store timing
            # ------------------------------------------------

            preprocessing_times.append(
                (
                    preprocessing_end_ns
                    - preprocessing_start_ns
                ) / 1e9
            )

            forward_times.append(
                (
                    forward_end_ns
                    - forward_start_ns
                ) / 1e9
            )

            total_times.append(
                (
                    total_end_ns
                    - total_start_ns
                ) / 1e9
            )

            del g
            del g_batch
            del output
            del preds

    # ========================================================
    # Timing statistics
    # ========================================================

    def _mean(values):
        return float(
            statistics.mean(values)
        )

    def _median(values):
        return float(
            statistics.median(values)
        )

    def _std(values):
        if len(values) <= 1:
            return 0.0

        return float(
            statistics.stdev(values)
        )

    return {
        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        "fitness":
            fitness_prediction,

        "precision":
            precision_prediction,

        # ----------------------------------------------------
        # Total timing
        # ----------------------------------------------------

        "time_mean":
            _mean(total_times),

        "time_median":
            _median(total_times),

        "time_std":
            _std(total_times),

        "time_min":
            min(total_times),

        "time_max":
            max(total_times),

        # ----------------------------------------------------
        # Preprocessing timing
        # ----------------------------------------------------

        "preprocessing_time_mean":
            _mean(preprocessing_times),

        "preprocessing_time_median":
            _median(preprocessing_times),

        # ----------------------------------------------------
        # Forward timing
        # ----------------------------------------------------

        "forward_time_mean":
            _mean(forward_times),

        "forward_time_median":
            _median(forward_times),

        # ----------------------------------------------------
        # Raw per-repeat timing
        # ----------------------------------------------------

        "preprocessing_repeat_times":
            preprocessing_times,

        "forward_repeat_times":
            forward_times,

        "repeat_times":
            total_times,

        "repeats":
            repeats,
    }


# def predict_records_timed(
#     model,
#     records,
#     variant="A",
#     max_local_onehot_dim=None,
#     trace_freq_mode="normalized",
#     repeats=1,
#     graph_module_name="graph",
# ):
#     """
#     Evaluate records one pair at a time.
#
#     XES/PNML files are loaded before timing each pair.
#     Only one log/model pair is kept in memory at a time.
#
#     graph_module_name examples:
#         "graph"
#         "graph_pgnn_attention"
#         "graph_pgnn_unweighted"
#     """
#
#     # ========================================================
#     # Load requested graph implementation
#     # ========================================================
#
#     graph_module = importlib.import_module(
#         graph_module_name
#     )
#
#     rows = []
#
#     model.eval()
#
#     first_pair = True
#
#     for record in records:
#
#         # ====================================================
#         # LOAD FROM DISK -- NOT TIMED
#         # ====================================================
#
#         log, net, im, fm = load_pair_to_memory(
#             log_path=record["log_path"],
#             model_path=record["model_path"],
#         )
#
#         # ====================================================
#         # IN-MEMORY INFERENCE -- TIMED
#         # ====================================================
#
#         result = predict_loaded_pair_timed(
#             model=model,
#             log=log,
#             net=net,
#             im=im,
#             fm=fm,
#
#             # Selected graph implementation
#             graph_module=graph_module,
#
#             variant=variant,
#             max_local_onehot_dim=max_local_onehot_dim,
#             trace_freq_mode=trace_freq_mode,
#             repeats=repeats,
#
#             # Only warm up the network once
#             warmup=first_pair,
#
#             # Important for fair repeated timing
#             clear_behavior_cache_each_repeat=True,
#         )
#
#         first_pair = False
#
#         true_fitness = float(
#             record["fitness"]
#         )
#
#         true_precision = float(
#             record["precision"]
#         )
#
#         rows.append({
#             "name":
#                 record["name"],
#
#             "log_path":
#                 str(record["log_path"]),
#
#             "model_path":
#                 str(record["model_path"]),
#
#             "predicted_fitness":
#                 result["fitness"],
#
#             "true_fitness":
#                 true_fitness,
#
#             "predicted_precision":
#                 result["precision"],
#
#             "true_precision":
#                 true_precision,
#
#             "fitness_error":
#                 result["fitness"]
#                 - true_fitness,
#
#             "fitness_abs_error":
#                 abs(
#                     result["fitness"]
#                     - true_fitness
#                 ),
#
#             "precision_error":
#                 result["precision"]
#                 - true_precision,
#
#             "precision_abs_error":
#                 abs(
#                     result["precision"]
#                     - true_precision
#                 ),
#
#             # Timing
#             "inference_time_mean":
#                 result["time_mean"],
#
#             "inference_time_std":
#                 result["time_std"],
#
#             "inference_time_min":
#                 result["time_min"],
#
#             "inference_time_max":
#                 result["time_max"],
#
#             "preprocessing_time_mean":
#                 result["preprocessing_time_mean"],
#
#             "forward_time_mean":
#                 result["forward_time_mean"],
#
#             # Prediction variability
#             "fitness_prediction_std":
#                 result["fitness_prediction_std"],
#
#             "precision_prediction_std":
#                 result["precision_prediction_std"],
#
#             "repeats":
#                 result["repeats"],
#         })
#
#         # Release pair before loading next
#         del log
#         del net
#         del im
#         del fm
#
#         gc.collect()
#
#     return pd.DataFrame(rows)

# def predict_records_timed(
#     model,
#     records,
#     variant="A",
#     max_local_onehot_dim=None,
#     trace_freq_mode="normalized",
#     repeats=1,
#     graph_module_name="graph",
# ):
#     """
#     Evaluate records one pair at a time.
#
#     Each distinct XES log is loaded only ONCE.
#
#     All models belonging to that log are evaluated before
#     moving to the next log.
#
#     XES/PNML disk loading remains outside the timed inference.
#
#     Original record order is restored before returning.
#     """
#
#     # ========================================================
#     # Load requested graph implementation
#     # ========================================================
#
#     graph_module = importlib.import_module(
#         graph_module_name
#     )
#
#     model.eval()
#
#     # ========================================================
#     # Group records by log
#     #
#     # Store original record index so that we can restore the
#     # exact original order afterward.
#     # ========================================================
#
#     records_by_log = {}
#
#     for record_idx, record in enumerate(records):
#
#         log_key = str(record["log_path"])
#
#         if log_key not in records_by_log:
#             records_by_log[log_key] = []
#
#         records_by_log[log_key].append(
#             (record_idx, record)
#         )
#
#     rows = []
#
#     first_pair = True
#
#     # ========================================================
#     # Process one log at a time
#     # ========================================================
#
#     for log_path, log_records in records_by_log.items():
#
#         # ====================================================
#         # LOAD XES ONCE -- NOT TIMED
#         # ====================================================
#
#         log = xes_importer.apply(
#             str(log_path),
#             parameters={
#                 xes_importer.Variants
#                 .ITERPARSE
#                 .value
#                 .Parameters
#                 .SHOW_PROGRESS_BAR:
#                     False
#             },
#         )
#
#         # ====================================================
#         # Evaluate every model associated with this log
#         # ====================================================
#
#         for record_idx, record in log_records:
#
#             # ================================================
#             # LOAD PNML -- NOT TIMED
#             # ================================================
#
#             net, im, fm = pnml_importer.apply(
#                 str(record["model_path"])
#             )
#
#             # ================================================
#             # IN-MEMORY INFERENCE -- TIMED
#             # ================================================
#
#             result = predict_loaded_pair_timed(
#                 model=model,
#                 log=log,
#                 net=net,
#                 im=im,
#                 fm=fm,
#
#                 graph_module=graph_module,
#
#                 variant=variant,
#
#                 max_local_onehot_dim=
#                     max_local_onehot_dim,
#
#                 trace_freq_mode=
#                     trace_freq_mode,
#
#                 repeats=repeats,
#
#                 # Only warm up the network once
#                 warmup=first_pair,
#
#                 clear_behavior_cache_each_repeat=True,
#             )
#
#             first_pair = False
#
#             true_fitness = float(
#                 record["fitness"]
#             )
#
#             true_precision = float(
#                 record["precision"]
#             )
#
#             row = {
#                 # Needed to restore original ordering
#                 "_record_idx":
#                     record_idx,
#
#                 "name":
#                     record["name"],
#
#                 "log_path":
#                     str(record["log_path"]),
#
#                 "model_path":
#                     str(record["model_path"]),
#
#                 "predicted_fitness":
#                     result["fitness"],
#
#                 "true_fitness":
#                     true_fitness,
#
#                 "predicted_precision":
#                     result["precision"],
#
#                 "true_precision":
#                     true_precision,
#
#                 "fitness_error":
#                     result["fitness"]
#                     - true_fitness,
#
#                 "fitness_abs_error":
#                     abs(
#                         result["fitness"]
#                         - true_fitness
#                     ),
#
#                 "precision_error":
#                     result["precision"]
#                     - true_precision,
#
#                 "precision_abs_error":
#                     abs(
#                         result["precision"]
#                         - true_precision
#                     ),
#
#                 # ============================================
#                 # Mean timing
#                 # ============================================
#
#                 "inference_time_mean":
#                     result["time_mean"],
#
#                 "inference_time_std":
#                     result["time_std"],
#
#                 "inference_time_min":
#                     result["time_min"],
#
#                 "inference_time_max":
#                     result["time_max"],
#
#                 "preprocessing_time_mean":
#                     result[
#                         "preprocessing_time_mean"
#                     ],
#
#                 "forward_time_mean":
#                     result[
#                         "forward_time_mean"
#                     ],
#
#                 # ============================================
#                 # Raw per-repeat timing
#                 # ============================================
#
#                 "preprocessing_repeat_times":
#                     result[
#                         "preprocessing_repeat_times"
#                     ],
#
#                 "forward_repeat_times":
#                     result[
#                         "forward_repeat_times"
#                     ],
#
#                 "repeat_times":
#                     result["repeat_times"],
#
#                 # ============================================
#                 # Prediction variability
#                 # ============================================
#
#                 "fitness_prediction_std":
#                     result[
#                         "fitness_prediction_std"
#                     ],
#
#                 "precision_prediction_std":
#                     result[
#                         "precision_prediction_std"
#                     ],
#
#                 "repeats":
#                     result["repeats"],
#             }
#
#             rows.append(row)
#
#             # ================================================
#             # Existing cleanup behaviour
#             # ================================================
#
#             del net
#             del im
#             del fm
#
#             gc.collect()
#
#         # ====================================================
#         # Finished every model belonging to this XES log
#         # ====================================================
#
#         del log
#
#     # ========================================================
#     # Restore exact original record order
#     # ========================================================
#
#     rows.sort(
#         key=lambda row: row["_record_idx"]
#     )
#
#     for row in rows:
#         del row["_record_idx"]
#
#     return pd.DataFrame(rows)

def predict_records_timed(
    model,
    records,
    variant="A",
    max_local_onehot_dim=None,
    trace_freq_mode="normalized",
    repeats=1,
    graph_module_name="graph",
):
    """
    Evaluate log/model records with repeated timing.

    Each distinct XES log is loaded only ONCE.

    All Petri-net models associated with that log are evaluated
    before moving to the next log.

    XES and PNML disk loading are excluded from timed inference.

    Both mean and median timings are retained.

    Original input record order is restored before returning.
    """

    # ========================================================
    # Load requested graph implementation
    # ========================================================

    graph_module = importlib.import_module(
        graph_module_name
    )

    model.eval()

    # ========================================================
    # Group records by log
    #
    # Keep original record index because downstream evaluation
    # assumes predictions have the same order as reference CSV.
    # ========================================================

    records_by_log = {}

    for record_idx, record in enumerate(records):

        log_key = str(
            record["log_path"]
        )

        if log_key not in records_by_log:
            records_by_log[log_key] = []

        records_by_log[log_key].append(
            (
                record_idx,
                record,
            )
        )

    rows = []

    first_pair = True

    # ========================================================
    # Process one log at a time
    # ========================================================

    for log_path, log_records in records_by_log.items():

        # ====================================================
        # LOAD XES ONCE -- NOT TIMED
        # ====================================================

        log = xes_importer.apply(
            str(log_path),
            parameters={
                xes_importer
                .Variants
                .ITERPARSE
                .value
                .Parameters
                .SHOW_PROGRESS_BAR:
                    False
            },
        )

        # ====================================================
        # Evaluate all models belonging to this log
        # ====================================================

        for record_idx, record in log_records:

            # ================================================
            # LOAD PNML -- NOT TIMED
            # ================================================

            net, im, fm = pnml_importer.apply(
                str(record["model_path"])
            )

            # ================================================
            # TIMED IN-MEMORY INFERENCE
            # ================================================

            result = predict_loaded_pair_timed(
                model=model,
                log=log,
                net=net,
                im=im,
                fm=fm,

                graph_module=graph_module,

                variant=variant,

                max_local_onehot_dim=
                    max_local_onehot_dim,

                trace_freq_mode=
                    trace_freq_mode,

                repeats=repeats,

                # Warm up only once for the whole benchmark.
                warmup=first_pair,

                # Keep your existing cold-cache behaviour.
                clear_behavior_cache_each_repeat=True,
            )

            first_pair = False

            # ================================================
            # Ground truth
            # ================================================

            true_fitness = float(
                record["fitness"]
            )

            true_precision = float(
                record["precision"]
            )

            # ================================================
            # Store
            # ================================================

            rows.append({
                # --------------------------------------------
                # Temporary index used to restore order
                # --------------------------------------------

                "_record_idx":
                    record_idx,

                # --------------------------------------------
                # Identification
                # --------------------------------------------

                "name":
                    record["name"],

                "log_path":
                    str(record["log_path"]),

                "model_path":
                    str(record["model_path"]),

                # --------------------------------------------
                # Predictions
                # --------------------------------------------

                "predicted_fitness":
                    result["fitness"],

                "true_fitness":
                    true_fitness,

                "predicted_precision":
                    result["precision"],

                "true_precision":
                    true_precision,

                # --------------------------------------------
                # Errors
                # --------------------------------------------

                "fitness_error":
                    result["fitness"]
                    - true_fitness,

                "fitness_abs_error":
                    abs(
                        result["fitness"]
                        - true_fitness
                    ),

                "precision_error":
                    result["precision"]
                    - true_precision,

                "precision_abs_error":
                    abs(
                        result["precision"]
                        - true_precision
                    ),

                # --------------------------------------------
                # Mean timing
                # --------------------------------------------

                "inference_time_mean":
                    result["time_mean"],

                "preprocessing_time_mean":
                    result[
                        "preprocessing_time_mean"
                    ],

                "forward_time_mean":
                    result[
                        "forward_time_mean"
                    ],

                # --------------------------------------------
                # Median timing
                # --------------------------------------------

                "inference_time_median":
                    result["time_median"],

                "preprocessing_time_median":
                    result[
                        "preprocessing_time_median"
                    ],

                "forward_time_median":
                    result[
                        "forward_time_median"
                    ],

                # --------------------------------------------
                # Other timing statistics
                # --------------------------------------------

                "inference_time_std":
                    result["time_std"],

                "inference_time_min":
                    result["time_min"],

                "inference_time_max":
                    result["time_max"],

                # --------------------------------------------
                # Raw repeated measurements
                # --------------------------------------------

                "preprocessing_repeat_times":
                    result[
                        "preprocessing_repeat_times"
                    ],

                "forward_repeat_times":
                    result[
                        "forward_repeat_times"
                    ],

                "repeat_times":
                    result[
                        "repeat_times"
                    ],

                "repeats":
                    result["repeats"],
            })

            # ================================================
            # Existing cleanup behaviour
            # ================================================

            del net
            del im
            del fm

            gc.collect()

        # ====================================================
        # Finished all models for this log
        # ====================================================

        del log

    # ========================================================
    # Restore original input order
    # ========================================================

    rows.sort(
        key=lambda row: row["_record_idx"]
    )

    for row in rows:
        del row["_record_idx"]

    return pd.DataFrame(rows)