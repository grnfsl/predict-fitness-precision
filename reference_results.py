from __future__ import annotations

import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from IPython.display import display

import importlib
import inference
importlib.reload(inference)

from inference import predict_records_timed


# ============================================================
# 1. CSV loading / cleaning
# ============================================================

_COLUMN_ALIASES = {
    "log id": "log_id",
    "log path": "log_path",
    "algorithm": "algorithm",
    "model name": "model_name",
    "model path": "model_path",

    "exact fit": "exact_fit",
    "exact fitness": "exact_fit",
    "exact avg fit": "exact_fit",

    "exact prec": "exact_prec",
    "exact precision": "exact_prec",

    "exact fit time s": "exact_fit_time",
    "exact fitness time s": "exact_fit_time",

    "exact prec time s": "exact_prec_time",
    "exact precision time s": "exact_prec_time",

    "approx fit": "sampling_fit",
    "approx fitness": "sampling_fit",
    "sampling fit": "sampling_fit",
    "sampling fitness": "sampling_fit",

    "approx time s": "sampling_time",
    "sampling time s": "sampling_time",
}


def _canonical_column_name(name):
    """
    Convert e.g.
        'exact fit time (s)' -> 'exact fit time s'
        'approx. fit'        -> 'approx fit'
    """
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(name).strip().lower(),
    ).strip()


def _to_number_or_nan(value):
    """
    Convert numeric CSV values to float.

    TL / empty / invalid values become NaN so pandas automatically
    excludes them from MAE / mean PI calculations.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text == "" or text.upper() == "TL":
        return np.nan

    try:
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def load_real_life_reference_csv(csv_path):
    """
    Expected logical CSV columns:

        log_id
        log_path
        algorithm
        model_name
        model_path
        exact fit
        exact prec
        exact fit time (s)
        exact prec time (s)
        approx. fit
        approx time (s)

    Alpha Miner rows are removed completely.
    """
    df = pd.read_csv(csv_path)

    rename_map = {}

    for col in df.columns:
        canonical = _canonical_column_name(col)

        if canonical in _COLUMN_ALIASES:
            rename_map[col] = _COLUMN_ALIASES[canonical]

    df = df.rename(columns=rename_map)

    required = [
        "log_id",
        "log_path",
        "algorithm",
        "model_name",
        "model_path",
        "exact_fit",
        "exact_prec",
        "exact_fit_time",
        "exact_prec_time",
        "sampling_fit",
        "sampling_time",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Reference CSV is missing required columns: "
            + ", ".join(missing)
            + "\nColumns found: "
            + str(list(df.columns))
        )

    # Keep original text as well, useful when inspecting/debugging.
    numeric_columns = [
        "exact_fit",
        "exact_prec",
        "exact_fit_time",
        "exact_prec_time",
        "sampling_fit",
        "sampling_time",
    ]

    for col in numeric_columns:
        df[f"{col}_raw"] = df[col].copy()
        df[col] = df[col].map(_to_number_or_nan)

    text_columns = [
        "log_id",
        "log_path",
        "algorithm",
        "model_name",
        "model_path",
    ]

    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()

    # Ignore Alpha Miner rows/models completely.
    alpha_mask = (
        df["algorithm"].str.contains(
            "alpha",
            case=False,
            na=False,
        )
        |
        df["model_name"].str.contains(
            "alpha",
            case=False,
            na=False,
        )
    )

    df = df.loc[~alpha_mask].copy()

    # Preserve original CSV order.
    df["_reference_order"] = np.arange(
        len(df),
        dtype=int,
    )

    # Extract compact algorithm/parameter labels.
    parsed = df.apply(
        lambda row: _parse_algorithm_and_parameters(
            row["model_name"],
            row["algorithm"],
        ),
        axis=1,
        result_type="expand",
    )

    parsed.columns = [
        "_algorithm_display",
        "_parameters_display",
        "_parameters_latex",
    ]

    df = pd.concat(
        [df, parsed],
        axis=1,
    )

    return df.reset_index(drop=True)


# ============================================================
# 2. Algorithm / parameter parsing
# ============================================================

def _decode_filename_number(token):
    """
    Filename encoding:
        0p200000 -> 0.2
        0p65     -> 0.65
        0p5      -> 0.5
    """
    if token is None:
        return None

    text = str(token).strip().lower()

    if re.fullmatch(r"-?\d+p\d+", text):
        text = text.replace("p", ".")

    try:
        value = float(text)
    except ValueError:
        return None

    return f"{value:g}"


def _model_segments(model_name):
    name = Path(str(model_name)).name

    if name.lower().endswith(".pnml"):
        name = name[:-5]

    return name.split("__")


def _find_segment_parameter(
    segments,
    keys,
):
    for segment in segments:
        segment_lower = segment.lower()

        for key in keys:
            prefix = key.lower() + "-"

            if segment_lower.startswith(prefix):
                raw_value = segment[len(prefix):]
                return _decode_filename_number(
                    raw_value
                )

    return None


def _parse_algorithm_and_parameters(
    model_name,
    algorithm,
):
    """
    Examples:

    BPIC-2012__inductive__noise_threshold-0p200000__...
        -> IM, n=0.2

    heuristics__and-0p65__dep-0p5__loop2-0p5.pnml
        -> HM, d=0.5, a=0.65, l=0.5
    """
    algorithm_text = str(algorithm).strip().lower()
    model_text = Path(str(model_name)).name.lower()
    segments = _model_segments(model_name)

    # --------------------------------------------------------
    # Inductive Miner
    # --------------------------------------------------------
    if (
        "inductive" in algorithm_text
        or "inductive" in model_text
    ):
        n = _find_segment_parameter(
            segments,
            [
                "noise_threshold",
                "noise-threshold",
                "noise",
            ],
        )

        if n is None:
            return "IM", "--", "--"

        return (
            "IM",
            f"n={n}",
            f"$n={n}$",
        )

    # --------------------------------------------------------
    # Heuristics Miner
    # --------------------------------------------------------
    if (
        "heur" in algorithm_text
        or "heuristic" in model_text
    ):
        d = _find_segment_parameter(
            segments,
            [
                "dep",
                "dependency_threshold",
                "dependency-threshold",
            ],
        )

        a = _find_segment_parameter(
            segments,
            [
                "and",
                "and_threshold",
                "and-threshold",
            ],
        )

        l = _find_segment_parameter(
            segments,
            [
                "loop2",
                "loop_two_threshold",
                "loop-two-threshold",
            ],
        )

        parts = []

        if d is not None:
            parts.append(f"d={d}")

        if a is not None:
            parts.append(f"a={a}")

        if l is not None:
            parts.append(f"l={l}")

        if not parts:
            return "HM", "--", "--"

        return (
            "HM",
            ", ".join(parts),
            "$" + r",\,".join(parts) + "$",
        )

    # Other non-alpha algorithms are retained as-is.
    return str(algorithm), "--", "--"


# ============================================================
# 3. Build inference records directly from CSV
# ============================================================

def _prepare_records_from_reference(
    reference_df,
):
    """
    Build exactly the records required by predict_records_timed().

    TL targets have already become np.nan. float(np.nan) is valid,
    so the inference code can still evaluate those log/model pairs.
    """
    records = []

    for _, row in reference_df.iterrows():
        records.append({
            "name":
                Path(row["model_name"]).stem,

            "log_id":
                row["log_id"],

            "log_path":
                Path(row["log_path"]),

            "model_path":
                Path(row["model_path"]),

            "fitness":
                row["exact_fit"],

            "precision":
                row["exact_prec"],
        })

    if not records:
        raise ValueError(
            "No non-alpha rows are available for evaluation."
        )

    return records


# ============================================================
# 4. Metrics
# ============================================================

def _performance_improvement(
    exact_time,
    approx_time,
):
    """
    Performance Improvement (PI):

        PI = T_exact / T_approx

    Examples:
        PI = 1   -> same speed
        PI = 10  -> approximation is 10x faster
        PI < 1   -> approximation is slower

    Missing / TL / zero / negative times produce NaN.
    """
    exact = pd.to_numeric(
        exact_time,
        errors="coerce",
    )

    approx = pd.to_numeric(
        approx_time,
        errors="coerce",
    )

    valid = (
        exact.notna()
        & approx.notna()
        & (exact > 0)
        & (approx > 0)
    )

    result = pd.Series(
        np.nan,
        index=exact.index,
        dtype=float,
    )

    result.loc[valid] = (
        exact.loc[valid]
        / approx.loc[valid]
    )

    return result


def _build_evaluation_df(
    results_df,
    reference_df,
):
    """
    Merge predictions/timing with CSV exact and sampling results.

    Row order is deliberately positional because
    predict_records_timed() preserves the input record order.
    """
    if len(results_df) != len(reference_df):
        raise ValueError(
            "Prediction/reference row-count mismatch: "
            f"{len(results_df)} predictions vs "
            f"{len(reference_df)} CSV rows."
        )

    out = pd.DataFrame({
        "log_id":
            reference_df["log_id"].to_numpy(),

        "log_path":
            reference_df["log_path"].to_numpy(),

        "algorithm":
            reference_df["_algorithm_display"].to_numpy(),

        "parameters":
            reference_df["_parameters_display"].to_numpy(),

        "parameters_latex":
            reference_df["_parameters_latex"].to_numpy(),

        "model_name":
            reference_df["model_name"].to_numpy(),

        "model_path":
            reference_df["model_path"].to_numpy(),

        "exact_fit":
            reference_df["exact_fit"].to_numpy(),

        "exact_prec":
            reference_df["exact_prec"].to_numpy(),

        "exact_fit_time":
            reference_df["exact_fit_time"].to_numpy(),

        "exact_prec_time":
            reference_df["exact_prec_time"].to_numpy(),

        "sampling_fit":
            reference_df["sampling_fit"].to_numpy(),

        "sampling_time":
            reference_df["sampling_time"].to_numpy(),

        "ours_fit":
            results_df["predicted_fitness"].to_numpy(),

        "ours_prec":
            results_df["predicted_precision"].to_numpy(),

        "ours_time_mean":
            results_df["inference_time_mean"].to_numpy(),

        "ours_time_median":
            results_df["inference_time_median"].to_numpy(),

        "_reference_order":
            reference_df["_reference_order"].to_numpy(),
    })

    # --------------------------------------------------------
    # Approximation errors
    # --------------------------------------------------------
    # If exact value is TL/NaN, result is NaN automatically.
    out["ours_fit_error"] = (
        out["ours_fit"]
        - out["exact_fit"]
    ).abs()

    out["sampling_fit_error"] = (
        out["sampling_fit"]
        - out["exact_fit"]
    ).abs()

    out["ours_prec_error"] = (
        out["ours_prec"]
        - out["exact_prec"]
    ).abs()

    # --------------------------------------------------------
    # PI = exact time / approximation time
    # --------------------------------------------------------
    # out["ours_fit_pi"] = (
    #     _performance_improvement(
    #         out["exact_fit_time"],
    #         out["ours_time"],
    #     )
    # )

    out["sampling_fit_pi"] = (
        _performance_improvement(
            out["exact_fit_time"],
            out["sampling_time"],
        )
    )

    # out["ours_prec_pi"] = (
    #     _performance_improvement(
    #         out["exact_prec_time"],
    #         out["ours_time"],
    #     )
    # )

    return (
        out
        .sort_values(
            "_reference_order",
            kind="stable",
        )
        .reset_index(drop=True)
    )


# ============================================================
# 5. Log-name formatting
# ============================================================

DEFAULT_LOG_NAME_MAP = {
    "BPIC-2012": "BPIC12",
    "BPIC-2017": "BPIC17",

    "Hospital-Billing":
        "Hospital Billing",

    "Road-Trafic-Fine-Management-Process":
        "Road Traffic",

    "Road-Traffic-Fine-Management-Process":
        "Road Traffic",

    "Sepsis-cases":
        "Sepsis",
}


def _remove_xes_suffix(text):
    return re.sub(
        r"\.xes$",
        "",
        str(text).strip(),
        flags=re.IGNORECASE,
    )


def _pretty_log_name(
    log_id,
    log_name_map=None,
):
    """
    Examples:
        BPIC-2012.xes
            -> BPIC12

        BPIC-2013-closed-problems.xes
            -> BPIC13 closed problems

        Hospital-Billing.xes
            -> Hospital Billing
    """
    raw = _remove_xes_suffix(
        str(log_id)
    )

    mapping = dict(
        DEFAULT_LOG_NAME_MAP
    )

    if log_name_map:
        # Also strip .xes from custom-map keys.
        mapping.update({
            _remove_xes_suffix(k): v
            for k, v in log_name_map.items()
        })

    if raw in mapping:
        return _remove_xes_suffix(
            mapping[raw]
        )

    # Generic BPIC conversion.
    match = re.match(
        r"^BPIC[-_]?(\d{4})(.*)$",
        raw,
        flags=re.IGNORECASE,
    )

    if match:
        year = match.group(1)[-2:]
        suffix = match.group(2).strip(
            "-_ "
        )

        if suffix:
            suffix = re.sub(
                r"[-_]+",
                " ",
                suffix,
            )

            suffix = re.sub(
                r"\s+",
                " ",
                suffix,
            ).strip()

            return f"BPIC{year} {suffix}"

        return f"BPIC{year}"

    # Generic fallback.
    raw = re.sub(
        r"[-_]+",
        " ",
        raw,
    )

    raw = re.sub(
        r"\s+",
        " ",
        raw,
    ).strip()

    return raw


def _wrap_log_name(
    name,
    width=18,
):
    """
    Wrap only the text inside the log cell.

    This function does NOT create dataframe/table rows.

    For a long BPIC name, prefer:
        BPIC13
        closed problems

    rather than:
        BPIC13 closed
        problems
    """
    name = str(name).strip()

    if width is None:
        return name

    width = int(width)

    if width <= 0:
        return name

    if len(name) <= width:
        return name

    # Prefer a clean break after BPICxx.
    match = re.match(
        r"^(BPIC\d{2})\s+(.+)$",
        name,
        flags=re.IGNORECASE,
    )

    if match:
        prefix = match.group(1)
        suffix = match.group(2)

        suffix_lines = textwrap.wrap(
            suffix,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )

        return "\n".join(
            [prefix] + suffix_lines
        )

    return "\n".join(
        textwrap.wrap(
            name,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


# ============================================================
# 6. Number formatting
# ============================================================

def _validate_decimal_places(
    decimal_places,
):
    value = int(decimal_places)

    if value < 0:
        raise ValueError(
            "decimal_places must be >= 0"
        )

    return value


def _validate_wrap_width(
    width,
):
    if width is None:
        return None

    value = int(width)

    if value < 0:
        raise ValueError(
            "log_wrap_width must be >= 0 or None"
        )

    return value


def _fmt_number(
    value,
    decimals=3,
    missing="--",
):
    if pd.isna(value):
        return missing

    return f"{float(value):.{decimals}f}"


def _fmt_exact(
    value,
    decimals=3,
):
    # Missing exact value means TL in this benchmark.
    return _fmt_number(
        value,
        decimals=decimals,
        missing="TL",
    )


# ============================================================
# 7. Notebook dataframe builders
# ============================================================

def _fitness_display_table(
    evaluation_df,
    log_name_map=None,
    decimal_places=3,
    log_wrap_width=18,
):
    decimal_places = _validate_decimal_places(
        decimal_places
    )

    log_wrap_width = _validate_wrap_width(
        log_wrap_width
    )

    rows = []
    previous_log = None

    for _, row in evaluation_df.iterrows():
        current_log = row["log_id"]

        log_cell = ""

        if current_log != previous_log:
            pretty_log = _pretty_log_name(
                current_log,
                log_name_map,
            )

            log_cell = _wrap_log_name(
                pretty_log,
                width=log_wrap_width,
            )

        previous_log = current_log

        rows.append([
            log_cell,
            row["algorithm"],
            row["parameters"],

            _fmt_exact(
                row["exact_fit"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_fit"],
                decimal_places,
            ),

            _fmt_number(
                row["sampling_fit"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_fit_error"],
                decimal_places,
            ),

            _fmt_number(
                row["sampling_fit_error"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_time"],
                decimal_places,
            ),

            _fmt_number(
                row["sampling_time"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_fit_pi"],
                decimal_places,
            ),

            _fmt_number(
                row["sampling_fit_pi"],
                decimal_places,
            ),
        ])

    # Final aggregate row.
    rows.append([
        "Mean / MAE",
        "",
        "",
        "",
        "",
        "",

        _fmt_number(
            evaluation_df[
                "ours_fit_error"
            ].mean(),
            decimal_places,
        ),

        _fmt_number(
            evaluation_df[
                "sampling_fit_error"
            ].mean(),
            decimal_places,
        ),

        _fmt_number(
            evaluation_df[
                "ours_time"
            ].mean(),
            decimal_places,
        ),

        _fmt_number(
            evaluation_df[
                "sampling_time"
            ].mean(),
            decimal_places,
        ),

        _fmt_number(
            evaluation_df[
                "ours_fit_pi"
            ].mean(),
            decimal_places,
        ),

        _fmt_number(
            evaluation_df[
                "sampling_fit_pi"
            ].mean(),
            decimal_places,
        ),
    ])

    columns = pd.MultiIndex.from_tuples([
        ("", "Log"),
        ("", "Algorithm"),
        ("", "Parameters"),
        ("", "Exact Fit."),

        ("Fitness", "Ours"),
        ("Fitness", "Sampling"),

        ("Approximation Error", "Ours"),
        ("Approximation Error", "Sampling"),

        ("Time (s)", "Ours"),
        ("Time (s)", "Sampling"),

        ("PI", "Ours"),
        ("PI", "Sampling"),
    ])

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def _precision_display_table(
    evaluation_df,
    log_name_map=None,
    decimal_places=3,
    log_wrap_width=18,
):
    decimal_places = _validate_decimal_places(
        decimal_places
    )

    log_wrap_width = _validate_wrap_width(
        log_wrap_width
    )

    rows = []
    previous_log = None

    for _, row in evaluation_df.iterrows():
        current_log = row["log_id"]

        log_cell = ""

        if current_log != previous_log:
            pretty_log = _pretty_log_name(
                current_log,
                log_name_map,
            )

            log_cell = _wrap_log_name(
                pretty_log,
                width=log_wrap_width,
            )

        previous_log = current_log

        rows.append([
            log_cell,
            row["algorithm"],
            row["parameters"],

            _fmt_exact(
                row["exact_prec"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_prec_error"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_prec_pi"],
                decimal_places,
            ),
        ])

    rows.append([
        "Mean / MAE",
        "",
        "",
        "",

        _fmt_number(
            evaluation_df[
                "ours_prec_error"
            ].mean(),
            decimal_places,
        ),

        _fmt_number(
            evaluation_df[
                "ours_prec_pi"
            ].mean(),
            decimal_places,
        ),
    ])

    return pd.DataFrame(
        rows,
        columns=[
            "Log",
            "Algorithm",
            "Parameters",
            "Exact Prec.",
            "Our Error",
            "Our PI",
        ],
    )


def _combined_display_table(
    evaluation_df,
    log_name_map=None,
    decimal_places=3,
    log_wrap_width=18,
):
    """
    Combined fitness + precision summary table.

    Structure:

        Fitness
            Exact
            Approx. Error
                Ours
                Sampling
            PI
                Ours
                Sampling

        Precision
            Exact
            Approx. Error
            PI

    Sampling precision is not available in the reference CSV, so
    precision contains only our approximation error and PI.
    """
    decimal_places = _validate_decimal_places(
        decimal_places
    )

    log_wrap_width = _validate_wrap_width(
        log_wrap_width
    )

    rows = []
    previous_log = None

    for _, row in evaluation_df.iterrows():
        current_log = row["log_id"]

        log_cell = ""

        if current_log != previous_log:
            pretty_log = _pretty_log_name(
                current_log,
                log_name_map,
            )

            log_cell = _wrap_log_name(
                pretty_log,
                width=log_wrap_width,
            )

        previous_log = current_log

        rows.append([
            # ---------------------------------------------
            # Identification
            # ---------------------------------------------
            log_cell,
            row["algorithm"],
            row["parameters"],

            # ---------------------------------------------
            # Fitness
            # ---------------------------------------------
            _fmt_exact(
                row["exact_fit"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_fit_error"],
                decimal_places,
            ),

            _fmt_number(
                row["sampling_fit_error"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_fit_pi"],
                decimal_places,
            ),

            _fmt_number(
                row["sampling_fit_pi"],
                decimal_places,
            ),

            # ---------------------------------------------
            # Precision
            # ---------------------------------------------
            _fmt_exact(
                row["exact_prec"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_prec_error"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_prec_pi"],
                decimal_places,
            ),
        ])

    # ========================================================
    # Aggregate row
    # ========================================================

    rows.append([
        "Mean / MAE",
        "",
        "",

        # Fitness exact
        "",

        # Fitness approximation error
        _fmt_number(
            evaluation_df[
                "ours_fit_error"
            ].mean(),
            decimal_places,
        ),

        _fmt_number(
            evaluation_df[
                "sampling_fit_error"
            ].mean(),
            decimal_places,
        ),

        # Fitness PI
        _fmt_number(
            evaluation_df[
                "ours_fit_pi"
            ].mean(),
            decimal_places,
        ),

        _fmt_number(
            evaluation_df[
                "sampling_fit_pi"
            ].mean(),
            decimal_places,
        ),

        # Precision exact
        "",

        # Precision approximation error
        _fmt_number(
            evaluation_df[
                "ours_prec_error"
            ].mean(),
            decimal_places,
        ),

        # Precision PI
        _fmt_number(
            evaluation_df[
                "ours_prec_pi"
            ].mean(),
            decimal_places,
        ),
    ])

    # Three header levels are needed because Fitness has
    # Ours/Sampling underneath Approx. Error and PI.
    columns = pd.MultiIndex.from_tuples([
        ("", "", "Log"),
        ("", "", "Algorithm"),
        ("", "", "Parameters"),

        ("Fitness", "Exact", ""),

        (
            "Fitness",
            "Approx. Error",
            "Ours",
        ),
        (
            "Fitness",
            "Approx. Error",
            "Sampling",
        ),

        (
            "Fitness",
            "PI",
            "Ours",
        ),
        (
            "Fitness",
            "PI",
            "Sampling",
        ),

        ("Precision", "Exact", ""),
        ("Precision", "Approx. Error", ""),
        ("Precision", "PI", ""),
    ])

    return pd.DataFrame(
        rows,
        columns=columns,
    )


# def _styled_table(
#     df,
# ):
#     """
#     Notebook-only styling.
#
#     Important:
#     - index is hidden
#     - wrapped log text stays inside the same dataframe cell
#     - all cells are top-aligned so a wrapped log label does not
#       visually shift the corresponding row values
#     """
#     return (
#         df.style
#         .hide(axis="index")
#         .set_table_styles([
#             {
#                 "selector": "th",
#                 "props": [
#                     ("text-align", "center"),
#                     ("vertical-align", "middle"),
#                 ],
#             },
#             {
#                 "selector": "td",
#                 "props": [
#                     ("text-align", "center"),
#                     ("vertical-align", "top"),
#                     ("white-space", "pre-line"),
#                 ],
#             },
#         ])
#     )
#
#

def _styled_table(
    df,
    heatmap=False,
    # heatmap_cmap="YlOrRd",
    heatmap_cmap = "Reds",
):
    """
    Notebook-only styling.

    Heatmap is optional and applied independently to each eligible column.

    Excluded from heatmap:
    - Log
    - Algorithm
    - Parameters
    - all Exact columns

    This affects only the notebook/Pandas display.
    It does NOT affect LaTeX generation.
    """

    styler = (
        df.style
        .hide(axis="index")
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("vertical-align", "middle"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "center"),
                    ("vertical-align", "top"),
                    ("white-space", "pre-line"),
                ],
            },
        ])
    )

    if not heatmap:
        return styler

    # --------------------------------------------------------
    # Determine whether a column should receive a heatmap
    # --------------------------------------------------------
    def should_heatmap(column):

        if isinstance(column, tuple):
            parts = [
                str(part).strip().lower()
                for part in column
                if str(part).strip()
            ]
        else:
            parts = [
                str(column).strip().lower()
            ]

        # Identification columns
        excluded_names = {
            "log",
            "algorithm",
            "parameters",
        }

        if any(
            part in excluded_names
            for part in parts
        ):
            return False

        # Exact columns:
        # Exact
        # Exact Fit.
        # Exact Prec.
        if any(
            part == "pi" or
            part == "exact"
            or part.startswith("exact ")
            for part in parts
        ):
            return False

        return True

    # --------------------------------------------------------
    # Apply heatmap independently to every eligible column
    # --------------------------------------------------------
    for column in df.columns:

        if not should_heatmap(column):
            continue

        # The display dataframe contains formatted strings,
        # so convert only for calculating heatmap intensity.
        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        # Skip columns with no numeric values.
        if not numeric_values.notna().any():
            continue

        styler = styler.background_gradient(
            cmap=heatmap_cmap,
            subset=[column],
            gmap=numeric_values,

            # IMPORTANT:
            # gmap is a Series, so use axis=0.
            axis=0,
        )

    return styler

# # ============================================================
# # 8. LaTeX helpers
# # ============================================================

def _latex_escape(text):
    text = str(text)

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }

    return "".join(
        replacements.get(char, char)
        for char in text
    )


def _latex_log_cell(
    name,
    width=18,
):
    """
    Wrap a long log name inside ONE LaTeX table cell.

    The important difference from the previous implementation:
    this uses a top-aligned parbox. Any internal line break belongs
    to the parbox, not to the outer tabular, so it cannot create an
    additional table row.

    No extra package is required.
    """
    wrapped = _wrap_log_name(
        name,
        width=width,
    )

    parts = [
        _latex_escape(part)
        for part in wrapped.split("\n")
    ]

    if len(parts) == 1:
        return parts[0]

    # Roughly convert character width to an em width.
    # This only controls the width of the cell; the actual wrapping
    # was already determined above.
    effective_width = (
        18
        if width is None or int(width) <= 0
        else int(width)
    )

    em_width = max(
        5.5,
        effective_width * 0.48,
    )

    body = r"\\".join(parts)

    return (
        rf"\parbox[t]{{{em_width:.1f}em}}"
        r"{\raggedright "
        + body
        + "}"
    )


def _latex_fitness_table(
    evaluation_df,
    log_name_map=None,
    caption=None,
    label="tab:real_fitness",
    decimal_places=3,
    log_wrap_width=18,
):
    decimal_places = _validate_decimal_places(
        decimal_places
    )

    log_wrap_width = _validate_wrap_width(
        log_wrap_width
    )

    if caption is None:
        caption = (
            "Fitness approximation on real-life log--model pairs. "
            "Lower approximation error is better; higher performance "
            "improvement (PI) is better."
        )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lll c cc cc cc cc}",
        r"\hline",
        (
            r"\textbf{Log} & "
            r"\textbf{Algorithm} & "
            r"\textbf{Parameters} & "
            r"\textbf{Exact Fit.} & "
            r"\multicolumn{2}{c}{\textbf{Fitness}} & "
            r"\multicolumn{2}{c}{\textbf{Approximation Error}} & "
            r"\multicolumn{2}{c}{\textbf{Time (s)}} & "
            r"\multicolumn{2}{c}{\textbf{PI}} \\"
        ),
        r"\cline{5-6}\cline{7-8}\cline{9-10}\cline{11-12}",
        (
            r"& & & & "
            r"\textbf{Ours} & "
            r"\textbf{Sampling} & "
            r"\textbf{Ours} & "
            r"\textbf{Sampling} & "
            r"\textbf{Ours} & "
            r"\textbf{Sampling} & "
            r"\textbf{Ours} & "
            r"\textbf{Sampling} \\"
        ),
        r"\hline",
    ]

    previous_log = None

    for _, row in evaluation_df.iterrows():
        current_log = row["log_id"]

        if (
            previous_log is not None
            and current_log != previous_log
        ):
            lines.append(r"\hline")

        log_cell = ""

        if current_log != previous_log:
            log_cell = _latex_log_cell(
                _pretty_log_name(
                    current_log,
                    log_name_map,
                ),
                width=log_wrap_width,
            )

        previous_log = current_log

        row_cells = [
            log_cell,
            _latex_escape(
                row["algorithm"]
            ),
            row["parameters_latex"],

            _fmt_exact(
                row["exact_fit"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_fit"],
                decimal_places,
            ),

            _fmt_number(
                row["sampling_fit"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_fit_error"],
                decimal_places,
            ),

            _fmt_number(
                row["sampling_fit_error"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_time"],
                decimal_places,
            ),

            _fmt_number(
                row["sampling_time"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_fit_pi"],
                decimal_places,
            ),

            _fmt_number(
                row["sampling_fit_pi"],
                decimal_places,
            ),
        ]

        # Exactly ONE outer-tabular row terminator.
        lines.append(
            " & ".join(row_cells)
            + r" \\"
        )

    lines.extend([
        r"\hline",
        (
            r"\multicolumn{6}{r}{\textbf{Mean / MAE}} & "
            + r"\textbf{"
            + _fmt_number(
                evaluation_df[
                    "ours_fit_error"
                ].mean(),
                decimal_places,
            )
            + "} & "
            + r"\textbf{"
            + _fmt_number(
                evaluation_df[
                    "sampling_fit_error"
                ].mean(),
                decimal_places,
            )
            + "} & "
            + r"\textbf{"
            + _fmt_number(
                evaluation_df[
                    "ours_time"
                ].mean(),
                decimal_places,
            )
            + "} & "
            + r"\textbf{"
            + _fmt_number(
                evaluation_df[
                    "sampling_time"
                ].mean(),
                decimal_places,
            )
            + "} & "
            + r"\textbf{"
            + _fmt_number(
                evaluation_df[
                    "ours_fit_pi"
                ].mean(),
                decimal_places,
            )
            + "} & "
            + r"\textbf{"
            + _fmt_number(
                evaluation_df[
                    "sampling_fit_pi"
                ].mean(),
                decimal_places,
            )
            + r"} \\"
        ),
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    return "\n".join(lines)




def _latex_precision_table(
    evaluation_df,
    log_name_map=None,
    caption=None,
    label="tab:real_precision",
    decimal_places=3,
    log_wrap_width=18,
):
    decimal_places = _validate_decimal_places(
        decimal_places
    )

    log_wrap_width = _validate_wrap_width(
        log_wrap_width
    )

    if caption is None:
        caption = (
            "Precision approximation on real-life log--model pairs. "
            "Lower approximation error is better; higher PI is better."
        )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lllccc}",
        r"\hline",
        (
            r"\textbf{Log} & "
            r"\textbf{Algorithm} & "
            r"\textbf{Parameters} & "
            r"\textbf{Exact Prec.} & "
            r"\textbf{Our Error} & "
            r"\textbf{Our PI} \\"
        ),
        r"\hline",
    ]

    previous_log = None

    for _, row in evaluation_df.iterrows():
        current_log = row["log_id"]

        if (
            previous_log is not None
            and current_log != previous_log
        ):
            lines.append(r"\hline")

        log_cell = ""

        if current_log != previous_log:
            log_cell = _latex_log_cell(
                _pretty_log_name(
                    current_log,
                    log_name_map,
                ),
                width=log_wrap_width,
            )

        previous_log = current_log

        row_cells = [
            log_cell,
            _latex_escape(
                row["algorithm"]
            ),
            row["parameters_latex"],

            _fmt_exact(
                row["exact_prec"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_prec_error"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_prec_pi"],
                decimal_places,
            ),
        ]

        lines.append(
            " & ".join(row_cells)
            + r" \\"
        )

    lines.extend([
        r"\hline",
        (
            r"\multicolumn{4}{r}{\textbf{Mean / MAE}} & "
            + r"\textbf{"
            + _fmt_number(
                evaluation_df[
                    "ours_prec_error"
                ].mean(),
                decimal_places,
            )
            + "} & "
            + r"\textbf{"
            + _fmt_number(
                evaluation_df[
                    "ours_prec_pi"
                ].mean(),
                decimal_places,
            )
            + r"} \\"
        ),
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    return "\n".join(lines)

def _latex_combined_table(
    evaluation_df,
    log_name_map=None,
    caption=None,
    label="tab:real_combined",
    decimal_places=3,
    log_wrap_width=18,
):
    """
    Combined fitness + precision LaTeX table.

    Columns:

        Log
        Algorithm
        Parameters

        Fitness:
            Exact
            Approx. Error:
                Ours
                Sampling
            PI:
                Ours
                Sampling

        Precision:
            Exact
            Approx. Error
            PI
    """
    decimal_places = _validate_decimal_places(
        decimal_places
    )

    log_wrap_width = _validate_wrap_width(
        log_wrap_width
    )

    if caption is None:
        caption = (
            "Fitness and precision approximation on real-life "
            "log--model pairs. Lower approximation error is better; "
            "higher performance improvement (PI) is better."
        )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",

        # 11 columns:
        # 3 identification
        # 5 fitness
        # 3 precision
        r"\begin{tabular}{lll c cc cc c c c}",

        r"\hline",

        # ====================================================
        # Header row 1
        # ====================================================
        (
            r"\textbf{Log} & "
            r"\textbf{Algorithm} & "
            r"\textbf{Parameters} & "
            r"\multicolumn{5}{c}{\textbf{Fitness}} & "
            r"\multicolumn{3}{c}{\textbf{Precision}} \\"
        ),

        # r"\cline{4-8}\cline{9-11}",
        # r"\cmidrule(r{0.03em}){4-8}\cmidrule(l{0.1em}){9-11}",
        r"\cmidrule(r{0.3em}){4-8}\cmidrule(l{0.3em}){9-11}",
        # ====================================================
        # Header row 2
        # ====================================================
        (
            r"& & & "
            r"\textbf{Exact} & "
            r"\multicolumn{2}{c}{\textbf{Approx. Error}} & "
            r"\multicolumn{2}{c}{\textbf{PI}} & "
            r"\textbf{Exact} & "
            r"\textbf{Approx. Error} & "
            r"\textbf{PI} \\"
        ),

        # r"\cline{5-6}\cline{7-8}",
        # r"\cmidrule(r{0.03em}){5-6}\cmidrule(l{0.1em}){7-8}",
        r"\cmidrule(r{0.2em}){5-6}\cmidrule(l{0.2em}r{0.3em}){7-8}",

        # ====================================================
        # Header row 3
        # ====================================================
        (
            r"& & & & "
            r"\textbf{Ours} & "
            r"\textbf{Sampling} & "
            r"\textbf{Ours} & "
            r"\textbf{Sampling} & "
            r"& & \\"
        ),

        r"\hline",
    ]

    previous_log = None

    # ========================================================
    # Data rows
    # ========================================================

    for _, row in evaluation_df.iterrows():
        current_log = row["log_id"]

        if (
            previous_log is not None
            and current_log != previous_log
        ):
            lines.append(r"\hline")

        log_cell = ""

        if current_log != previous_log:
            log_cell = _latex_log_cell(
                _pretty_log_name(
                    current_log,
                    log_name_map,
                ),
                width=log_wrap_width,
            )

        previous_log = current_log

        row_cells = [
            # ------------------------------------------------
            # Identification
            # ------------------------------------------------
            log_cell,

            _latex_escape(
                row["algorithm"]
            ),

            row["parameters_latex"],

            # ------------------------------------------------
            # Fitness
            # ------------------------------------------------
            _fmt_exact(
                row["exact_fit"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_fit_error"],
                decimal_places,
            ),

            _fmt_number(
                row["sampling_fit_error"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_fit_pi"],
                decimal_places,
            ),

            _fmt_number(
                row["sampling_fit_pi"],
                decimal_places,
            ),

            # ------------------------------------------------
            # Precision
            # ------------------------------------------------
            _fmt_exact(
                row["exact_prec"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_prec_error"],
                decimal_places,
            ),

            _fmt_number(
                row["ours_prec_pi"],
                decimal_places,
            ),
        ]

        lines.append(
            " & ".join(row_cells)
            + r" \\"
        )

    # ========================================================
    # Mean / MAE row
    # ========================================================

    lines.extend([
        r"\hline",
        (
            # First 4 columns:
            # Log + Algorithm + Parameters + Fitness Exact
            r"\multicolumn{4}{r}{\textbf{Mean / MAE}} & "

            # Fitness error
            + r"\textbf{"
            + _fmt_number(
                evaluation_df[
                    "ours_fit_error"
                ].mean(),
                decimal_places,
            )
            + "} & "

            + r"\textbf{"
            + _fmt_number(
                evaluation_df[
                    "sampling_fit_error"
                ].mean(),
                decimal_places,
            )
            + "} & "

            # Fitness PI
            + r"\textbf{"
            + _fmt_number(
                evaluation_df[
                    "ours_fit_pi"
                ].mean(),
                decimal_places,
            )
            + "} & "

            + r"\textbf{"
            + _fmt_number(
                evaluation_df[
                    "sampling_fit_pi"
                ].mean(),
                decimal_places,
            )
            + "} & "

            # Precision exact
            + r" & "

            # Precision error
            + r"\textbf{"
            + _fmt_number(
                evaluation_df[
                    "ours_prec_error"
                ].mean(),
                decimal_places,
            )
            + "} & "

            # Precision PI
            + r"\textbf{"
            + _fmt_number(
                evaluation_df[
                    "ours_prec_pi"
                ].mean(),
                decimal_places,
            )
            + r"} \\"
        ),

        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    return "\n".join(lines)


# ============================================================
# 9. EXPENSIVE STEP: compute once
# ============================================================

# def compute_results_target(
#     model_obj,
#     reference_csv,
#     repeats=1,
#     graph_module_name="graph",
# ):
#     """
#     EXPENSIVE STEP.
#
#     Run this once in the notebook.
#
#     It:
#       1. reads the CSV
#       2. removes Alpha Miner rows
#       3. builds the log/model records from log_path/model_path
#       4. runs model inference + timing
#       5. computes errors and PI
#
#     It does NOT do presentation/layout formatting.
#
#     Returns:
#         {
#             "raw": evaluation_df,
#             "reference": reference_df,
#             "prediction_raw": results_df,
#         }
#     """
#     reference_df = (
#         load_real_life_reference_csv(
#             reference_csv
#         )
#     )
#
#     records = (
#         _prepare_records_from_reference(
#             reference_df
#         )
#     )
#
#     model_obj.eval()
#
#     with torch.inference_mode():
#         results_df = predict_records_timed(
#             model=model_obj,
#             records=records,
#             variant="A",
#             repeats=repeats,
#             graph_module_name=graph_module_name,
#         )
#
#     evaluation_df = _build_evaluation_df(
#         results_df=results_df,
#         reference_df=reference_df,
#     )
#
#     return {
#         "raw":
#             evaluation_df,
#
#         "reference":
#             reference_df,
#
#         "prediction_raw":
#             results_df,
#     }

def compute_results_target(
    model_obj,
    reference_csv,
    repeats=1,
    graph_module_name="graph",
):
    """
    EXPENSIVE STEP.

    Run inference and timing once.

    Both mean and median runtime statistics are retained.

    The choice between mean and median for reporting and PI
    is made later in show_computed_results().

    It:
      1. reads the CSV
      2. removes Alpha Miner rows
      3. builds log/model records
      4. runs model inference + repeated timing
      5. computes prediction errors

    It does NOT choose mean vs median for the final tables.

    Returns:
        {
            "raw": evaluation_df,
            "reference": reference_df,
            "prediction_raw": results_df,
        }
    """

    # ========================================================
    # Load reference results
    # ========================================================

    reference_df = (
        load_real_life_reference_csv(
            reference_csv
        )
    )

    # ========================================================
    # Build inference records
    # ========================================================

    records = (
        _prepare_records_from_reference(
            reference_df
        )
    )

    # ========================================================
    # Inference
    # ========================================================

    model_obj.eval()

    with torch.inference_mode():

        results_df = predict_records_timed(
            model=model_obj,
            records=records,
            variant="A",
            repeats=repeats,
            graph_module_name=graph_module_name,
        )

    # ========================================================
    # Build evaluation dataframe
    # ========================================================

    evaluation_df = _build_evaluation_df(
        results_df=results_df,
        reference_df=reference_df,
    )

    # ========================================================
    # Return
    # ========================================================

    return {
        "raw":
            evaluation_df,

        "reference":
            reference_df,

        "prediction_raw":
            results_df,
    }

# ============================================================
# Spearman correlation helper functions
# ============================================================

def _spearman_correlation(
    x,
    y,
):
    """
    Spearman rank correlation with pairwise NaN removal.

    Returns NaN when:
      - fewer than 2 valid pairs exist
      - either side is constant

    Ties are handled using average ranks.
    """

    pair = pd.DataFrame({
        "x": pd.to_numeric(
            x,
            errors="coerce",
        ),
        "y": pd.to_numeric(
            y,
            errors="coerce",
        ),
    }).dropna()

    if len(pair) < 2:
        return np.nan

    if pair["x"].nunique() < 2:
        return np.nan

    if pair["y"].nunique() < 2:
        return np.nan

    x_rank = pair["x"].rank(
        method="average"
    )

    y_rank = pair["y"].rank(
        method="average"
    )

    return float(
        x_rank.corr(
            y_rank,
            method="pearson",
        )
    )


def _build_spearman_table(
    evaluation_df,
    log_name_map=None,
    decimal_places=3,
):
    """
    Build notebook-only Spearman ranking table.

    Rows:
      - Overall (all pairs)
      - one row per log
      - Mean per-log

    Columns:
      - Fitness: Ours
      - Fitness: Sampling
      - Precision: Ours
    """

    rows = []

    # ========================================================
    # 1. Overall Spearman across ALL log/model pairs
    # ========================================================

    overall_fit_ours = _spearman_correlation(
        evaluation_df["exact_fit"],
        evaluation_df["ours_fit"],
    )

    overall_fit_sampling = _spearman_correlation(
        evaluation_df["exact_fit"],
        evaluation_df["sampling_fit"],
    )

    overall_prec_ours = _spearman_correlation(
        evaluation_df["exact_prec"],
        evaluation_df["ours_prec"],
    )

    rows.append({
        "Scope":
            "Overall (all pairs)",

        "Fitness — Ours":
            overall_fit_ours,

        "Fitness — Sampling":
            overall_fit_sampling,

        "Precision — Ours":
            overall_prec_ours,
    })

    # ========================================================
    # 2. Spearman separately inside each log
    # ========================================================

    per_log_rows = []

    for log_id, group in evaluation_df.groupby(
        "log_id",
        sort=False,
    ):

        fit_ours = _spearman_correlation(
            group["exact_fit"],
            group["ours_fit"],
        )

        fit_sampling = _spearman_correlation(
            group["exact_fit"],
            group["sampling_fit"],
        )

        prec_ours = _spearman_correlation(
            group["exact_prec"],
            group["ours_prec"],
        )

        pretty_log = _pretty_log_name(
            log_id,
            log_name_map,
        )

        per_log_rows.append({
            "Scope":
                pretty_log,

            "Fitness — Ours":
                fit_ours,

            "Fitness — Sampling":
                fit_sampling,

            "Precision — Ours":
                prec_ours,
        })

    rows.extend(
        per_log_rows
    )

    # ========================================================
    # 3. Arithmetic mean of per-log Spearman values
    #
    # NaN logs are automatically ignored.
    # ========================================================

    per_log_df = pd.DataFrame(
        per_log_rows
    )

    rows.append({
        "Scope":
            "Mean per-log",

        "Fitness — Ours":
            per_log_df[
                "Fitness — Ours"
            ].mean(),

        "Fitness — Sampling":
            per_log_df[
                "Fitness — Sampling"
            ].mean(),

        "Precision — Ours":
            per_log_df[
                "Precision — Ours"
            ].mean(),
    })

    # ========================================================
    # Format for notebook display
    # ========================================================

    table = pd.DataFrame(
        rows
    )

    for col in [
        "Fitness — Ours",
        "Fitness — Sampling",
        "Precision — Ours",
    ]:
        table[col] = table[col].map(
            lambda value:
                "--"
                if pd.isna(value)
                else f"{float(value):.{decimal_places}f}"
        )

    return table

# ============================================================
# 10. CHEAP STEP: show/reformat computed results
# ============================================================

def show_computed_results(
    results,
    target="both",

    time_aggregation="median",

    show_fitness_plot=False,
    fitness_decimal_places=3,
    fitness_log_wrap_width=25,

    show_precision_plot=False,
    precision_decimal_places=3,
    precision_log_wrap_width=15,

    show_combined_plot=True,
    combined_decimal_places=3,
    combined_log_wrap_width=15,

    log_name_map=None,

    fitness_heatmap=False,
    precision_heatmap=False,
    combined_heatmap=True,

    show_spearman_table=True,
    spearman_decimal_places=3,

    generate_latex=False,
    latex_output_dir=None,
    print_latex=False,

    full_test_dataset=False,

    save_table=False,
    output_path="prediction_results.xlsx",
):
    """
    CHEAP STEP.

    Reuse this as many times as you want after inference.

    time_aggregation:
        "median" -> use median repeated inference time
        "mean"   -> use mean repeated inference time

    This choice controls:
        - ours_time shown in the fitness table
        - fitness PI
        - precision PI
        - LaTeX tables
        - exported tables

    No model inference, graph construction, XES loading,
    or PNML loading happens here.
    """

    # ========================================================
    # Validate target
    # ========================================================

    target = str(target).lower()

    if target not in {
        "both",
        "fitness",
        "precision",
    }:
        raise ValueError(
            'target must be "both", "fitness", or "precision"'
        )

    # ========================================================
    # Validate time aggregation
    # ========================================================

    time_aggregation = (
        str(time_aggregation)
        .strip()
        .lower()
    )

    if time_aggregation not in {
        "mean",
        "median",
    }:
        raise ValueError(
            "time_aggregation must be 'mean' or 'median'"
        )

    # ========================================================
    # Get evaluation dataframe
    # ========================================================

    if isinstance(results, pd.DataFrame):

        evaluation_df = results.copy()

    elif (
        isinstance(results, dict)
        and "raw" in results
    ):

        evaluation_df = results["raw"].copy()

    else:

        raise TypeError(
            "results must be either the dictionary returned by "
            "compute_results_target()/show_results_target(), "
            "or the raw evaluation DataFrame."
        )

    if not full_test_dataset:
        mask_rows = (
                evaluation_df["parameters"].str.contains("d=0.7", case=False, na=False, ) |
                evaluation_df["parameters"].str.contains("d=0.9", case=False, na=False, ) |
                (evaluation_df["log_id"] == "BPIC-2012.xes")
        )

        evaluation_df = evaluation_df.loc[~mask_rows].copy()
    # ========================================================
    # Select mean or median runtime
    # ========================================================

    if time_aggregation == "median":

        time_column = "ours_time_median"

    else:

        time_column = "ours_time_mean"

    if time_column not in evaluation_df.columns:
        raise ValueError(
            f"Missing required timing column: {time_column}. "
            "Recompute results using the updated inference code."
        )

    evaluation_df["ours_time"] = (
        evaluation_df[time_column]
    )

    # ========================================================
    # Recompute PI using selected runtime
    #
    # PI = T_exact / T_ours
    # ========================================================

    evaluation_df["ours_fit_pi"] = (
        _performance_improvement(
            evaluation_df["exact_fit_time"],
            evaluation_df["ours_time"],
        )
    )

    evaluation_df["ours_prec_pi"] = (
        _performance_improvement(
            evaluation_df["exact_prec_time"],
            evaluation_df["ours_time"],
        )
    )

    # Sampling PI is independent of mean/median choice.
    evaluation_df["sampling_fit_pi"] = (
        _performance_improvement(
            evaluation_df["exact_fit_time"],
            evaluation_df["sampling_time"],
        )
    )

    # ========================================================
    # Output holders
    # ========================================================

    fitness_table = None
    precision_table = None
    combined_table = None

    spearman_table = None

    fitness_latex = None
    precision_latex = None
    combined_latex = None

    # --------------------------------------------------------
    # Fitness notebook table
    # --------------------------------------------------------

    if show_fitness_plot and target in {
        "both",
        "fitness",
    }:

        fitness_table = (
            _fitness_display_table(
                evaluation_df,
                log_name_map=log_name_map,
                decimal_places=fitness_decimal_places,
                log_wrap_width=fitness_log_wrap_width,
            )
        )

        display(
            _styled_table(
                fitness_table,
                heatmap=fitness_heatmap,
            )
        )

    # --------------------------------------------------------
    # Precision notebook table
    # --------------------------------------------------------

    if show_precision_plot and target in {
        "both",
        "precision",
    }:

        precision_table = (
            _precision_display_table(
                evaluation_df,
                log_name_map=log_name_map,
                decimal_places=precision_decimal_places,
                log_wrap_width=precision_log_wrap_width,
            )
        )

        display(
            _styled_table(
                precision_table,
                heatmap=precision_heatmap,
            )
        )

    # --------------------------------------------------------
    # Combined notebook table
    # --------------------------------------------------------

    if show_combined_plot and target == "both":

        combined_table = (
            _combined_display_table(
                evaluation_df,
                log_name_map=log_name_map,
                decimal_places=combined_decimal_places,
                log_wrap_width=combined_log_wrap_width,
            )
        )

        display(
            _styled_table(
                combined_table,
                heatmap=combined_heatmap,
            )
        )

    # ============================================================
    # Spearman ranking table
    # ============================================================

    if show_spearman_table:
        spearman_table = (
            _build_spearman_table(
                evaluation_df,
                log_name_map=log_name_map,
                decimal_places=spearman_decimal_places,
            )
        )

        display(
            _styled_table(
                spearman_table,
                heatmap=False,
            )
        )

    # --------------------------------------------------------
    # LaTeX generation
    # --------------------------------------------------------

    if generate_latex:

        if target in {
            "both",
            "fitness",
        }:

            fitness_latex = (
                _latex_fitness_table(
                    evaluation_df,
                    log_name_map=log_name_map,
                    decimal_places=fitness_decimal_places,
                    log_wrap_width=fitness_log_wrap_width,
                )
            )

        if target in {
            "both",
            "precision",
        }:

            precision_latex = (
                _latex_precision_table(
                    evaluation_df,
                    log_name_map=log_name_map,
                    decimal_places=precision_decimal_places,
                    log_wrap_width=precision_log_wrap_width,
                )
            )

        if target == "both":

            combined_latex = (
                _latex_combined_table(
                    evaluation_df,
                    log_name_map=log_name_map,
                    decimal_places=combined_decimal_places,
                    log_wrap_width=combined_log_wrap_width,
                )
            )

        # ----------------------------------------------------
        # Save LaTeX
        # ----------------------------------------------------

        if latex_output_dir is not None:

            latex_output_dir = Path(
                latex_output_dir
            )

            latex_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            if fitness_latex is not None:

                fitness_path = (
                    latex_output_dir
                    / "real_fitness.tex"
                )

                fitness_path.write_text(
                    fitness_latex,
                    encoding="utf-8",
                )

                print(
                    "Saved fitness LaTeX table to: "
                    f"{fitness_path.resolve()}"
                )

            if precision_latex is not None:

                precision_path = (
                    latex_output_dir
                    / "real_precision.tex"
                )

                precision_path.write_text(
                    precision_latex,
                    encoding="utf-8",
                )

                print(
                    "Saved precision LaTeX table to: "
                    f"{precision_path.resolve()}"
                )

            if combined_latex is not None:

                combined_path = (
                    latex_output_dir
                    / "real_combined.tex"
                )

                combined_path.write_text(
                    combined_latex,
                    encoding="utf-8",
                )

                print(
                    "Saved combined LaTeX table to: "
                    f"{combined_path.resolve()}"
                )

        # ----------------------------------------------------
        # Print LaTeX
        # ----------------------------------------------------

        if print_latex:

            if fitness_latex is not None:

                print(
                    "\n===== FITNESS LATEX =====\n"
                )

                print(fitness_latex)

            if precision_latex is not None:

                print(
                    "\n===== PRECISION LATEX =====\n"
                )

                print(precision_latex)

            if combined_latex is not None:

                print(
                    "\n===== COMBINED LATEX =====\n"
                )

                print(combined_latex)

    # --------------------------------------------------------
    # Optional Excel export
    # --------------------------------------------------------

    if save_table:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            with pd.ExcelWriter(
                output_path
            ) as writer:

                evaluation_df.to_excel(
                    writer,
                    sheet_name="Raw results",
                    index=False,
                )

                if fitness_table is not None:

                    fitness_table.to_excel(
                        writer,
                        sheet_name="Fitness",
                        index=False,
                    )

                if precision_table is not None:

                    precision_table.to_excel(
                        writer,
                        sheet_name="Precision",
                        index=False,
                    )

                if combined_table is not None:

                    combined_table.to_excel(
                        writer,
                        sheet_name="Combined",
                        index=False,
                    )

        except ModuleNotFoundError as exc:

            if exc.name == "openpyxl":

                raise ModuleNotFoundError(
                    "Excel export requires openpyxl. "
                    "Install it with `%pip install openpyxl`, "
                    "or call show_computed_results(..., "
                    "save_table=False)."
                ) from exc

            raise

        print(
            "Saved Excel workbook to: "
            f"{output_path.resolve()}"
        )

    # ========================================================
    # Return
    # ========================================================

    return {
        "raw":
            evaluation_df,

        "time_aggregation":
            time_aggregation,

        "fitness_table":
            fitness_table,

        "precision_table":
            precision_table,

        "combined_table":
            combined_table,

        "spearman_table":
            spearman_table,

        "fitness_latex":
            fitness_latex,

        "precision_latex":
            precision_latex,

        "combined_latex":
            combined_latex,
    }


# ============================================================
# 11. Convenience wrapper: compute + show
# ============================================================

def show_results_target(
    model_obj,
    reference_csv,
    target="both",
    repeats=1,
    graph_module_name="graph",

    fitness_decimal_places=3,
    fitness_log_wrap_width=18,

    precision_decimal_places=3,
    precision_log_wrap_width=18,

    combined_decimal_places=3,
    combined_log_wrap_width=18,

    log_name_map=None,

    generate_latex=False,
    latex_output_dir=None,
    print_latex=False,

    save_table=False,
    output_path="prediction_results.xlsx",
):
    """
    Convenience function for the old one-call workflow.

    IMPORTANT:
    Calling this function reruns inference.

    For notebook work where layout changes frequently, prefer:

        results = compute_results_target(...)
        show_computed_results(results, ...)

    Then only rerun show_computed_results() when changing layout.
    """
    results = compute_results_target(
        model_obj=model_obj,
        reference_csv=reference_csv,
        repeats=repeats,
        graph_module_name=graph_module_name,
    )

    shown = show_computed_results(
        results=results,
        target=target,

        fitness_decimal_places=
            fitness_decimal_places,

        fitness_log_wrap_width=
            fitness_log_wrap_width,

        precision_decimal_places=
            precision_decimal_places,

        precision_log_wrap_width=
            precision_log_wrap_width,

        combined_decimal_places=combined_decimal_places,

        combined_log_wrap_width=combined_log_wrap_width,

        log_name_map=
            log_name_map,

        generate_latex=
            generate_latex,

        latex_output_dir=
            latex_output_dir,

        print_latex=
            print_latex,

        save_table=
            save_table,

        output_path=
            output_path,
    )

    # Keep the expensive raw outputs too.
    shown["reference"] = (
        results["reference"]
    )

    shown["prediction_raw"] = (
        results["prediction_raw"]
    )

    return shown