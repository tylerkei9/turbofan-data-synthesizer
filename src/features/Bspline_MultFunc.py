"""
B-Spline Interpolation Feature
================================
Identifies clusters/gaps in synthetic data and adds intermediate data points
using B-spline interpolation to create smoother, more continuous datasets.

This script is part of the Rolls-Royce synthetic data generation pipeline.
It is intended to be used alongside the linear interpolation script, providing
an alternative method that produces smoother curves between cluster boundaries.

Location: src/features/spline_interpolation.py

Usage
-----
    from src.features.spline_interpolation import apply_interpolation

    synth_data = pd.read_csv("synthetic_data.csv")

    # Spline interpolation on all numeric columns
    result = apply_interpolation(synth_data, method="spline")

    # Spline interpolation on specific columns
    result = apply_interpolation(
        synth_data,
        method="spline",
        target_columns=["sensor_4", "sensor_9"]
    )
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from typing import Optional, Literal


# ---------------------------------------------------------------------------
# Cluster Detection
# ---------------------------------------------------------------------------

def detect_clusters(
    x: np.ndarray,
    gap_threshold: float = 1.0
) -> list[tuple[int, int]]:
    """
    Detect clusters in data based on spacing between sorted x-values.

    A gap is defined as any consecutive pair of x-values whose difference
    exceeds ``gap_threshold``. Each contiguous run of points between gaps
    is treated as one cluster.

    Parameters
    ----------
    x : np.ndarray
        Sorted 1-D array of independent variable values (e.g. time index).
    gap_threshold : float, default=1.0
        Minimum spacing between consecutive x-values that constitutes a gap.
        Points separated by more than this value belong to different clusters.

    Returns
    -------
    list of tuple[int, int]
        Each tuple is ``(start_index, end_index)`` marking the first and last
        row indices of a cluster within the sorted array.

    Examples
    --------
    >>> x = np.array([0, 1, 2, 10, 11, 12])
    >>> detect_clusters(x, gap_threshold=1.0)
    [(0, 2), (3, 5)]
    """
    if len(x) == 0:
        return []

    gap_positions = np.where(np.diff(x) > gap_threshold)[0]

    clusters: list[tuple[int, int]] = []
    start = 0
    for gap in gap_positions:
        clusters.append((start, int(gap)))
        start = int(gap) + 1
    clusters.append((start, len(x) - 1))

    return clusters


# ---------------------------------------------------------------------------
# Interpolation Path Generation
# ---------------------------------------------------------------------------

def generate_interpolation_path(
    x_values: np.ndarray,
    y_values: np.ndarray,
    num_points: int = 10
) -> pd.DataFrame:
    """
    Generate B-spline interpolated points between two cluster boundary values.

    Uses a cubic B-spline (degree ``k=3``) when four or more anchor points
    are supplied, and falls back to a linear spline (``k=1``) for smaller
    inputs. Both endpoints are excluded from the output so that the returned
    rows represent only the *new* points inserted inside the gap.

    Parameters
    ----------
    x_values : np.ndarray
        Boundary x-values derived from the edges of adjacent clusters.
        Must contain at least 2 points (the gap endpoints).
    y_values : np.ndarray
        Corresponding boundary y-values, same length as ``x_values``.
    num_points : int, default=10
        Number of new interpolation points to generate inside the gap.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["x_interp", "y_interp"]`` containing the
        newly generated interpolation points (endpoints excluded).

    Raises
    ------
    ValueError
        If ``x_values`` or ``y_values`` contain fewer than 2 points.

    Notes
    -----
    B-spline fitting requires strictly increasing x-values. If duplicate
    x-values are present the function deduplicates them before fitting.

    Examples
    --------
    >>> x = np.array([0.0, 5.0])
    >>> y = np.array([0.0, 25.0])
    >>> generate_interpolation_path(x, y, num_points=4)
    """
    if len(x_values) < 2 or len(y_values) < 2:
        raise ValueError(
            "x_values and y_values must each contain at least 2 points."
        )

    # Remove duplicate x entries to satisfy splrep's strictly-increasing req.
    _, unique_idx = np.unique(x_values, return_index=True)
    x_values = x_values[unique_idx]
    y_values = y_values[unique_idx]

    # New x positions inside the gap (endpoints stripped off)
    x_new = np.linspace(x_values[0], x_values[-1], num_points + 2)[1:-1]

    # Degree: cubic when we have enough knots, otherwise linear
    k = 3 if len(x_values) >= 4 else 1
    tck = interpolate.splrep(x_values, y_values, s=0, k=k)
    spline = interpolate.BSpline(*tck)
    y_new = spline(x_new)

    return pd.DataFrame({"x_interp": x_new, "y_interp": y_new})


# ---------------------------------------------------------------------------
# Main Public Interface
# ---------------------------------------------------------------------------

def apply_interpolation(
    data: pd.DataFrame,
    x_column: str,
    method: Literal["spline"] = "spline",
    target_columns: Optional[list] = None,
    gap_threshold: float = 1.0,
    num_points: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply B-spline interpolation between detected clusters in synthetic data.

    Identifies gaps between clusters using ``detect_clusters``, then fills
    each gap with ``num_points`` new rows generated via B-spline interpolation.
    The function returns **both** the original DataFrame and an augmented
    DataFrame so that callers can compare them or pass either downstream.

    Parameters
    ----------
    data : pd.DataFrame
        The synthetic dataset to interpolate. Must contain at least one
        numeric column and the column specified by ``x_column``.
    x_column : str
        Name of the column to use as the independent variable (e.g.
        ``"Time_in_cycles"``). The DataFrame is sorted by this column
        before processing.
    method : {"spline"}, default="spline"
        Interpolation method. Currently only ``"spline"`` (B-spline) is
        supported in this script. Linear interpolation is handled by the
        companion ``linear_interpolation.py`` module.
    target_columns : list of str, optional
        Columns to interpolate. If ``None``, all numeric columns except
        ``x_column`` are used.
    gap_threshold : float, default=1.0
        Passed directly to ``detect_clusters``. Consecutive x-values
        differing by more than this are treated as belonging to separate
        clusters.
    num_points : int, default=10
        Number of new data points to insert inside each detected gap.

    Returns
    -------
    original_df : pd.DataFrame
        An unmodified copy of the input data, sorted by ``x_column``.
    interpolated_df : pd.DataFrame
        A new DataFrame containing both the original rows and the newly
        generated interpolation rows, sorted by ``x_column`` and with the
        index reset. A boolean column ``"is_interpolated"`` marks which
        rows were synthetically generated (``True``) versus original
        (``False``).

    Raises
    ------
    ValueError
        If ``x_column`` is not present in ``data``, if no numeric target
        columns are found, or if an unsupported method is requested.

    Notes
    -----
    - Only the *boundary* points of adjacent clusters are used as anchor
      values for each individual spline fit. For richer curve shapes across
      a cluster consider passing a wider window of anchor points.
    - Columns not listed in ``target_columns`` are set to ``NaN`` in the
      interpolated rows and can be handled by downstream pipeline steps.

    Examples
    --------
    >>> import pandas as pd
    >>> from src.features.spline_interpolation import apply_interpolation

    >>> synth_data = pd.read_csv("synthetic_data.csv")

    >>> original, result = apply_interpolation(
    ...     synth_data,
    ...     x_column="Time_in_cycles",
    ...     method="spline",
    ...     gap_threshold=1.0,
    ...     num_points=10
    ... )

    >>> original, result = apply_interpolation(
    ...     synth_data,
    ...     x_column="Time_in_cycles",
    ...     method="spline",
    ...     target_columns=["sensor_4", "sensor_9"],
    ...     num_points=15
    ... )

    >>> print(f"Original rows : {len(original)}")
    >>> print(f"After interp  : {len(result)}")
    >>> print(result[result["is_interpolated"]].head())
    """
    if method != "spline":
        raise ValueError(
            f"This module only supports method='spline'. "
            f"Received: '{method}'. "
            f"For linear interpolation use src.features.linear_interpolation."
        )

    if x_column not in data.columns:
        raise ValueError(
            f"x_column '{x_column}' not found in DataFrame. "
            f"Available columns: {list(data.columns)}"
        )

    # ------------------------------------------------------------------ #
    # 1. Prepare working copy and original copy                          #
    # ------------------------------------------------------------------ #
    original_df = (
        data.copy()
        .sort_values(by=x_column)
        .reset_index(drop=True)
    )
    original_df["is_interpolated"] = False

    df = original_df.copy()

    # ------------------------------------------------------------------ #
    # 2. Resolve target columns                                          #
    # ------------------------------------------------------------------ #
    if target_columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_columns = [c for c in numeric_cols if c != x_column]

    if len(target_columns) == 0:
        raise ValueError(
            "No numeric target columns found. "
            "Ensure the DataFrame contains numeric data or specify "
            "target_columns explicitly."
        )

    # ------------------------------------------------------------------ #
    # 3. Detect clusters                                                 #
    # ------------------------------------------------------------------ #
    x = df[x_column].to_numpy()
    clusters = detect_clusters(x, gap_threshold)

    if len(clusters) < 2:
        print(
            "[spline_interpolation] No gaps detected with "
            f"gap_threshold={gap_threshold}. Returning original data."
        )
        # Return original without the helper column stripped
        clean_original = original_df.drop(columns=["is_interpolated"])
        return clean_original, original_df

    print(
        f"[spline_interpolation] Detected {len(clusters)} cluster(s) — "
        f"interpolating across {len(clusters) - 1} gap(s) "
        f"with {num_points} point(s) each."
    )

    # ------------------------------------------------------------------ #
    # 4. Generate interpolation rows for each gap                        #
    # ------------------------------------------------------------------ #
    new_rows: list[dict] = []

    for i in range(len(clusters) - 1):
        end_of_a = clusters[i][1]      # last row index of cluster A
        start_of_b = clusters[i + 1][0]  # first row index of cluster B

        x_boundary = np.array([x[end_of_a], x[start_of_b]])

        # Build one interpolated row per generated point per column,
        # then merge by x_interp position so each output row is complete.
        col_interp: dict[float, dict] = {}

        for col in target_columns:
            y_boundary = np.array([
                df[col].iloc[end_of_a],
                df[col].iloc[start_of_b]
            ])

            path_df = generate_interpolation_path(
                x_boundary,
                y_boundary,
                num_points=num_points
            )

            for _, row in path_df.iterrows():
                xi = row["x_interp"]
                if xi not in col_interp:
                    col_interp[xi] = {x_column: xi}
                col_interp[xi][col] = row["y_interp"]

        new_rows.extend(col_interp.values())

    # ------------------------------------------------------------------ #
    # 5. Assemble augmented DataFrame                                    #
    # ------------------------------------------------------------------ #
    if new_rows:
        interp_df = pd.DataFrame(new_rows)
        interp_df["is_interpolated"] = True

        # Align columns — fill any missing ones with NaN
        for col in df.columns:
            if col not in interp_df.columns:
                interp_df[col] = np.nan

        interp_df = interp_df[df.columns]  # preserve column order

        interpolated_df = (
            pd.concat([df, interp_df], ignore_index=True)
            .sort_values(by=x_column)
            .reset_index(drop=True)
        )
    else:
        interpolated_df = df.copy()

    # Strip the helper column from the clean original before returning
    clean_original = original_df.drop(columns=["is_interpolated"])

    return clean_original, interpolated_df