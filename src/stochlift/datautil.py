"""Addressing numeric entries ("leaves") inside a nested data dictionary.

A leaf path is a tuple of keys/indices, e.g. ``("yield", "wheat")`` or
``("demand", 3)``. Its display name joins the parts with dots: ``yield.wheat``.

Containers: dicts, lists, tuples (read only), NumPy arrays, pandas Series (one
path step: the index label) and pandas DataFrames (two steps: row label, then
column label). Missing values in pandas objects are not leaves.
"""
from __future__ import annotations

import copy
import numbers

import numpy as np


def is_number(v) -> bool:
    return isinstance(v, (numbers.Real, np.integer, np.floating)) and not isinstance(v, bool)


def _is_series(obj) -> bool:
    return type(obj).__name__ == "Series" and type(obj).__module__.startswith("pandas")


def _is_frame(obj) -> bool:
    return type(obj).__name__ == "DataFrame" and type(obj).__module__.startswith("pandas")


def leaf_name(path) -> str:
    return ".".join(str(p) for p in path)


def numeric_leaves(obj, prefix=()) -> list:
    """All numeric leaves under ``obj`` as ``[(path, value), ...]`` in a stable order."""
    out = []
    if is_number(obj):
        out.append((tuple(prefix), float(obj)))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(numeric_leaves(v, tuple(prefix) + (k,)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(numeric_leaves(v, tuple(prefix) + (i,)))
    elif isinstance(obj, np.ndarray) and obj.dtype.kind in "fiu" and obj.ndim == 0:
        out.append((tuple(prefix), float(obj)))
    elif isinstance(obj, np.ndarray) and obj.dtype.kind in "fiu":
        for i in np.ndindex(obj.shape):
            out.append((tuple(prefix) + (i if len(i) > 1 else i[0],), float(obj[i])))
    elif _is_series(obj) and obj.dtype.kind in "fiu":
        for label, v in obj.items():
            if np.isfinite(v):
                out.append((tuple(prefix) + (label,), float(v)))
    elif _is_frame(obj):
        cols = [c for c in obj.columns if obj[c].dtype.kind in "fiu"]
        for row in obj.index:
            for col in cols:
                v = obj.at[row, col]
                if np.isfinite(v):
                    out.append((tuple(prefix) + (row, col), float(v)))
    return out


def get_leaf(data, path):
    cur, i = data, 0
    while i < len(path):
        if _is_frame(cur):
            if i + 1 >= len(path):
                raise KeyError(f"{leaf_name(path)}: a DataFrame entry needs a row and a column label")
            cur = cur.at[path[i], path[i + 1]]
            i += 2
        elif _is_series(cur):
            cur = cur.at[path[i]]
            i += 1
        else:
            cur = cur[path[i]]
            i += 1
    return cur


def set_leaf(data, path, value) -> None:
    """Set one entry in place. Integer arrays and columns become float when written."""
    parent, key, cur, i = None, None, data, 0
    while True:
        if _is_frame(cur):
            row, col = path[i], path[i + 1]
            if cur[col].dtype.kind in "iub":
                cur[col] = cur[col].astype(float)
            cur.at[row, col] = value
            return
        if _is_series(cur) or (isinstance(cur, np.ndarray) and i == len(path) - 1):
            if cur.dtype.kind in "iub":
                if parent is None or isinstance(parent, tuple):
                    raise TypeError(f"cannot set {leaf_name(path)}: integer container inside a tuple")
                cur = cur.astype(float)
                parent[key] = cur
            if _is_series(cur):
                cur.at[path[i]] = value
            else:
                cur[path[i]] = value
            return
        if i == len(path) - 1:
            if isinstance(cur, tuple):
                raise TypeError(f"cannot set {leaf_name(path)}: tuples are immutable, use a list")
            cur[path[i]] = value
            return
        parent, key, cur = cur, path[i], cur[path[i]]
        i += 1


def apply(data: dict, paths, values) -> dict:
    """Deep copy of ``data`` with ``paths`` set to ``values``."""
    if len(paths) != len(values):
        raise ValueError(f"expected {len(paths)} values (one per uncertain entry), got {len(values)}")
    new = copy.deepcopy(data)
    for p, v in zip(paths, values):
        old = get_leaf(new, p)
        set_leaf(new, p, type(old)(v) if isinstance(old, float) else float(v))
    return new


def expand_keys(data: dict, keys) -> list:
    """Leaf paths for uncertain ``keys``.

    A key is a top-level name (``"yield"``: every numeric leaf below it) or a
    dotted leaf name (``"yield.wheat"``).
    """
    all_leaves = numeric_leaves(data)
    by_name = {leaf_name(p): p for p, _ in all_leaves}
    paths = []
    for key in keys:
        key = str(key)
        if key in by_name:
            found = [by_name[key]]
        else:
            found = [p for p, _ in all_leaves
                     if leaf_name(p).startswith(key + ".")]
        if not found:
            raise KeyError(f"uncertain key '{key}' matches no numeric entry in the data; "
                           f"available top-level keys: {sorted(map(str, data.keys()))}")
        for p in found:
            if p not in paths:
                paths.append(p)
    return paths


def summarize(data: dict, max_items: int = 6) -> str:
    """Compact description of the data dictionary (used in the LLM prompt)."""
    lines = []
    for k, v in data.items():
        leaves = numeric_leaves(v)
        if is_number(v):
            lines.append(f"- {k}: number = {v}")
        elif leaves:
            sample = ", ".join(f"{leaf_name(p)}={val:g}" for p, val in leaves[:max_items])
            more = f", ... ({len(leaves)} numbers)" if len(leaves) > max_items else ""
            lines.append(f"- {k}: {type(v).__name__} with numeric entries: {sample}{more}")
        else:
            text = repr(v)
            lines.append(f"- {k}: {type(v).__name__} = {text[:80]}{'...' if len(text) > 80 else ''}")
    return "\n".join(lines)
