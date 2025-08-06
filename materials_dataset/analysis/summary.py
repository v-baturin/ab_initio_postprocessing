from __future__ import annotations

from typing import Any, Callable, Collection, Dict, Iterable, List, Mapping, Sequence
from prettytable import PrettyTable

#--optional imports ----------------------------------------------
try:
    import pandas as _pd
except ModuleNotFoundError as _err_pd:          # fail late, not at import time
    _pd = None
    _PANDAS_ERR = _err_pd

try:
    from prettytable import PrettyTable as _Pt
except ModuleNotFoundError as _err_pt:
    _Pt = None
    _PRETTYTABLE_ERROR = _err_pt

DEBUG = True

# -- helpers -------------------------------------------------------------
def _value_from_native(entry, col: str) -> Any:
    """Return a native attribute or metadata.<key> value (dot syntax)."""
    if col.startswith("metadata."):
        k = col.split(".", 1)[1]
        return entry.metadata.get(k)
    return getattr(entry, col, None)


def _rows_from_dataset(
    ds, native_cols: Sequence[str], callables: Mapping[str, Callable]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for e in ds:
        row: Dict[str, Any] = {col: _value_from_native(e, col) for col in native_cols}

        for name, fn in callables.items():
            val = fn(e)
            if isinstance(val, dict):
                # flatten dict into dotted sub-columns
                for k, v in val.items():
                    row[f"{name}.{k}"] = v
            else:
                row[name] = val

        rows.append(row)
    return rows


# -- public API ---------------------------------------------------------
def collect_summary_df(
    ds,
    native_columns: Collection[str] | None = None,
    extra_columns: Collection[Dict[str, Sequence[Any]]] | None = None,
    callables: Mapping[str, Callable] | None = None,
):
    """
    Build an overview table for `ds`.

    Parameters
    ----------
    ds : CrystalDataset
    native_columns : list/tuple of str
        Column names drawn directly from CrystalEntry attributes
        (`energy`, `id` ...) or metadata keys using `"metadata.<key>"`.
        If None → `("id", "energy")`.
    extra_columns : iterable of dicts `{column_name: sequence}`
        Pre-computed per-row vectors aligned with `ds` length.
    callables : dict `{column_name: fn(entry)}`
        Per-entry functions producing scalars or dicts. Dict results expand
        into dotted sub-columns `column.key`.

    Returns
    -------
    pandas.DataFrame
    """
    if _pd is None:  # pandas not installed
        raise RuntimeError(
            "collect_summary_df() requires pandas. "
            "Install with:  pip install crystaldataset[table]"
        ) from _PANDAS_ERR

    native_cols = tuple(native_columns) if native_columns else ("id", "composition", "energy")
    callables = callables or {}
    extra_columns = extra_columns or ()

    # 1) build row dicts
    rows = _rows_from_dataset(ds, native_cols, callables)

    # 2) create DataFrame
    df = _pd.DataFrame(rows)

    # 3) attach extra columns
    for mapping in extra_columns:
        for name, values in mapping.items():
            if len(values) != len(ds):
                raise ValueError(
                    f"extra_columns[{name!r}] length {len(values)} "
                    f"!= dataset length {len(ds)}"
                )
            df[name] = list(values)

    return df

def print_pretty_df(df, dump_path, columns=None, sort_by=None, pretty=True, ):
    df["_int_ID"] = df["id"].str.extract(r'(\d+)(?!.*\d)')[0].astype(int)
    if isinstance(sort_by, list):
        sort_by.append("_int_ID")
    elif sort_by:
        sort_by = [sort_by, "_int_ID"]
    else:
        sort_by = ["_int_ID"]

    df = df.sort_values(by=sort_by)
    df.drop(columns="_int_ID", inplace=True)
    if not columns:
        columns = list(df.columns)
    else:
        # Validate that each requested column exists
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"Columns {missing} are not in the DataFrame ({df.columns})")

    if pretty and not _Pt: # pandas not installed
        raise RuntimeError(
            "pretty=True requires Prettytable "
            "Install with:  pip install crystaldataset[table]"
        ) from _PRETTYTABLE_ERROR

    if pretty:
        pt = PrettyTable()
        pt.field_names = columns
        float_fmt = "{:.4f}".format
        # Slice df to just the chosen columns, then iterate
        for row in df[columns].itertuples(index=False, name=None):
            formatted_row = [float_fmt(x) if isinstance(x, float) else x for x in row]
            pt.add_row(formatted_row)

        # Write the table string to a file
        with open(dump_path, "w") as file:
            file.write(pt.get_string())

        if DEBUG:
            print(pt)
    else:
        df.to_csv(dump_path, columns=columns)
        if DEBUG:
            print(df)


