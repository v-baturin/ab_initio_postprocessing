# crystaldata/storage.py
import pickle, gzip
from pathlib import Path
from typing import Union
from ..crystal_dataset import CrystalDataset


_PathLike = Union[str, Path]
_PROTOCOL = 5                       # fastest for 3.8+

# ---------- save / load -------------------------------------------
def save(ds: CrystalDataset, path: _PathLike, *, compress=True) -> None:
    blob = pickle.dumps(ds, protocol=_PROTOCOL)
    if compress:
        with gzip.open(path, "wb", compresslevel=6) as fh:
            fh.write(blob)
    else:
        Path(path).write_bytes(blob)

def load(path: _PathLike) -> CrystalDataset:
    try:
        with gzip.open(path, "rb") as fh:
            blob = fh.read()
    except OSError:                  # not gzipped
        blob = Path(path).read_bytes()

    ds = pickle.loads(blob)
    if not isinstance(ds, CrystalDataset):
        raise TypeError(f"{path} does not contain a CrystalDataset")
    return ds