from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import yaml
import networkx as nx

# --- CrystalDataset helpers ------------------------------------------
from ..io.yaml_csv_poscars import read as cd_read, write as cd_write
from ..crystal_dataset import CrystalDataset

__all__ = ["DatasetRepo"]

# --------------------------------------------------------------------
# Helper constants ----------------------------------------------------
# --------------------------------------------------------------------

_MANIFEST_NAME = "manifest.yaml"
_INDEX_NAME = "index.yaml"  # repository‑level summary mapping


class DatasetRepo:
    """Filesystem‑backed provenance repository for **CrystalDataset** objects.

    Design choices
    ---------------
    * **Exactly one `manifest.yml` per node** with schema::

        message: null | str
        dataset_id: <str>              # unique full hash
        entries_csv: data.csv
        metadata: {}
        parent_ids: [<id>, ...] | null
        poscars: manifestPOSCARS
        created: 2025‑07‑28T14:31:02Z  # auto‑filled on commit

    * Heavy artefacts (CSV/POSCARS) are produced by `cd_write()`.
    * A repository‑level **index file** (`index.yml`) is updated on each commit.
    * A cached `networkx.DiGraph` represents the DAG (parent → child).
    """

    # ------------------------------------------------------------------
    # Construction & graph ---------------------------------------------
    # ------------------------------------------------------------------
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self._graph: Optional[nx.DiGraph] = None  # lazily built cache

    # ------------------------------------------------------------------
    # Graph building & listing -----------------------------------------
    # ------------------------------------------------------------------
    def build_graph(self, force: bool = False) -> nx.DiGraph:
        """Return (and cache) a `networkx.DiGraph` of the repository."""
        if self._graph is None or force:
            G = nx.DiGraph()
            for yml in self.root.rglob(_MANIFEST_NAME):
                try:
                    meta = yaml.safe_load(yml.read_text())
                except Exception(yaml.YAMLError, OSError):
                    continue  # skip malformed YAML silently

                node_id = meta.get("dataset_id")
                if not node_id:
                    continue  # invalid manifest

                G.add_node(node_id, **meta, _path=yml)

                for parent in meta.get("parent_ids") or []:
                    G.add_edge(parent, node_id)
            self._graph = G
        return self._graph

    def list_nodes(self) -> List[str]:
        """Return list of dataset IDs present in the repo."""
        return list(self.build_graph().nodes())

    # ------------------------------------------------------------------
    # Load / Commit -----------------------------------------------------
    # ------------------------------------------------------------------
    def load_node(self, node_id: str) -> Tuple[CrystalDataset, Dict[str, Any]]:
        """Load node *node_id* and return `(CrystalDataset, manifest_dict)`."""
        manifest_path = self._node_manifest(node_id)
        dataset = cd_read(manifest_path)
        manifest = yaml.safe_load(manifest_path.read_text())
        return dataset, manifest

    def commit_node(
        self,
        dataset: CrystalDataset,
        *,
        suffix: str = "",
        overwrite: bool = False,
        **kwargs
    ) -> str:
        """Persist *dataset* in the repo and return its ``dataset_id``.

        All on‑disk files (YAML/CSV/POSCARS) are produced by ``cd_write``.
        We **leave the manifest untouched** and simply append an entry to
        the repository‑level ``index.yml`` so the node can be discovered
        quickly.
        """
        if not dataset.dataset_id:
            raise ValueError("CrystalDataset must have a non‑empty dataset_id")

        node_id = dataset.dataset_id
        folder = self.root / f"{node_id}{('_' + suffix) if suffix else ''}"
        if folder.exists() and not overwrite:
            raise FileExistsError(f"Node folder {folder} already exists")
        folder.mkdir(parents=True, exist_ok=True)

        # Delegate writing to the user helper
        cd_write(dataset, enforce_base_path=folder)

        manifest_path = folder / _MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"cd_write() did not create {_MANIFEST_NAME}")

        meta = {
    "parent_ids": dataset.parent_ids,
    "created": dataset.metadata.get("created_on"),
    "message": dataset.metadata.get("message"),
}

        # Update repo‑level index and invalidate cache
        self._append_index(node_id, folder, meta)
        self._graph = None
        return node_id

    # ------------------------------------------------------------------
    # Internal helpers --------------------------------------------------
    # ------------------------------------------------------------------
    def _node_manifest(self, node_id: str) -> Path:
        """Return path to `manifest.yml` for *node_id* or raise KeyError."""
        G = self.build_graph()
        if node_id not in G:
            raise KeyError(f"Node '{node_id}' not found in repo")
        return G.nodes[node_id]["_path"]

    def _append_index(self, node_id: str, folder: Path, meta: Dict[str, Any]):
        """Update repository‑level `index.yml` with the new node."""
        index_path = self.root / _INDEX_NAME
        try:
            index: Dict[str, Any] = (
                yaml.safe_load(index_path.read_text()) if index_path.exists() else {}
            ) or {}
        except Exception:
            index = {}

        rel_path = folder.relative_to(self.root).as_posix()
        index[node_id] = {
            "path": rel_path,
            "parent_ids": meta.get("parent_ids") or [],
            "created": meta.get("created"),
            "message": meta.get("message"),
        }

        with index_path.open("w") as fh:
            yaml.safe_dump(index, fh, sort_keys=False)

    def rebuild_index(self) -> None:
        G = self.build_graph(force=True)
        index = {}
        for node_id in G.nodes:
            node = G.nodes[node_id]
            meta = {
                "path": node["_path"].parent.relative_to(self.root).as_posix(),
                "parent_ids": node.get("parent_ids", []),
                "created": node.get("metadata",{}).get("created_on", {}),
                "message": node.get("metadata",{}).get("message", {}),
            }
            index[node_id] = meta

        index_path = self.root / _INDEX_NAME
        with index_path.open("w") as fh:
            yaml.safe_dump(index, fh, sort_keys=False)
