from __future__ import annotations
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

##############################################################################
# helpers
##############################################################################

def _copy_zips_preserving_tree(src_root: Path, dest_root: Path) -> None:
    """
    Copy every *.zip under *src_root* into *dest_root*, keeping the
    directory hierarchy (cp --parents analogue).
    """
    for z in src_root.rglob("*.zip"):
        rel_path = z.relative_to(src_root)
        target = dest_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(z, target)


def _unzip_in_place(root: Path) -> None:
    """
    Recursively walk *root*; for each foo.zip:
      * ensure sibling folder foo/ does not already exist,
      * create foo/, extract contents there,
      * delete foo.zip
    Loops until no zip files remain anywhere under *root*.
    """
    while True:
        zips = list(root.rglob("*.zip"))
        if not zips:
            break

        for z in zips:
            target_dir = z.with_suffix("")  # foo.zip → foo/
            if target_dir.exists():
                raise FileExistsError(
                    f"{target_dir} already exists next to archive {z}"
                )
            target_dir.mkdir()
            with ZipFile(z) as zf:
                zf.extractall(target_dir)
            z.unlink()  # remove the archive

##############################################################################
# public façade
##############################################################################

@contextmanager
def exploded_zip_tree(src_root: str | Path):
    """
    Context manager that produces a fully‑exploded copy of all zip archives
    under *src_root* and cleans it up afterwards.

    Usage:
        with exploded_zip_tree("zip_files") as tmp_root:
            f(tmp_root)     # <- your operation runs here
    """
    src_root = Path(src_root)
    with tempfile.TemporaryDirectory(prefix="tmp_zip_file_") as tmp:
        tmp_root = Path(tmp)
        _copy_zips_preserving_tree(src_root, tmp_root)
        _unzip_in_place(tmp_root)
        yield tmp_root     # caller does its work here
        # automatic cleanup by TemporaryDirectory

##############################################################################
# example
##############################################################################

def f(tree_root: Path):
    """Dummy operation that just prints the expanded tree."""
    print("Expanded tree located at:", tree_root)
    for p in tree_root.rglob("*"):
        print(" ", p.relative_to(tree_root))

if __name__ == "__main__":
    with exploded_zip_tree("zip_files") as tmp_tree:
        f(tmp_tree)  # do real work here
