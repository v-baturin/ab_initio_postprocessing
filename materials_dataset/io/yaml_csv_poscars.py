import os
from typing import Dict, Any
from pathlib import Path
import yaml
from pymatgen.core import Structure
from ..crystal_dataset import CrystalDataset, CrystalEntry


def _get_str_value_for_csv(e: CrystalEntry, attr_name):
    if attr_name == 'structure':
        val = e.poscarname
    elif 'metadata.' in attr_name:
        val = e.metadata.get(attr_name.split('.')[-1], 'NA')
    else:
        val = getattr(e, attr_name)
    if isinstance(val, float):
        val = f"{val:.4f}"
    else:
        val = str(val)
    return val

def _remove_private_keys(dct, recursive=True):
    for key in [k for k in dct if k.startswith('_')]:
        if isinstance(dct[key], dict) and recursive:
            _remove_private_keys(dct[key])
        else:
            dct.pop(key)

def _prepare_for_yaml_write(dct: dict):
    for k, v in dct.items():
        if isinstance(v, dict):
            _prepare_for_yaml_write(v)
        elif isinstance(v, Path):
            dct[k] = v.as_posix()
        elif isinstance(v, float):
            dct[k] = f"{v:.4f}"


def read(manifest_yaml: str | Path) -> CrystalDataset:
    def _rebase_path(pth: Path, base: Path):
        return pth if pth.is_absolute() else base / pth
    with open(manifest_yaml, 'rt') as ds_h:
        dataset_info = yaml.safe_load(ds_h.read())
    dataset_info.pop("comment")
    _remove_private_keys(dataset_info)
    entry_descr_csv = _rebase_path(Path(dataset_info.pop('entries_csv')), manifest_yaml.parent)
    poscars_dir = _rebase_path(Path(dataset_info.pop('poscars')), manifest_yaml.parent)
    dataset = read_csv_poscars(entry_descr_csv, poscars_dir=poscars_dir, use_fname_as_id=False)
    for k in dataset_info:
        setattr(dataset, k, dataset_info[k])
    dataset.override_base_path(manifest_yaml.parent)
    return dataset


def write(dataset: CrystalDataset, enforce_base_path: str | Path | None = None, comment=None, **kwargs):
    if enforce_base_path:
        dataset.override_base_path(Path(enforce_base_path))
    dataset.base_path.mkdir(exist_ok=True)
    manifest_yaml = dataset.base_path / "manifest.yaml"
    csv_file = manifest_yaml.with_name('data.csv')
    poscars_path = manifest_yaml.with_name('POSCARS')
    write_csv_poscars(dataset, csv_file, poscars_path, **kwargs)
    ds_dict = dataset.__dict__.copy()
    _remove_private_keys(ds_dict)
    _prepare_for_yaml_write(ds_dict)
    ds_dict["entries_csv"] = csv_file.relative_to(manifest_yaml.parent).as_posix()
    ds_dict["poscars"] = poscars_path.relative_to(manifest_yaml.parent).as_posix()
    ds_dict["comment"] = comment
    with open(manifest_yaml, 'wt') as ds_desc:
        yaml.dump(ds_dict, ds_desc)
    print(f"Data saved to {dataset.base_path.as_posix()}")


def read_csv_poscars(csv_file, poscars_dir=None, use_fname_as_id=True) -> CrystalDataset:
    csv_file = Path(csv_file)
    poscars_dir = poscars_dir or csv_file.with_name(csv_file.stem + 'POSCARS')
    dataset_ID = csv_file.stem if use_fname_as_id else None
    with open(csv_file, 'rt') as csv_h:
        labels = [h.strip() for h in csv_h.readline().split(',')]
        entries = []
        for i, line in enumerate(csv_h):
            entry_data: Dict[str, Any] = {k: v for (k,v) in zip(labels, [h.strip() for h in line.split(',')])}
            entry_data["metadata"] = dict()
            if not "id" in labels:
                entry_data["id"] = str(i)
            if 'energy' in entry_data:
                entry_data['energy'] = float(entry_data['energy']) if entry_data['energy'] != 'None' else None
            if 'structure' in entry_data:
                entry_data['structure'] = Structure.from_file(poscars_dir / entry_data['structure'])
            for l in labels:
                if 'metadata.' in l:
                    try:
                        val = entry_data.pop(l)
                    except:
                        print('hi')
                    if val in ['True', 'False']:
                        val = eval(val)
                    elif val in ['NA', 'None']:
                        val = None
                    entry_data["metadata"][l.split('.')[-1]] = val
            entry_keys = [k for k in entry_data if k in CrystalEntry.__annotations__.keys()]
            entry_data = {k: entry_data[k] for k in entry_keys}
            entries.append(CrystalEntry(**entry_data))
    return CrystalDataset(entries,
                          dataset_id=dataset_ID,
                          message=f'Loaded from {csv_file}',
                          base_path=csv_file.parent,
                          )

def write_csv_poscars(ds: CrystalDataset,
          csv_file: str | Path | None = None,
          poscars_path: str | Path | None = None,
          labels: list[str] | None = None):

    labels = (labels or (["id", "composition", "energy", "structure"]
              + [f"metadata.{k}" for k in gather_metadata_fields(ds)]))
    base_path = getattr(ds, "basepath", Path(os.getcwd()))
    base_path.mkdir(exist_ok=True)
    csv_file = csv_file or base_path / 'data.csv'
    poscars_path = poscars_path or csv_file.with_name(csv_file.stem + 'POSCARS')
    poscars_path.mkdir(exist_ok=True)
    with open(csv_file, 'wt') as csv_h:
        csv_h.write(','.join(labels) + '\n')
        for e in ds:
            csv_h.write((",".join([_get_str_value_for_csv(e, label) for label in labels])).strip() + '\n')
            e.structure.to(fmt="poscar", filename=poscars_path / _get_str_value_for_csv(e, "structure"), comment=e.id)


def gather_metadata_fields(dataset):
    fields = set()
    for e in dataset:
        fields |= set(e.metadata.keys())
    return list(fields)