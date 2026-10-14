"""Replay an archived slice manifest. Does not reselect slices or download data."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from ct_preprocessing import extract_slice_patches

p = argparse.ArgumentParser()
p.add_argument('--dicom-root', type=Path, required=True)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--split', type=Path, default=Path('configs/patient_split_seed42.json'))
p.add_argument('--subset', choices=['train','validation','test'], required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
manifest = json.loads(a.manifest.read_text())
records = manifest['slices'] if isinstance(manifest, dict) else manifest
split = json.loads(a.split.read_text())['splits']
groups = defaultdict(list)
for row in records:
    if row['patient_id'] in split[a.subset]: groups[row['patient_id']].append(row)
a.out.mkdir(parents=True, exist_ok=True)
for patient in split[a.subset]:
    if not groups[patient]: raise ValueError(f'Manifest lacks {patient}')
    destination = a.out / f'{patient}.npz'
    if destination.exists(): raise FileExistsError(destination)
    rng = np.random.default_rng([42,int(patient.rsplit('-',1)[1])])
    patches, locations = [], []
    for row in groups[patient]:
        relative = Path(row['relative_path'])
        path = (a.dicom_root / relative).resolve()
        if not path.is_relative_to(a.dicom_root.resolve()): raise ValueError('Invalid manifest path')
        extracted, coordinates = extract_slice_patches(path, rng, count=8)
        patches.extend(extracted)
        locations.extend((row['sorted_slice_index'], y, x) for y,x in coordinates)
    if not patches: raise ValueError(f'No patches for {patient}')
    array = np.stack(patches)[:,None]
    temporary = a.out / f'{patient}.partial.npz'
    np.savez_compressed(temporary, patches=array,
                       locations=np.asarray(locations,dtype=np.int32),patient_id=patient,seed=42)
    temporary.replace(destination)
    print(patient,len(array))
