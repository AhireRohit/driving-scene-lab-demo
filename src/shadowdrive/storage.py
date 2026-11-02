"""Atomic, compressed inference snapshots shared by experiments and replay."""
import gzip
import hashlib
import json
from pathlib import Path


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.part')
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(temporary,'wt',encoding='utf8',**({'compresslevel':3} if path.suffix=='.gz' else {})) as f:
        json.dump(value,f,allow_nan=False,separators=(',',':'))
    temporary.replace(path)


def read_json(path):
    path=Path(path);opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf8') as f:return json.load(f)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
