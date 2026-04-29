"""Local CLI: download ProteinGym substitution archive.

Prefer the Modal entrypoint (`modal run modal_app.py::download_proteingym`)
for the canonical pipeline — that one writes into the persistent volume.
This script is here for ad-hoc local downloads (e.g. inspecting a few CSVs).
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_URL = (
    "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--out-dir", default="data/raw/proteingym_v1_3")
    args = p.parse_args()

    target = Path(args.out_dir)
    target.mkdir(parents=True, exist_ok=True)
    print(f"[download] {args.url}")
    try:
        with requests.get(args.url, stream=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length") or 0)
            buf = io.BytesIO()
            chunk = 8 * 1024 * 1024
            with tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for c in r.iter_content(chunk_size=chunk):
                    if c:
                        buf.write(c)
                        pbar.update(len(c))
            buf.seek(0)
    except Exception as exc:
        print(f"[download] failed: {exc}", file=sys.stderr)
        sys.exit(1)
    with zipfile.ZipFile(buf) as z:
        z.extractall(target)
    n = sum(1 for _ in target.rglob("*.csv"))
    print(f"[download] extracted {n} CSVs to {target}")


if __name__ == "__main__":
    main()
