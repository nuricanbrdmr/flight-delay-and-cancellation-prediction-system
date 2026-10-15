"""Unpack the gzip-compressed final models of the article (model-api/trained_models/*.json.gz) into csv/models_v2/<tag>.json,
where 14_test_report.py and 19_test_breakdown_calibration.py expect them. Existing files are not overwritten."""
import gzip, shutil
from pathlib import Path
SRC = Path(__file__).resolve().parents[1] / "trained_models"
DST = Path(__file__).resolve().parents[2] / "csv" / "models_v2"
DST.mkdir(parents=True, exist_ok=True)
for gz in sorted(SRC.glob("*.json.gz")):
    out = DST / gz.name[:-3]
    if out.exists():
        print("exists, kept:", out.name); continue
    with gzip.open(gz, "rb") as f, open(out, "wb") as g: shutil.copyfileobj(f, g)
    print("unpacked:", out.name)
