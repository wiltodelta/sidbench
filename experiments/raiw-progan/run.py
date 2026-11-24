"""Run the RAIW v10 pilot on a balanced sample of the Wang2020 ProGAN test set."""

import argparse
import csv
import hashlib
import io
import json
import logging
import platform
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image
from remove_ai_watermarks.source_classify import classify_source

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.evaluation_utils import calculate_for_threshold, calculate_performance_metrics

log = logging.getLogger(__name__)
ARCHIVE_SHA256 = "206c39e2b12395680c50cc631cfeecc6ca51db0cb1ac543b0a9558e4a11e7d8b"
WEIGHTS_SHA256 = "b3eb961cc68cf283b399bebf11cdf61b2bf41b98fabca0579f8b61d9141f695f"
LABELS = {"0_real": 0, "1_fake": 1}
PER_GROUP = 4


def verify(path: Path, expected: str) -> None:
    with path.open("rb") as file:
        actual = hashlib.file_digest(file, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}")


def sample(archive: ZipFile) -> list[tuple[str, int]]:
    groups: dict[tuple[str, int], list[str]] = {}
    for member in archive.namelist():
        parts = PurePosixPath(member).parts
        if len(parts) == 4 and parts[0] == "progan" and parts[2] in LABELS and parts[3].endswith(".png"):
            groups.setdefault((parts[1], LABELS[parts[2]]), []).append(member)
    if len(groups) != 40 or any(len(group) != 200 for group in groups.values()):
        raise ValueError("Unexpected ProGAN archive layout")
    return [
        (member, label)
        for (_, label), group in sorted(groups.items())
        for member in sorted(group, key=lambda name: hashlib.sha256(name.encode("utf-8")).digest())[:PER_GROUP]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for the CSV and metrics")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must name a new directory")
    verify(args.weights, WEIGHTS_SHA256)
    verify(args.archive, ARCHIVE_SHA256)
    rows = []
    with ZipFile(args.archive) as archive:
        for member, label in sample(archive):
            data = archive.read(member)
            with Image.open(io.BytesIO(data)) as image:
                result = classify_source(image.convert("RGB"), weights_path=args.weights)
            rows.append(
                {
                    "member": member,
                    "image_sha256": hashlib.sha256(data).hexdigest(),
                    "label": label,
                    "score": result.synthetic_probability,
                    "raiw_synthetic_detected": int(result.synthetic_detected),
                }
            )
    labels = np.asarray([row["label"] for row in rows])
    scores = np.asarray([row["score"] for row in rows])
    decisions = np.asarray([row["raiw_synthetic_detected"] for row in rows])
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("RAIW returned an invalid score")
    metrics = calculate_performance_metrics(labels, scores)
    summary = {
        "dataset": "Wang2020 ProGAN test set",
        "image_count": len(rows),
        "real_count": int((labels == 0).sum()),
        "synthetic_count": int((labels == 1).sum()),
        "archive_sha256": ARCHIVE_SHA256,
        "weights_sha256": WEIGHTS_SHA256,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["ap"],
        "sidbench_score_gt_0_5": {key: float(value) for key, value in metrics["threshold_05"].items()},
        "raiw_native_decision": {
            key: float(value) for key, value in calculate_for_threshold(labels, decisions, 0.5).items()
        },
        "decision_disagreements": int(((scores > 0.5) != decisions).sum()),
    }
    args.output.mkdir()
    with (args.output / "predictions.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote %s predictions and metrics to %s", len(rows), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
