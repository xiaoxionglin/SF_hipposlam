"""Combine CPU/GPU launcher manifests and audit one 42-run StudySpec."""

import argparse
import csv
import json
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.submission import audit_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs=2)
    parser.add_argument("--submitted", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = None
    rows = []
    for path in args.inputs:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError("Launcher manifests have different columns")
            rows.extend(reader)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    audit = audit_submission(load_study(args.study), args.output, require_submitted=args.submitted)
    report = args.output.with_name("study_audit.json")
    report.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"rows": audit["rows"], "sha256": audit["study_sha256"],
                      "submitted_complete": audit["submitted_complete"]}))


if __name__ == "__main__":
    main()
