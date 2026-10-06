#!/usr/bin/env python3
"""Recompute TinyStress WER, speaker similarity and StressLM F1 on a subset of
the test split, without re-running any model.

Every metric is re-derived from the per-sample records the evaluation pipelines
already wrote, so restricting the report to a subset is just a matter of
dropping records before aggregating:

    WER          {wav_dir}/_wer_results.jsonl            keyed by "utt"  (faster-whisper large-v3)
    SIM          {wav_dir}/_sim_results.jsonl            keyed by "utt"  (WavLM-ECAPA, where it was scored)
    StressLM F1  StressTest/results/<name>_ssd.json      keyed by "utterance_id"

The subset is defined by the gold emphasis count in the TinyStress test Parquet
shard (--max-emphasis, 2 by default), which is the same source the pipelines
score against.

Aggregation mirrors the evaluation scripts: WER and SIM are means over
samples; F1 pools word tokens across samples and then computes binary
precision/recall/F1, as StressTest does. Verification (on by default) recomputes
the *unfiltered* numbers as well and checks them against the stored summaries
(``_wer_results.json``, ``_sim_results.json`` and ``<name>_metrics_ssd.json``), so a
mismatch in record format cannot pass unnoticed.

Examples
--------
    # One system
    scripts/filter_tinystress.sh \\
        --wav-dir results/tinystress \\
        --ssd-json StressTest/results/tinystress_ssd.json

    # Several systems with explicit labels, machine-readable output too
    scripts/filter_tinystress.sh --max-emphasis 2 \\
        --wav-dir Baseline=results/tinystress_baseline \\
        --wav-dir EmphTTS=results/tinystress \\
        --ssd-json Baseline=StressTest/results/tinystress_baseline_ssd.json \\
        --ssd-json EmphTTS=StressTest/results/tinystress_ssd.json \\
        --json-out results/tinystress_le2emph_metrics.json
"""

import argparse
import json
import os
import sys

from emphtts.tts.eval.utils_eval import TINYSTRESS_DEFAULT_PARQUET

# Per-sample record files inside a generation directory: (file name, id field).
WAV_DIR_SOURCES = {
    "wer": ("_wer_results.jsonl", "utt"),
    "sim": ("_sim_results.jsonl", "utt"),
}


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def parse_labelled_path(value, kind):
    """Accept either ``PATH`` or ``LABEL=PATH``; derive a label when absent."""
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError(f"Empty label in {value!r}")
        return label, os.path.expanduser(path.strip())

    path = os.path.expanduser(value)
    if kind == "wav-dir":
        label = os.path.basename(path.rstrip("/"))
    else:
        # results/tinystress_baseline_ssd.json -> baseline
        label = os.path.basename(path)
        for suffix in (".json", "_ssd"):
            if label.endswith(suffix):
                label = label[: -len(suffix)]
        if label.startswith("tinystress_"):
            label = label[len("tinystress_"):]
    return label, path


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
    return records


def load_emphasis_counts(parquet_path):
    """Map utterance id -> number of distinct gold emphasis words."""
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(parquet_path)
    available = set(parquet_file.schema_arrow.names)
    missing = [c for c in ("id", "emphasis_indices") if c not in available]
    if missing:
        raise ValueError(
            f"TinyStress Parquet {parquet_path} is missing column(s) {missing}; "
            f"found {sorted(available)}"
        )

    counts = {}
    for batch in parquet_file.iter_batches(batch_size=256, columns=["id", "emphasis_indices"]):
        for row in batch.to_pylist():
            # Ids carry gaps in the published split, so file names keep them as-is.
            counts[f"tinystress_{row['id']:05d}"] = len(set(row["emphasis_indices"] or []))
    if not counts:
        raise ValueError(f"TinyStress Parquet {parquet_path} yielded no rows.")
    return counts


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def aggregate_scalar(records, id_field, value_field, keep):
    """Mean of a per-sample scalar, over all records and over the kept subset."""
    all_values, kept_values, unknown = [], [], []
    for record in records:
        utt = record[id_field]
        value = float(record[value_field])
        all_values.append(value)
        if utt not in keep.universe:
            unknown.append(utt)
        elif utt in keep.subset:
            kept_values.append(value)
    return {
        "n_all": len(all_values),
        "all": (sum(all_values) / len(all_values)) if all_values else None,
        "n_kept": len(kept_values),
        "kept": (sum(kept_values) / len(kept_values)) if kept_values else None,
        "unknown_ids": unknown,
    }


def binary_metrics(tp, fp, fn, tn):
    """Pooled binary scores over word tokens (StressTest's SSD precision/recall/F1, plus accuracies)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    total = tp + fp + fn + tn
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "accuracy": (tp + tn) / total if total else 0.0,
        "specificity": specificity,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "num_tokens": total,
    }


def aggregate_binary(records, id_field, gold_field, pred_field, keep):
    """Pool word tokens across samples, then score — for all records and the subset."""
    counts = {"all": [0, 0, 0, 0], "kept": [0, 0, 0, 0]}  # tp, fp, fn, tn
    n_all = n_kept = 0
    unknown, length_mismatch, label_mismatch = [], [], []

    for record in records:
        utt = record[id_field]
        gold = [int(g) for g in record[gold_field]]
        pred = [int(p) for p in record[pred_field]]
        if len(gold) != len(pred):
            length_mismatch.append(utt)
            continue

        buckets = ["all"]
        n_all += 1
        if utt not in keep.universe:
            unknown.append(utt)
        else:
            if sum(gold) != keep.universe[utt]:
                # The stored gold should be the Parquet emphasis mask; if it is
                # not, the subset filter would be silently scoring something else.
                label_mismatch.append(utt)
            if utt in keep.subset:
                buckets.append("kept")
                n_kept += 1

        for g, p in zip(gold, pred):
            idx = 0 if (p == 1 and g == 1) else 1 if (p == 1) else 2 if (g == 1) else 3
            for bucket in buckets:
                counts[bucket][idx] += 1

    return {
        "n_all": n_all,
        "all": binary_metrics(*counts["all"]) if n_all else None,
        "n_kept": n_kept,
        "kept": binary_metrics(*counts["kept"]) if n_kept else None,
        "unknown_ids": unknown,
        "length_mismatch_ids": length_mismatch,
        "label_mismatch_ids": label_mismatch,
    }


class Keep:
    """The full id universe (id -> emphasis count) and the ids passing the filter."""

    def __init__(self, emphasis_counts, min_emphasis, max_emphasis):
        self.universe = emphasis_counts
        self.min_emphasis = min_emphasis
        self.max_emphasis = max_emphasis
        self.subset = {
            utt for utt, n in emphasis_counts.items() if min_emphasis <= n <= max_emphasis
        }


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #

def collect_wav_dir(label, wav_dir, keep, warnings):
    """Whisper-large-v3 WER and WavLM speaker similarity from one generation directory."""
    if not os.path.isdir(wav_dir):
        raise FileNotFoundError(f"[{label}] not a directory: {wav_dir}")

    metrics, stored = {}, {}

    for metric in ("wer", "sim"):
        file_name, id_field = WAV_DIR_SOURCES[metric]
        record_path = os.path.join(wav_dir, file_name)
        if not os.path.exists(record_path):
            warnings.append(f"[{label}] no {file_name} in {wav_dir}; skipping {metric.upper()}")
            continue

        metrics[metric] = aggregate_scalar(read_jsonl(record_path), id_field, metric, keep)
        metrics[metric]["source"] = record_path

        # The _wer_results.json summary also carries CER; only WER is reported here.
        summary = read_stored_summary(os.path.join(wav_dir, f"_{metric}_results.json"))
        if metric in summary:
            stored[metric] = summary[metric]

    return metrics, stored


def collect_ssd_json(label, ssd_path, keep, warnings):
    """StressLM F1 from a StressTest SSD inference results file."""
    with open(ssd_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    evaluations = data.get("evaluations", [])
    if not evaluations:
        warnings.append(f"[{label}] {ssd_path} holds no evaluations; skipping StressLM F1")
        return {}, {}

    id_field = "utterance_id" if "utterance_id" in evaluations[0] else None
    if id_field is None:
        raise ValueError(
            f"[{label}] {ssd_path} has no 'utterance_id' field, so its samples cannot be "
            f"matched to TinyStress ids. Re-run the SSD evaluation to regenerate it."
        )

    result = aggregate_binary(evaluations, id_field, "stress_labels", "stress_pred", keep)
    result["source"] = ssd_path

    stored = {}
    # StressTest's naming: <name>_ssd.json (inference results) and <name>_metrics_ssd.json (summary).
    metrics_path = ssd_path.replace("_ssd.json", "_metrics_ssd.json")
    if metrics_path != ssd_path and os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            ssd_metrics = json.load(f).get("ssd_metrics", {})
        if "f1" in ssd_metrics:
            stored["stresslm_f1"] = ssd_metrics["f1"]
    else:
        warnings.append(f"[{label}] no {os.path.basename(metrics_path)} next to {ssd_path}; StressLM F1 not verified")

    return {"stresslm_f1": result}, stored


def read_stored_summary(path):
    """The ``metrics`` block of an F5-TTS ``_*_results.json`` summary, if present."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("metrics", {})


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

SCALAR_METRICS = [("wer", "WER"), ("sim", "SIM")]
BINARY_METRICS = [("stresslm_f1", "StressLM F1")]


def fmt(value, digits=5):
    return "-" if value is None else f"{value:.{digits}f}"


def print_table(rows, headers, aligns=None):
    aligns = aligns or ["<"] + [">"] * (len(headers) - 1)
    widths = [
        max(len(str(headers[i])), max((len(str(r[i])) for r in rows), default=0))
        for i in range(len(headers))
    ]
    line = "  ".join(f"{str(h):{aligns[i]}{widths[i]}}" for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(f"{str(c):{aligns[i]}{widths[i]}}" for i, c in enumerate(row)))


def report(systems, keep, verify):
    subset_label = f"<={keep.max_emphasis}"
    if keep.min_emphasis > 0:
        subset_label = f"{keep.min_emphasis}-{keep.max_emphasis}"

    for key, title in SCALAR_METRICS:
        rows = []
        for label, entry in systems.items():
            m = entry["metrics"].get(key)
            if not m:
                continue
            rows.append([
                label, m["n_all"], fmt(m["all"]), m["n_kept"], fmt(m["kept"]),
                fmt((m["kept"] - m["all"]) if None not in (m["kept"], m["all"]) else None),
            ])
        if rows:
            print(f"\n=== {title} (mean over samples) ===")
            print_table(
                rows,
                ["system", "n(all)", title.lower(), f"n({subset_label})", f"{title.lower()} ({subset_label})", "delta"],
            )

    for key, title in BINARY_METRICS:
        rows = []
        for label, entry in systems.items():
            m = entry["metrics"].get(key)
            if not m:
                continue
            a, k = m["all"] or {}, m["kept"] or {}
            rows.append([
                label,
                m["n_all"], fmt(a.get("precision"), 4), fmt(a.get("recall"), 4), fmt(a.get("f1"), 4),
                m["n_kept"], fmt(k.get("precision"), 4), fmt(k.get("recall"), 4), fmt(k.get("f1"), 4),
                fmt((k["f1"] - a["f1"]) if ("f1" in k and "f1" in a) else None, 4),
            ])
        if rows:
            print(f"\n=== {title} (pooled word tokens) ===")
            print_table(
                rows,
                ["system", "n(all)", "P", "R", "F1",
                 f"n({subset_label})", f"P ({subset_label})", f"R ({subset_label})", f"F1 ({subset_label})",
                 "delta F1"],
            )

    if verify:
        print_verification(systems)


def print_verification(systems):
    """Recomputed unfiltered values vs. the numbers the original runs stored."""
    rows = []
    for label, entry in systems.items():
        for key, stored_value in entry["stored"].items():
            m = entry["metrics"].get(key)
            if m is None:
                continue
            recomputed = m["all"]["f1"] if isinstance(m["all"], dict) else m["all"]
            if recomputed is None:
                continue
            # Stored WER/CER/SIM are rounded to 5 decimals by the original script.
            delta = abs(recomputed - stored_value)
            rows.append([label, key, fmt(stored_value), fmt(recomputed),
                         "ok" if delta <= 1e-5 else f"MISMATCH ({delta:.2e})"])
    if rows:
        print("\n=== verification: unfiltered recomputation vs. stored summaries ===")
        print_table(rows, ["system", "metric", "stored", "recomputed", "status"])
        bad = [r for r in rows if r[4] != "ok"]
        if bad:
            print(f"\n{len(bad)} metric(s) did not reproduce the stored value; treat the "
                  f"filtered numbers above with suspicion.")


def print_warnings(systems, warnings):
    issues = list(warnings)
    for label, entry in systems.items():
        for key, m in entry["metrics"].items():
            if m.get("unknown_ids"):
                ids = m["unknown_ids"]
                issues.append(
                    f"[{label}/{key}] {len(ids)} record(s) are not in the Parquet split "
                    f"and were excluded from the subset, e.g. {ids[:5]}"
                )
            if m.get("length_mismatch_ids"):
                ids = m["length_mismatch_ids"]
                issues.append(
                    f"[{label}/{key}] {len(ids)} record(s) have gold/pred length mismatch "
                    f"and were dropped entirely, e.g. {ids[:5]}"
                )
            if m.get("label_mismatch_ids"):
                ids = m["label_mismatch_ids"]
                issues.append(
                    f"[{label}/{key}] {len(ids)} record(s) carry a gold mask that disagrees "
                    f"with the Parquet emphasis count, e.g. {ids[:5]}"
                )
    if issues:
        print("\n=== warnings ===")
        for issue in issues:
            print(f"  {issue}")


# --------------------------------------------------------------------------- #

def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--wav-dir", action="append", default=[], metavar="[LABEL=]DIR",
        help="Generation directory holding _wer_results.jsonl / _sim_results.jsonl. "
             "Repeatable.",
    )
    parser.add_argument(
        "--ssd-json", action="append", default=[], metavar="[LABEL=]PATH",
        help="StressTest SSD inference results JSON (StressTest/results/<name>_ssd.json), "
             "the source of the StressLM F1. Repeatable.",
    )
    parser.add_argument("--tinystress-parquet", default=TINYSTRESS_DEFAULT_PARQUET,
                        help="TinyStress test Parquet shard (source of the gold emphasis counts)")
    parser.add_argument("--max-emphasis", type=int, default=2,
                        help="Keep samples with at most this many gold emphasis words (default: 2)")
    parser.add_argument("--min-emphasis", type=int, default=0,
                        help="Keep samples with at least this many gold emphasis words (default: 0)")
    parser.add_argument("--json-out", default="", help="Write the full report as JSON to this path")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip checking the unfiltered recomputation against stored summaries")
    args = parser.parse_args()
    if not args.wav_dir and not args.ssd_json:
        parser.error("give at least one --wav-dir or --ssd-json")
    if args.min_emphasis > args.max_emphasis:
        parser.error("--min-emphasis cannot exceed --max-emphasis")
    return args


def main():
    args = get_args()

    emphasis_counts = load_emphasis_counts(args.tinystress_parquet)
    keep = Keep(emphasis_counts, args.min_emphasis, args.max_emphasis)
    print(
        f"TinyStress split: {len(keep.universe)} samples, "
        f"{len(keep.subset)} with {args.min_emphasis} <= emphasis words <= {args.max_emphasis} "
        f"({100 * len(keep.subset) / len(keep.universe):.1f}%)"
    )

    systems, warnings = {}, []

    for value in args.wav_dir:
        label, path = parse_labelled_path(value, "wav-dir")
        entry = systems.setdefault(label, {"metrics": {}, "stored": {}, "inputs": {}})
        metrics, stored = collect_wav_dir(label, path, keep, warnings)
        entry["metrics"].update(metrics)
        entry["stored"].update(stored)
        entry["inputs"]["wav_dir"] = path

    for value in args.ssd_json:
        label, path = parse_labelled_path(value, "ssd-json")
        entry = systems.setdefault(label, {"metrics": {}, "stored": {}, "inputs": {}})
        metrics, stored = collect_ssd_json(label, path, keep, warnings)
        entry["metrics"].update(metrics)
        entry["stored"].update(stored)
        entry["inputs"]["ssd_json"] = path

    report(systems, keep, verify=not args.no_verify)
    print_warnings(systems, warnings)

    if args.json_out:
        payload = {
            "tinystress_parquet": args.tinystress_parquet,
            "filter": {
                "min_emphasis": args.min_emphasis,
                "max_emphasis": args.max_emphasis,
                "num_samples_total": len(keep.universe),
                "num_samples_kept": len(keep.subset),
            },
            "systems": systems,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nFull report written to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
