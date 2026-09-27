import csv
from collections import defaultdict
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

INPUT_CSV = BASE / "scoring_results.csv"
OUTPUT_CSV = BASE / "model_summary.csv"


# ============================================================
# METRICS TO AVERAGE ACROSS RFPs
#
# This reproduces the table exactly as previously calculated:
# simple arithmetic mean across the 5 RFP rows for each model.
# ============================================================

AVERAGE_METRICS = [
    "field_accuracy_pct",
    "avg_field_similarity_pct",
    "completeness_pct",
    "hallucination_rate_pct",
    "wrong_value_rate_pct",
    "latency_seconds",
    "total_tokens",
    "raw_json_valid_pct",
    "json_after_cleanup_pct",
]


# ============================================================
# OUTPUT COLUMN NAMES
# ============================================================

OUTPUT_COLUMNS = [
    "model",
    "rfps_evaluated",
    "field_accuracy_pct",
    "avg_field_similarity_pct",
    "completeness_pct",
    "hallucination_rate_pct",
    "wrong_value_rate_pct",
    "avg_latency_seconds",
    "avg_total_tokens",
    "raw_json_valid_pct",
    "json_after_cleanup_pct",
]


# ============================================================
# HELPERS
# ============================================================

def mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def load_scoring_results():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}"
        )

    with INPUT_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(
            f"{INPUT_CSV} is empty."
        )

    return rows


# ============================================================
# BUILD MODEL SUMMARY
# ============================================================

def build_model_summary(rows):
    grouped = defaultdict(list)

    for row in rows:
        model = row["model"].strip()

        if not model:
            continue

        grouped[model].append(row)

    summary_rows = []

    for model, model_rows in grouped.items():

        # --------------------------------------------
        # Simple arithmetic mean across RFP rows
        # --------------------------------------------

        field_accuracy = mean(
            [
                float(r["field_accuracy_pct"])
                for r in model_rows
            ]
        )

        avg_similarity = mean(
            [
                float(r["avg_field_similarity_pct"])
                for r in model_rows
            ]
        )

        completeness = mean(
            [
                float(r["completeness_pct"])
                for r in model_rows
            ]
        )

        hallucination = mean(
            [
                float(r["hallucination_rate_pct"])
                for r in model_rows
            ]
        )

        wrong_value = mean(
            [
                float(r["wrong_value_rate_pct"])
                for r in model_rows
            ]
        )

        avg_latency = mean(
            [
                float(r["latency_seconds"])
                for r in model_rows
            ]
        )

        avg_tokens = mean(
            [
                float(r["total_tokens"])
                for r in model_rows
            ]
        )

        raw_json_valid = mean(
            [
                float(r["raw_json_valid_pct"])
                for r in model_rows
            ]
        )

        json_after_cleanup = mean(
            [
                float(r["json_after_cleanup_pct"])
                for r in model_rows
            ]
        )

        summary_rows.append(
            {
                "model": model,
                "rfps_evaluated": len(model_rows),

                "field_accuracy_pct": round(
                    field_accuracy,
                    2,
                ),

                "avg_field_similarity_pct": round(
                    avg_similarity,
                    2,
                ),

                "completeness_pct": round(
                    completeness,
                    2,
                ),

                "hallucination_rate_pct": round(
                    hallucination,
                    2,
                ),

                "wrong_value_rate_pct": round(
                    wrong_value,
                    2,
                ),

                "avg_latency_seconds": round(
                    avg_latency,
                    2,
                ),

                "avg_total_tokens": round(
                    avg_tokens
                ),

                "raw_json_valid_pct": round(
                    raw_json_valid,
                    2,
                ),

                "json_after_cleanup_pct": round(
                    json_after_cleanup,
                    2,
                ),
            }
        )

    # Stable output order
    preferred_order = {
        "Granite-8B-AWQ": 0,
        "OpenAI": 1,
        "Qwen-4B-AWQ": 2,
    }

    summary_rows.sort(
        key=lambda r: (
            preferred_order.get(
                r["model"],
                999,
            ),
            r["model"],
        )
    )

    return summary_rows


# ============================================================
# SAVE CSV
# ============================================================

def save_summary(summary_rows):
    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )


# ============================================================
# PRINT SUMMARY FOR VERIFICATION
# ============================================================

def print_summary(summary_rows):
    print("\nMODEL SUMMARY")
    print("=" * 100)

    for row in summary_rows:
        print(
            f"{row['model']}: "
            f"accuracy={row['field_accuracy_pct']}% | "
            f"similarity={row['avg_field_similarity_pct']}% | "
            f"completeness={row['completeness_pct']}% | "
            f"hallucination={row['hallucination_rate_pct']}% | "
            f"wrong_value={row['wrong_value_rate_pct']}% | "
            f"latency={row['avg_latency_seconds']}s | "
            f"avg_tokens={row['avg_total_tokens']} | "
            f"raw_json={row['raw_json_valid_pct']}% | "
            f"cleaned_json={row['json_after_cleanup_pct']}%"
        )

    print("=" * 100)


# ============================================================
# MAIN
# ============================================================

def main():
    rows = load_scoring_results()

    summary_rows = build_model_summary(
        rows
    )

    save_summary(
        summary_rows
    )

    print_summary(
        summary_rows
    )

    print(
        f"\nSaved: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()
