import csv
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

GROUND_TRUTH_FILE = BASE / "ground_truth.json"
BENCHMARK_FILE = BASE / "benchmark_results.json"

# Final CSV for dashboard / report / presentation.
RESULTS_CSV = BASE / "scoring_results.csv"


# ============================================================
# CHOOSE WHICH DOCUMENT(S) TO SCORE
#
# First run:
#   ["rfp1.txt"]
#
# Later, score all five:
#   ["rfp1.txt", "rfp2.txt", "rfp3.txt", "rfp4.txt", "rfp5.txt"]
#
# Existing CSV rows are preserved and updated by rfp + model.
# ============================================================

DOCUMENTS_TO_SCORE = [
    "rfp1.txt",
    "rfp2.txt",
    "rfp3.txt",
    "rfp4.txt",
    "rfp5.txt",
]


# ============================================================
# MISSING VALUES
# ============================================================

MISSING_PHRASES = {
    "",
    "null",
    "none",
    "n/a",
    "na",
    "not applicable",
    "not specified",
    "not stated",
    "not provided",
    "not available",
    "not found",
    "not explicitly provided",
    "not explicitly stated",
    "not specified in the rfp",
    "not stated in the rfp",
    "not provided in the rfp",
}


# ============================================================
# LONG / DESCRIPTIVE FIELDS
#
# These allow looser wording because correct answers can be
# paraphrased while preserving the same meaning.
# ============================================================

LONG_FIELDS = {
    "Contract Term / Duration (including renewal options)",
    "Scope of Deliverables / Services Requested",
    "Mandatory Submission Requirements",
    "Mandatory Technical Requirements",
    "Evaluation Criteria & Weighting (points breakdown by category)",
    "Minimum Score Threshold to Advance (where a pass/fail cutoff is specified)",
    "Pricing Structure / Cost Submission Requirements",
    "Minimum Insurance Coverage Requirements (type and dollar amount)",
    "Required Vendor Experience / Qualifications (e.g., years of experience)",
    "Data Security / Privacy Compliance Requirements",
    "Data Hosting / Residency Requirements (where specified)",
    "Vendor Demonstration Requirement",
    "Summary",
}


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def clean_missing_text(value):
    if value is None:
        return ""

    s = str(value).strip()

    simple = re.sub(
        r"\s+",
        " ",
        s.lower(),
    ).strip(" .;:")

    if simple in MISSING_PHRASES:
        return ""

    return s


def is_missing(value):
    return clean_missing_text(value) == ""


def normalize(value):
    s = clean_missing_text(value)

    if not s:
        return ""

    s = unicodedata.normalize(
        "NFKD",
        s,
    )

    s = (
        s.replace("–", "-")
        .replace("—", "-")
        .replace("&", " and ")
    )

    s = s.lower()

    s = re.sub(
        r"https?://",
        "",
        s,
    )

    s = re.sub(
        r"[^a-z0-9@.$%:/+-]+",
        " ",
        s,
    )

    s = re.sub(
        r"\s+",
        " ",
        s,
    ).strip()

    return s


def tokens(value):
    return re.findall(
        r"[a-z0-9@.$%+-]+",
        normalize(value),
    )


def token_f1(expected, predicted):
    a = set(tokens(expected))
    b = set(tokens(predicted))

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    common = len(a & b)

    precision = (
        common / len(b)
    )

    recall = (
        common / len(a)
    )

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def emails(value):
    return set(
        re.findall(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            str(value or ""),
        )
    )


def numbers(value):
    return re.findall(
        r"\b\d+(?:\.\d+)?\b",
        normalize(value),
    )


# ============================================================
# FIELD SIMILARITY
# ============================================================

def field_similarity(
    field,
    expected,
    predicted,
):
    expected_missing = (
        is_missing(expected)
    )

    predicted_missing = (
        is_missing(predicted)
    )

    # Correctly left empty.
    if expected_missing:
        return (
            1.0
            if predicted_missing
            else 0.0
        )

    # Omission.
    if predicted_missing:
        return 0.0

    e = normalize(expected)
    p = normalize(predicted)

    if e == p:
        return 1.0

    # --------------------------------------------------------
    # Contact:
    # matching the expected email is strong evidence.
    # --------------------------------------------------------

    if field == (
        "RFP Contact (name and/or email)"
    ):
        expected_emails = (
            emails(expected)
        )

        predicted_emails = (
            emails(predicted)
        )

        if (
            expected_emails
            and expected_emails.issubset(
                predicted_emails
            )
        ):
            return 1.0

    # --------------------------------------------------------
    # Number of References:
    # "minimum of three (3)" and "3" should match.
    # --------------------------------------------------------

    if field == (
        "Number of References Required"
    ):
        expected_numbers = (
            numbers(expected)
        )

        predicted_numbers = (
            numbers(predicted)
        )

        if (
            expected_numbers
            and predicted_numbers
            and expected_numbers[0]
            == predicted_numbers[0]
        ):
            return 1.0

    sequence_score = (
        SequenceMatcher(
            None,
            e,
            p,
        ).ratio()
    )

    token_score = token_f1(
        expected,
        predicted,
    )

    return max(
        sequence_score,
        token_score,
    )


def field_is_correct(
    field,
    expected,
    predicted,
):
    similarity = field_similarity(
        field,
        expected,
        predicted,
    )

    # Long descriptive fields allow paraphrasing.
    threshold = (
        0.45
        if field in LONG_FIELDS
        else 0.70
    )

    return (
        similarity >= threshold,
        similarity,
    )


# ============================================================
# CSV HELPERS
# ============================================================

CSV_COLUMNS = [
    "rfp",
    "model",

    # ----- 5 MAIN QUALITY RESULTS -----
    "field_accuracy_pct",
    "avg_field_similarity_pct",
    "completeness_pct",
    "hallucination_rate_pct",
    "wrong_value_rate_pct",

    # ----- COUNTS FOR INTERPRETATION -----
    "fields_correct",
    "fields_total",
    "expected_nonmissing_fields",
    "expected_found_fields",
    "gt_null_fields",
    "hallucinated_null_fields",
    "wrong_value_fields",
    "omitted_expected_fields",

    # ----- BENCHMARK METRICS FOR DASHBOARD -----
    "latency_seconds",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "chunks_total",
    "chunks_successful",
    "raw_json_valid_pct",
    "json_after_cleanup_pct",
]


def load_existing_csv():
    if not RESULTS_CSV.exists():
        return []

    with RESULTS_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def save_csv(rows):
    with RESULTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)


def upsert_rows(
    existing_rows,
    new_rows,
):
    by_key = {}

    for row in existing_rows:
        key = (
            row.get("rfp"),
            row.get("model"),
        )

        by_key[key] = row

    for row in new_rows:
        key = (
            row["rfp"],
            row["model"],
        )

        by_key[key] = row

    return sorted(
        by_key.values(),
        key=lambda x: (
            x["rfp"],
            x["model"],
        ),
    )


# ============================================================
# VALIDATE INPUT FILES
# ============================================================

def validate_inputs(
    ground_truth,
    benchmark,
):
    allowed = set(
        ground_truth.keys()
    )

    if not DOCUMENTS_TO_SCORE:
        raise SystemExit(
            "DOCUMENTS_TO_SCORE is empty."
        )

    unknown = [
        name
        for name in DOCUMENTS_TO_SCORE
        if name not in allowed
    ]

    if unknown:
        raise SystemExit(
            f"Unknown RFP(s) in "
            f"DOCUMENTS_TO_SCORE: {unknown}"
        )

    missing_benchmark = [
        name
        for name in DOCUMENTS_TO_SCORE
        if name not in benchmark
    ]

    if missing_benchmark:
        raise SystemExit(
            "These RFPs are not present in "
            f"benchmark_results.json: "
            f"{missing_benchmark}"
        )

    schemas = {
        tuple(
            data.keys()
        )
        for data in ground_truth.values()
    }

    if len(schemas) != 1:
        raise SystemExit(
            "Ground-truth RFPs do not all "
            "use the same schema."
        )


# ============================================================
# SCORE ONE RFP / MODEL
# ============================================================

def score_run(
    filename,
    model,
    expected,
    run,
):
    predicted = (
        run.get("output")
        or {}
    )

    fields = list(
        expected.keys()
    )

    correct = 0
    similarity_total = 0.0

    expected_nonmissing = 0
    expected_found = 0

    gt_missing = 0
    hallucinated = 0

    wrong_value = 0
    omitted_expected = 0

    for field in fields:
        exp = expected.get(field)
        pred = predicted.get(field)

        ok, similarity = (
            field_is_correct(
                field,
                exp,
                pred,
            )
        )

        correct += int(ok)

        similarity_total += (
            similarity
        )

        # ----------------------------------------------------
        # Ground truth is missing:
        # populated prediction = unsupported-field hallucination
        # ----------------------------------------------------

        if is_missing(exp):
            gt_missing += 1

            if not is_missing(pred):
                hallucinated += 1

        # ----------------------------------------------------
        # Ground truth contains a real value
        # ----------------------------------------------------

        else:
            expected_nonmissing += 1

            if is_missing(pred):
                omitted_expected += 1

            else:
                expected_found += 1

                # Non-empty but incorrect value.
                if not ok:
                    wrong_value += 1

    total_fields = len(fields)

    field_accuracy = (
        correct
        / total_fields
        * 100
        if total_fields
        else 0.0
    )

    avg_similarity = (
        similarity_total
        / total_fields
        * 100
        if total_fields
        else 0.0
    )

    completeness = (
        expected_found
        / expected_nonmissing
        * 100
        if expected_nonmissing
        else 100.0
    )

    # Unsupported-field hallucination rate:
    # GT missing, but model filled the field.
    hallucination_rate = (
        hallucinated
        / gt_missing
        * 100
        if gt_missing
        else 0.0
    )

    # Wrong-value rate:
    # GT has a value, model also returned a value,
    # but it failed the correctness threshold.
    #
    # Denominator = all GT non-missing fields.
    wrong_value_rate = (
        wrong_value
        / expected_nonmissing
        * 100
        if expected_nonmissing
        else 0.0
    )

    return {
        "rfp": filename,
        "model": model,

        # ----- 5 MAIN RESULTS -----
        "field_accuracy_pct":
            round(
                field_accuracy,
                2,
            ),

        "avg_field_similarity_pct":
            round(
                avg_similarity,
                2,
            ),

        "completeness_pct":
            round(
                completeness,
                2,
            ),

        "hallucination_rate_pct":
            round(
                hallucination_rate,
                2,
            ),

        "wrong_value_rate_pct":
            round(
                wrong_value_rate,
                2,
            ),

        # ----- COUNTS -----
        "fields_correct":
            correct,

        "fields_total":
            total_fields,

        "expected_nonmissing_fields":
            expected_nonmissing,

        "expected_found_fields":
            expected_found,

        "gt_null_fields":
            gt_missing,

        "hallucinated_null_fields":
            hallucinated,

        "wrong_value_fields":
            wrong_value,

        "omitted_expected_fields":
            omitted_expected,

        # ----- BENCHMARK DATA -----
        "latency_seconds":
            run.get(
                "time_seconds",
                0,
            ),

        "prompt_tokens":
            run.get(
                "prompt_tokens",
                0,
            ),

        "completion_tokens":
            run.get(
                "completion_tokens",
                0,
            ),

        "total_tokens":
            run.get(
                "total_tokens",
                0,
            ),

        "chunks_total":
            run.get(
                "chunks_total",
                0,
            ),

        "chunks_successful":
            run.get(
                "chunks_successful",
                0,
            ),

        "raw_json_valid_pct":
            run.get(
                "raw_json_valid_rate",
                0,
            ),

        "json_after_cleanup_pct":
            run.get(
                "json_valid_after_cleanup_rate",
                0,
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    with GROUND_TRUTH_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        ground_truth = json.load(f)

    with BENCHMARK_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        benchmark = json.load(f)

    validate_inputs(
        ground_truth,
        benchmark,
    )

    new_rows = []

    for filename in (
        DOCUMENTS_TO_SCORE
    ):
        expected = (
            ground_truth[
                filename
            ]
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"SCORING: {filename}"
        )

        print(
            "=" * 70
        )

        for model, run in (
            benchmark[
                filename
            ].items()
        ):
            row = score_run(
                filename,
                model,
                expected,
                run,
            )

            new_rows.append(row)

            print(
                f"{model}: "
                f"accuracy="
                f"{row['field_accuracy_pct']}% | "
                f"similarity="
                f"{row['avg_field_similarity_pct']}% | "
                f"completeness="
                f"{row['completeness_pct']}% | "
                f"hallucination="
                f"{row['hallucination_rate_pct']}% | "
                f"wrong-value="
                f"{row['wrong_value_rate_pct']}%"
            )

    existing_rows = (
        load_existing_csv()
    )

    final_rows = upsert_rows(
        existing_rows,
        new_rows,
    )

    save_csv(
        final_rows
    )

    print(
        "\nSCORING FINISHED ✅"
    )

    print(
        f"Saved/updated: "
        f"{RESULTS_CSV}"
    )

    print(
        f"Rows in CSV: "
        f"{len(final_rows)}"
    )


if __name__ == "__main__":
    main()
