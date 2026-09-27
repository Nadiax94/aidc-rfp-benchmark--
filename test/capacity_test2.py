import csv
import json
import math
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


# ============================================================
# FULL-RFP CAPACITY TEST
#
# IMPORTANT:
# One "request" in this test = one COMPLETE V5 RFP extraction.
#
# Example:
#   concurrency=5
#   -> launch 5 full RFP jobs together
#   -> each RFP is chunked exactly like V5
#   -> wait until ALL 5 full RFP jobs finish
#   -> then start concurrency=7
#
# This is intentionally different from the earlier lightweight
# capacity test, which used one short API call per request.
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE.parent / "documents"

SUMMARY_CSV = BASE / "capacity_full_rfp_summary.csv"
RESULTS_JSON = BASE / "capacity_full_rfp_results.json"


# ============================================================
# CAPACITY LEVELS
# ============================================================

CONCURRENCY_LEVELS = [
    30, 35, 40
]

DOCUMENTS = [
    "rfp1.txt",
    "rfp2.txt",
    "rfp3.txt",
    "rfp4.txt",
    "rfp5.txt",
]

# None means:
# concurrency=5  -> exactly 5 full RFP jobs
# concurrency=7  -> exactly 7 full RFP jobs
# concurrency=15 -> exactly 15 full RFP jobs
#
# For levels > 5, the five RFPs are cycled/repeated.
REQUESTS_PER_LEVEL = None


# ============================================================
# V5 EXTRACTION SETTINGS
#
# These intentionally match your V5 benchmark architecture.
# ============================================================

CHUNK_SIZE_CHARS = 8000
OVERLAP_CHARS = 150

# Each full RFP job may process this many chunks internally
# at the same time.
#
# Therefore:
# document concurrency=5 with CHUNK_WORKERS=2
# may produce up to roughly 10 simultaneous chunk API calls.
CHUNK_WORKERS = 2

MAX_OUTPUT_TOKENS = 650
REQUEST_TIMEOUT_SECONDS = 180

RECOVERY_OVERLAP_CHARS = 100
MAX_CANDIDATES_PER_FIELD = 4
SOURCE_EXCERPT_CHARS = 900


# ============================================================
# MODELS
#
# By default this capacity test runs local models only.
# OpenAI is included but disabled.
# ============================================================

MODELS = {
    "Granite-8B-AWQ": {
        "enabled": False,
        "provider": "local",
        "url": "http://localhost:9000/v1",
        "model": "drawais/Granite-4.1-8B-AWQ-INT4",
    },
    "Qwen-4B-AWQ": {
        "enabled": True,
        "provider": "local",
        "url": "http://localhost:9001/v1",
        "model": "cyankiwi/Qwen3.5-4B-AWQ-4bit",
    },

}


# ============================================================
# V5 OUTPUT FIELDS
# ============================================================

FIELDS = [
    "RFP Name",
    "Organization",
    "Issue Date",
    "Start Date",
    "End Date",
    "Submission Deadline (date & time)",
    "Deadline for Questions / Inquiries",
    "Budget",
    "RFP Contact (name and/or email)",
    "Submission Method / Portal",
    "Contract Term / Duration (including renewal options)",
    "Scope of Deliverables / Services Requested",
    "Mandatory Submission Requirements",
    "Mandatory Technical Requirements",
    "Evaluation Criteria & Weighting (points breakdown by category)",
    "Minimum Score Threshold to Advance (where a pass/fail cutoff is specified)",
    "Pricing Structure / Cost Submission Requirements",
    "Minimum Insurance Coverage Requirements (type and dollar amount)",
    "Required Vendor Experience / Qualifications (e.g., years of experience)",
    "Number of References Required",
    "Data Security / Privacy Compliance Requirements",
    "Data Hosting / Residency Requirements (where specified)",
    "Vendor Demonstration Requirement",
    "Summary",
]


# ============================================================
# V5 EXTRACTION PROMPT
# ============================================================

FIELD_LIST = "\n".join(
    f"- {field}"
    for field in FIELDS
)

PROMPT = f"""Extract RFP facts from the document chunk below.

Return ONE valid JSON object only.
No markdown. No explanation. No extra text.

STRICT RULES:
- Use ONLY information explicitly supported by this chunk.
- Do NOT guess, infer, assume, or calculate missing information.
- Use ONLY the exact field names listed below.
- If a field is unsupported, OMIT THE FIELD COMPLETELY.
- NEVER return null.
- NEVER return "Not Specified" or any similar missing-value phrase.
- Returning unsupported/missing fields is an error.
- Keep values concise and preserve exact dates, times, numbers, percentages,
  monetary amounts, emails, URLs, thresholds, and required counts.
- Start Date = overall contract/project start only; do not use a phase date,
  milestone date, deadline, vendor-selection date, or expected completion date.
- End Date = overall contract/project end only; do not use a phase date,
  milestone date, deadline, vendor-selection date, or expected completion date.
- Budget = explicitly stated project/contract budget only.
- RFP Contact = official issuing/questions contact only.
- Submission Method / Portal = how the proposal itself must be submitted.
- Contract Term / Duration = overall contract duration and renewal options.
- Number of References Required = exact required/minimum count when stated.
- Evaluation Criteria & Weighting = actual criteria/points/weights, not just
  a section heading.
- Mandatory Submission Requirements = actual required proposal items/actions,
  not only a section heading.
- Mandatory Technical Requirements = actual technical/system requirements,
  not pricing, vendor experience, or submission-format requirements.
- Summary = one concise summary of the RFP purpose.
- Do not repeat the same information inside a field.

Allowed fields:
{FIELD_LIST}

DOCUMENT CHUNK:
"""


# ============================================================
# MISSING VALUE CLEANUP
# ============================================================

MISSING_PREFIXES = (
    "not specified",
    "not stated",
    "not provided",
    "not available",
    "not found",
    "not explicitly provided",
    "not explicitly stated",
    "not mentioned",
    "not included",
    "no information",
    "none specified",
)


def clean_value(value):
    if value is None:
        return None

    if isinstance(value, (dict, list)):
        try:
            value = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except Exception:
            value = str(value)

    value = str(value).strip()

    if not value:
        return None

    simplified = re.sub(
        r"\s+",
        " ",
        value.lower(),
    ).strip(" .;:")

    if simplified in {
        "null",
        "none",
        "n/a",
        "na",
    }:
        return None

    if any(
        simplified.startswith(prefix)
        for prefix in MISSING_PREFIXES
    ):
        return None

    return value


def normalize_text(value):
    value = clean_value(value)

    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        value.lower(),
    ).strip(" .;:")


# ============================================================
# CHUNKING
# ============================================================

def split_text(text):
    text = text.strip()

    if not text:
        return []

    if len(text) <= CHUNK_SIZE_CHARS:
        return [text]

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        hard_end = min(
            start + CHUNK_SIZE_CHARS,
            n,
        )

        if hard_end == n:
            end = n

        else:
            search_start = max(
                start + CHUNK_SIZE_CHARS - 1200,
                start,
            )

            window = text[
                search_start:hard_end
            ]

            cut = window.rfind("\n\n")

            if cut < 0:
                cut = window.rfind("\n")

            if cut >= 0:
                end = (
                    search_start
                    + cut
                )
            else:
                end = hard_end

            if end <= start + 1000:
                end = hard_end

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(
            end - OVERLAP_CHARS,
            start + 1,
        )

    return chunks


def split_failed_chunk(chunk):
    if len(chunk) < 2000:
        return [chunk]

    midpoint = (
        len(chunk) // 2
    )

    search_radius = min(
        800,
        midpoint,
    )

    left_bound = max(
        0,
        midpoint - search_radius,
    )

    right_bound = min(
        len(chunk),
        midpoint + search_radius,
    )

    window = chunk[
        left_bound:right_bound
    ]

    local_mid = (
        midpoint
        - left_bound
    )

    cut = window.rfind(
        "\n",
        0,
        local_mid,
    )

    if cut < 0:
        cut = window.find(
            "\n",
            local_mid,
        )

    if cut >= 0:
        cut = (
            left_bound
            + cut
        )
    else:
        cut = midpoint

    first = chunk[
        :cut
    ].strip()

    second_start = max(
        0,
        cut - RECOVERY_OVERLAP_CHARS,
    )

    second = chunk[
        second_start:
    ].strip()

    return [
        part
        for part in (
            first,
            second,
        )
        if part
    ]


# ============================================================
# JSON PARSING
# ============================================================

def parse_json(text):
    if not text:
        return None, False

    cleaned = text.strip()

    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
    )

    cleaned = (
        cleaned
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    try:
        obj = json.loads(
            cleaned
        )

        if isinstance(
            obj,
            dict,
        ):
            return obj, True

    except Exception:
        pass

    decoder = (
        json.JSONDecoder()
    )

    for i, c in enumerate(
        cleaned
    ):
        if c != "{":
            continue

        try:
            obj, _ = (
                decoder.raw_decode(
                    cleaned[i:]
                )
            )

            if isinstance(
                obj,
                dict,
            ):
                return obj, True

        except Exception:
            pass

    return None, False


# ============================================================
# MODEL CLIENT
# ============================================================

def make_client(
    model_name,
    info,
):
    if (
        info["provider"]
        == "openai"
    ):
        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY "
                "is not set."
            )

        return OpenAI(
            api_key=api_key,
            timeout=(
                REQUEST_TIMEOUT_SECONDS
            ),
        )

    return OpenAI(
        base_url=info["url"],
        api_key="none",
        timeout=(
            REQUEST_TIMEOUT_SECONDS
        ),
    )


# ============================================================
# ONE CHUNK MODEL REQUEST
# ============================================================

def run_extraction_once(
    model_name,
    info,
    chunk,
):
    client = make_client(
        model_name,
        info,
    )

    content = (
        PROMPT
        + chunk
    )

    start = (
        time.perf_counter()
    )

    if (
        info["provider"]
        == "openai"
    ):
        response = (
            client.responses.create(
                model=info["model"],
                input=content,
                max_output_tokens=(
                    MAX_OUTPUT_TOKENS
                ),
                reasoning={
                    "effort": "none",
                },
            )
        )

        raw = (
            response.output_text
            or ""
        )

    else:
        kwargs = {
            "model":
                info["model"],

            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],

            "temperature": 0,
            "max_tokens":
                MAX_OUTPUT_TOKENS,
        }

        if (
            "Qwen"
            in model_name
        ):
            kwargs[
                "extra_body"
            ] = {
                "chat_template_kwargs": {
                    "enable_thinking":
                        False
                }
            }

        response = (
            client
            .chat
            .completions
            .create(
                **kwargs
            )
        )

        raw = (
            response
            .choices[0]
            .message
            .content
            or ""
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    parsed, ok = (
        parse_json(raw)
    )

    return {
        "time": elapsed,
        "parsed": parsed,
        "cleaned_ok": ok,
        "raw": raw,
    }


# ============================================================
# INVALID CHUNK RECOVERY
# ============================================================

def run_chunk_with_recovery(
    model_name,
    info,
    chunk,
):
    first = (
        run_extraction_once(
            model_name,
            info,
            chunk,
        )
    )

    if first["cleaned_ok"]:
        return [
            {
                "parsed":
                    first["parsed"],
                "source_text":
                    chunk,
            }
        ]

    recovery_parts = (
        split_failed_chunk(
            chunk
        )
    )

    recovered = []

    for part in (
        recovery_parts
    ):
        rr = (
            run_extraction_once(
                model_name,
                info,
                part,
            )
        )

        if rr["cleaned_ok"]:
            recovered.append(
                {
                    "parsed":
                        rr["parsed"],
                    "source_text":
                        part,
                }
            )

    return recovered


# ============================================================
# CANDIDATE HELPERS
# ============================================================

def keyword_tokens(text):
    words = re.findall(
        r"[a-z0-9@.$%+-]{4,}",
        str(text).lower(),
    )

    seen = []

    for word in words:
        if word not in seen:
            seen.append(word)

    return seen


def make_source_excerpt(
    source_text,
    field,
    value,
):
    source_text = str(
        source_text
        or ""
    )

    if len(source_text) <= (
        SOURCE_EXCERPT_CHARS
    ):
        return (
            source_text.strip()
        )

    lower = (
        source_text.lower()
    )

    words = (
        keyword_tokens(
            value
        )[:10]
    )

    positions = []

    for word in words:
        pos = lower.find(
            word
        )

        if pos >= 0:
            positions.append(
                pos
            )

    if positions:
        center = min(
            positions
        )
    else:
        center = (
            len(source_text)
            // 2
        )

    half = (
        SOURCE_EXCERPT_CHARS
        // 2
    )

    start = max(
        0,
        center - half,
    )

    end = min(
        len(source_text),
        start
        + SOURCE_EXCERPT_CHARS,
    )

    start = max(
        0,
        end
        - SOURCE_EXCERPT_CHARS,
    )

    return source_text[
        start:end
    ].strip()


def new_candidate_map():
    return {
        field: []
        for field in FIELDS
    }


def add_candidate(
    candidates,
    field,
    value,
    source_text,
):
    if field not in FIELDS:
        return

    value = clean_value(
        value
    )

    if value is None:
        return

    normalized = (
        normalize_text(
            value
        )
    )

    if not normalized:
        return

    for item in (
        candidates[field]
    ):
        if (
            normalize_text(
                item["value"]
            )
            == normalized
        ):
            return

    if (
        len(candidates[field])
        >= MAX_CANDIDATES_PER_FIELD
    ):
        return

    candidates[field].append(
        {
            "value": value,
            "excerpt":
                make_source_excerpt(
                    source_text,
                    field,
                    value,
                ),
        }
    )


# ============================================================
# V5 DETERMINISTIC FIELD SELECTION
# ============================================================

def contains_any(
    text,
    terms,
):
    text = text.lower()

    return any(
        term in text
        for term in terms
    )


def count_matches(
    text,
    terms,
):
    text = text.lower()

    return sum(
        1
        for term in terms
        if term in text
    )


def looks_like_heading_only(
    value
):
    v = normalize_text(
        value
    )

    if not v:
        return True

    heading_terms = (
        "section",
        "appendix",
        "rated criteria",
        "pricing schedule",
        "scope of work",
        "mandatory submission requirements",
        "mandatory technical requirements",
    )

    if (
        len(v) <= 90
        and contains_any(
            v,
            heading_terms,
        )
    ):
        has_detail = bool(
            re.search(
                r"\b\d+(?:\.\d+)?\s*%"
                r"|\$\s*\d"
                r"|\bminimum\b"
                r"|\brequired\b"
                r"|\bmust\b",
                v,
            )
        )

        if not has_detail:
            return True

    return False


def normalize_reference_count(
    value
):
    text = normalize_text(
        value
    )

    digit_match = re.search(
        r"\b(\d+)\b",
        text,
    )

    if digit_match:
        return (
            digit_match.group(1)
        )

    word_numbers = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }

    for word, number in (
        word_numbers.items()
    ):
        if re.search(
            rf"\b{word}\b",
            text,
        ):
            return number

    return None


def field_candidate_score(
    field,
    value,
    excerpt,
):
    v = normalize_text(
        value
    )

    e = normalize_text(
        excerpt
    )

    combined = (
        f"{v} {e}"
    )

    if not v:
        return None

    score = 1.0

    if field == (
        "RFP Contact "
        "(name and/or email)"
    ):
        if re.search(
            r"[A-Za-z0-9._%+-]+"
            r"@[A-Za-z0-9.-]+"
            r"\.[A-Za-z]{2,}",
            value,
        ):
            score += 4

    elif field == (
        "Mandatory Technical Requirements"
    ):
        if looks_like_heading_only(
            value
        ):
            return None

        good = (
            "technical",
            "integration",
            "system",
            "security",
            "encryption",
            "api",
            "cloud",
            "hosting",
            "architecture",
            "compatibility",
            "testing",
            "validation",
        )

        bad = (
            "unit price",
            "total bid",
            "pricing schedule",
            "years of experience",
            "client references",
        )

        score += (
            2
            * count_matches(
                combined,
                good,
            )
        )

        score -= (
            3
            * count_matches(
                combined,
                bad,
            )
        )

        if score <= 0:
            return None

    elif field == (
        "Evaluation Criteria & Weighting "
        "(points breakdown by category)"
    ):
        if looks_like_heading_only(
            value
        ):
            return None

        if not (
            re.search(
                r"\b\d+(?:\.\d+)?\s*%",
                value,
            )
            or re.search(
                r"\b\d+(?:\.\d+)?\s*"
                r"(?:points?|pts?)\b",
                value.lower(),
            )
        ):
            return None

        score += 3

    elif field == (
        "Pricing Structure / "
        "Cost Submission Requirements"
    ):
        if looks_like_heading_only(
            value
        ):
            return None

        terms = (
            "price",
            "pricing",
            "cost",
            "rate",
            "fee",
            "unit price",
            "total bid",
            "currency",
            "canadian",
            "tax",
        )

        matches = (
            count_matches(
                combined,
                terms,
            )
        )

        if matches == 0:
            return None

        score += (
            2 * matches
        )

    elif field == (
        "Required Vendor Experience / "
        "Qualifications "
        "(e.g., years of experience)"
    ):
        terms = (
            "experience",
            "years",
            "qualification",
            "certification",
            "project manager",
            "team member",
            "similar projects",
        )

        matches = (
            count_matches(
                combined,
                terms,
            )
        )

        if matches == 0:
            return None

        score += (
            2 * matches
        )

    elif field == (
        "Number of References Required"
    ):
        count = (
            normalize_reference_count(
                value
            )
        )

        if (
            count is None
            or "reference"
            not in combined
        ):
            return None

        score += 4

    elif field == (
        "Data Security / "
        "Privacy Compliance Requirements"
    ):
        terms = (
            "security",
            "privacy",
            "encryption",
            "confidential",
            "cyber",
            "soc2",
            "data protection",
            "access control",
        )

        matches = (
            count_matches(
                combined,
                terms,
            )
        )

        if matches == 0:
            return None

        score += (
            2 * matches
        )

    elif field == (
        "Data Hosting / "
        "Residency Requirements "
        "(where specified)"
    ):
        terms = (
            "hosting",
            "hosted",
            "residency",
            "data centre",
            "data center",
            "stored in canada",
            "hosted in canada",
        )

        matches = (
            count_matches(
                combined,
                terms,
            )
        )

        if matches == 0:
            return None

        score += (
            2 * matches
        )

    elif field == (
        "Vendor Demonstration Requirement"
    ):
        if not contains_any(
            combined,
            (
                "demonstration",
                "demo",
                "presentation",
                "interview",
                "shortlisted",
            ),
        ):
            return None

        score += 4

    elif field == (
        "Scope of Deliverables / "
        "Services Requested"
    ):
        if looks_like_heading_only(
            value
        ):
            return None

        score += 1

    elif field == (
        "Mandatory Submission Requirements"
    ):
        if looks_like_heading_only(
            value
        ):
            return None

        score += 1

    elif field == "Summary":
        if len(v) < 30:
            return None

    # Prefer concise candidate when otherwise similar.
    if len(v) <= 500:
        score += 0.5

    return score


def select_final_output(
    candidates
):
    final = {
        field: "Not Specified"
        for field in FIELDS
    }

    for field in FIELDS:
        items = (
            candidates[field]
        )

        if not items:
            continue

        scored = []

        for index, item in enumerate(
            items
        ):
            score = (
                field_candidate_score(
                    field,
                    item["value"],
                    item["excerpt"],
                )
            )

            if score is None:
                continue

            scored.append(
                (
                    score,
                    -index,
                    item,
                )
            )

        if not scored:
            continue

        scored.sort(
            key=lambda x: (
                x[0],
                x[1],
            ),
            reverse=True,
        )

        value = (
            scored[0][2]["value"]
        )

        if field == (
            "Number of References Required"
        ):
            count = (
                normalize_reference_count(
                    value
                )
            )

            if count:
                value = count

        final[field] = value

    return final


# ============================================================
# ONE COMPLETE V5 RFP EXTRACTION
#
# This is the "request" measured by the capacity test.
# ============================================================

def run_full_rfp_job(
    model_name,
    info,
    job_id,
    document,
):
    job_start = (
        time.perf_counter()
    )

    try:
        chunks = split_text(
            document["text"]
        )

        candidates = (
            new_candidate_map()
        )

        # Internal chunk-level concurrency,
        # exactly like V5.
        workers = min(
            CHUNK_WORKERS,
            max(
                len(chunks),
                1,
            ),
        )

        chunk_results = {}

        with ThreadPoolExecutor(
            max_workers=workers
        ) as executor:

            futures = {
                executor.submit(
                    run_chunk_with_recovery,
                    model_name,
                    info,
                    chunk,
                ): i
                for i, chunk in enumerate(
                    chunks,
                    1,
                )
            }

            for future in (
                as_completed(
                    futures
                )
            ):
                i = futures[
                    future
                ]

                chunk_results[
                    i
                ] = (
                    future.result()
                )

        for i in sorted(
            chunk_results
        ):
            parsed_parts = (
                chunk_results[i]
            )

            for item in (
                parsed_parts
            ):
                parsed = item[
                    "parsed"
                ]

                source_text = item[
                    "source_text"
                ]

                if not isinstance(
                    parsed,
                    dict,
                ):
                    continue

                for (
                    field,
                    value,
                ) in parsed.items():
                    add_candidate(
                        candidates,
                        field,
                        value,
                        source_text,
                    )

        # Same Python-only final selection idea as V5.
        _final_output = (
            select_final_output(
                candidates
            )
        )

        success = True
        error = ""

    except Exception as exc:
        success = False
        error = str(exc)

    latency = (
        time.perf_counter()
        - job_start
    )

    return {
        "job_id": job_id,
        "document":
            document["filename"],
        "success": success,
        "latency_seconds":
            round(
                latency,
                4,
            ),
        "error": error,
    }


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():
    documents = []

    for filename in DOCUMENTS:
        path = (
            DOCUMENTS_DIR
            / filename
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: "
                f"{path}"
            )

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        documents.append(
            {
                "filename":
                    filename,
                "text":
                    text,
            }
        )

    return documents


# ============================================================
# P95
# ============================================================

def percentile(
    values,
    percentile_value,
):
    if not values:
        return 0.0

    values = sorted(
        values
    )

    if len(values) == 1:
        return values[0]

    k = (
        (len(values) - 1)
        * percentile_value
        / 100
    )

    floor_index = (
        math.floor(k)
    )

    ceil_index = (
        math.ceil(k)
    )

    if (
        floor_index
        == ceil_index
    ):
        return values[
            int(k)
        ]

    lower = values[
        floor_index
    ]

    upper = values[
        ceil_index
    ]

    return (
        lower
        * (ceil_index - k)
        + upper
        * (k - floor_index)
    )


# ============================================================
# RUN ONE DOCUMENT-CONCURRENCY LEVEL
#
# concurrency=5:
#   launch 5 COMPLETE RFP EXTRACTIONS together
#   wait until all 5 are completely done
#   then return
# ============================================================

def run_capacity_level(
    model_name,
    info,
    documents,
    concurrency,
):
    requests_total = (
        REQUESTS_PER_LEVEL
        if REQUESTS_PER_LEVEL
        is not None
        else concurrency
    )

    jobs = []

    for i in range(
        requests_total
    ):
        jobs.append(
            {
                "job_id":
                    i + 1,

                "document":
                    documents[
                        i
                        % len(documents)
                    ],
            }
        )

    print(
        "\n"
        + "-" * 72
    )

    print(
        f"{model_name}"
        f" | FULL-RFP concurrency="
        f"{concurrency}"
        f" | full RFP jobs="
        f"{requests_total}"
    )

    print(
        "-" * 72
    )

    level_start = (
        time.perf_counter()
    )

    results = []

    # Outer executor = concurrent COMPLETE RFP jobs.
    with ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = {
            executor.submit(
                run_full_rfp_job,
                model_name,
                info,
                job["job_id"],
                job["document"],
            ): job
            for job in jobs
        }

        for future in (
            as_completed(
                futures
            )
        ):
            result = (
                future.result()
            )

            results.append(
                result
            )

            print(
                f"  full job "
                f"{result['job_id']}"
                f" | "
                f"{result['document']}"
                f" | success="
                f"{result['success']}"
                f" | full latency="
                f"{result['latency_seconds']:.2f}s"
            )

            if (
                not result["success"]
            ):
                print(
                    f"    ERROR: "
                    f"{result['error']}"
                )

    # The next concurrency level cannot begin before this.
    total_test_time = (
        time.perf_counter()
        - level_start
    )

    successful = [
        r
        for r in results
        if r["success"]
    ]

    failed = [
        r
        for r in results
        if not r["success"]
    ]

    latencies = [
        r["latency_seconds"]
        for r in successful
    ]

    successful_count = len(
        successful
    )

    failed_count = len(
        failed
    )

    success_rate = (
        successful_count
        / requests_total
        * 100
        if requests_total
        else 0
    )

    avg_latency = (
        statistics.mean(
            latencies
        )
        if latencies
        else 0
    )

    p95_latency = (
        percentile(
            latencies,
            95,
        )
    )

    throughput = (
        successful_count
        / total_test_time
        * 60
        if total_test_time > 0
        else 0
    )

    summary = {
        "model":
            model_name,

        "concurrency":
            concurrency,

        "requests_total":
            requests_total,

        "requests_successful":
            successful_count,

        "requests_failed":
            failed_count,

        "success_rate_pct":
            round(
                success_rate,
                2,
            ),

        "avg_latency_s":
            round(
                avg_latency,
                4,
            ),

        "p95_latency_s":
            round(
                p95_latency,
                4,
            ),

        "total_test_time_s":
            round(
                total_test_time,
                4,
            ),

        "throughput_requests_per_min":
            round(
                throughput,
                2,
            ),
    }

    print(
        f"\n✅ ALL "
        f"{requests_total} "
        f"full RFP jobs at "
        f"concurrency "
        f"{concurrency} "
        f"finished."
    )

    print(
        f"   success="
        f"{summary['success_rate_pct']}%"
        f" | avg full-RFP latency="
        f"{summary['avg_latency_s']}s"
        f" | p95="
        f"{summary['p95_latency_s']}s"
        f" | throughput="
        f"{summary['throughput_requests_per_min']}"
        f" RFP/min"
    )

    return summary


# ============================================================
# OUTPUT COLUMNS
# ============================================================

SUMMARY_COLUMNS = [
    "model",
    "concurrency",
    "requests_total",
    "requests_successful",
    "requests_failed",
    "success_rate_pct",
    "avg_latency_s",
    "p95_latency_s",
    "total_test_time_s",
    "throughput_requests_per_min",
]


# ============================================================
# LOAD / SAVE RESULTS
# ============================================================

def load_existing_results():
    # Prefer the existing JSON because it preserves numeric types.
    if RESULTS_JSON.exists():
        try:
            with RESULTS_JSON.open(
                "r",
                encoding="utf-8",
            ) as f:
                payload = json.load(f)

            rows = payload.get(
                "summary",
                [],
            )

            if isinstance(rows, list):
                return [
                    {
                        column: row.get(column)
                        for column in SUMMARY_COLUMNS
                    }
                    for row in rows
                    if isinstance(row, dict)
                ]

        except Exception as exc:
            print(
                f"Warning: could not read existing JSON: {exc}"
            )

    # Fall back to CSV if JSON is unavailable.
    if SUMMARY_CSV.exists():
        try:
            with SUMMARY_CSV.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as f:
                reader = csv.DictReader(f)
                rows = []

                int_columns = {
                    "concurrency",
                    "requests_total",
                    "requests_successful",
                    "requests_failed",
                }

                float_columns = {
                    "success_rate_pct",
                    "avg_latency_s",
                    "p95_latency_s",
                    "total_test_time_s",
                    "throughput_requests_per_min",
                }

                for row in reader:
                    clean = {}
                    for column in SUMMARY_COLUMNS:
                        value = row.get(column)

                        if column in int_columns:
                            try:
                                value = int(float(value))
                            except Exception:
                                pass

                        elif column in float_columns:
                            try:
                                value = float(value)
                            except Exception:
                                pass

                        clean[column] = value

                    rows.append(clean)

                return rows

        except Exception as exc:
            print(
                f"Warning: could not read existing CSV: {exc}"
            )

    return []


def save_summary_csv(
    rows
):
    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = (
            csv.DictWriter(
                f,
                fieldnames=(
                    SUMMARY_COLUMNS
                ),
            )
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def save_results_json(
    rows
):
    payload = {
        "configuration": {
            "test_type":
                "full_v5_rfp_extraction",

            "current_run_concurrency_levels":
                CONCURRENCY_LEVELS,

            "completed_concurrency_levels":
                sorted({
                    int(row["concurrency"])
                    for row in rows
                    if row.get("concurrency") is not None
                }),

            "documents":
                DOCUMENTS,

            "requests_per_level":
                REQUESTS_PER_LEVEL,

            "chunk_size_chars":
                CHUNK_SIZE_CHARS,

            "chunk_overlap_chars":
                OVERLAP_CHARS,

            "chunk_workers_per_rfp":
                CHUNK_WORKERS,

            "max_output_tokens":
                MAX_OUTPUT_TOKENS,

            "important_note":
                (
                    "Concurrency refers to "
                    "simultaneous full RFP jobs. "
                    "Each full RFP job may itself "
                    "run up to CHUNK_WORKERS chunk "
                    "API calls concurrently."
                ),
        },

        "summary":
            rows,
    }

    with RESULTS_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    documents = (
        load_documents()
    )

    enabled_models = {
        name: info
        for name, info
        in MODELS.items()
        if info.get(
            "enabled",
            False,
        )
    }

    if not enabled_models:
        raise SystemExit(
            "No models enabled."
        )

    summary_rows = load_existing_results()

    completed_keys = {
        (
            str(row.get("model")),
            int(row.get("concurrency")),
        )
        for row in summary_rows
        if row.get("model") is not None
        and row.get("concurrency") is not None
    }

    if summary_rows:
        print(
            f"\nLoaded {len(summary_rows)} existing result rows. "
            "New results will be added without deleting them."
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FULL V5 RFP CAPACITY TEST"
    )

    print(
        "=" * 72
    )

    print(
        f"Concurrency levels: "
        f"{CONCURRENCY_LEVELS}"
    )

    print(
        f"Models: "
        f"{list(enabled_models)}"
    )

    print(
        "\nOne capacity request = "
        "one COMPLETE V5 RFP extraction."
    )

    print(
        f"Each full RFP internally uses "
        f"up to {CHUNK_WORKERS} "
        f"chunk workers."
    )

    print(
        "\nThe test waits for every "
        "full RFP job at the current "
        "concurrency level to finish "
        "before moving to the next level."
    )

    for (
        model_name,
        info,
    ) in enabled_models.items():

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"MODEL: "
            f"{model_name}"
        )

        print(
            "=" * 72
        )

        for (
            level_index,
            concurrency,
        ) in enumerate(
            CONCURRENCY_LEVELS,
            1,
        ):
            key = (
                model_name,
                concurrency,
            )

            if key in completed_keys:
                print(
                    f"\n⏭️  Skipping {model_name} concurrency={concurrency}: "
                    "already exists in saved results."
                )
                continue

            summary = (
                run_capacity_level(
                    model_name,
                    info,
                    documents,
                    concurrency,
                )
            )

            summary_rows.append(
                summary
            )

            completed_keys.add(key)

            # Preserve completed levels immediately.
            save_summary_csv(
                summary_rows
            )

            save_results_json(
                summary_rows
            )

            if (
                level_index
                < len(
                    CONCURRENCY_LEVELS
                )
            ):
                next_level = (
                    CONCURRENCY_LEVELS[
                        level_index
                    ]
                )

                print(
                    f"\n➡️  Full-RFP "
                    f"concurrency "
                    f"{concurrency} "
                    f"is complete."
                )

                print(
                    f"   Starting "
                    f"concurrency "
                    f"{next_level} "
                    f"next..."
                )

        print(
            f"\n✅ All full-RFP "
            f"capacity levels "
            f"finished for "
            f"{model_name}."
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FULL-RFP CAPACITY "
        "TEST FINISHED ✅"
    )

    print(
        "=" * 72
    )

    print(
        f"CSV: "
        f"{SUMMARY_CSV}"
    )

    print(
        f"JSON: "
        f"{RESULTS_JSON}"
    )


if __name__ == "__main__":
    main()
