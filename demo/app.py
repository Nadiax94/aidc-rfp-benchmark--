import json
import re

import streamlit as st
import pandas as pd
import plotly.express as px
import pymupdf
from pathlib import Path
from openai import OpenAI


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RFP AI Extractor & Evaluation Dashboard",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DATA PATHS
#
# Expected structure:
#
# project/
# ├── app.py
# └── datasets/
#     ├── capacity_full_rfp_summary.csv
#     ├── model_summary.csv
#     └── scoring_results.csv
#
# model_summary(1).csv is also accepted automatically.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"

CAPACITY_FILE = DATA_DIR / "capacity_full_rfp_summary.csv"

SUMMARY_CANDIDATES = [
    DATA_DIR / "model_summary.csv",
    DATA_DIR / "model_summary(1).csv",
]

SCORING_FILE = DATA_DIR / "scoring_results.csv"

SUMMARY_FILE = next(
    (p for p in SUMMARY_CANDIDATES if p.exists()),
    SUMMARY_CANDIDATES[0],
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    required = [
        CAPACITY_FILE,
        SUMMARY_FILE,
        SCORING_FILE,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing dataset file(s):\n"
            + "\n".join(missing)
        )

    capacity = pd.read_csv(CAPACITY_FILE)
    summary = pd.read_csv(SUMMARY_FILE)
    scoring = pd.read_csv(SCORING_FILE)

    return capacity, summary, scoring


try:
    capacity_df, summary_df, scoring_df = load_data()
except Exception as exc:
    st.error(
        "Could not load dashboard datasets.\n\n"
        f"{exc}"
    )
    st.stop()


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

REQUIRED_SUMMARY_COLUMNS = {
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
}

REQUIRED_CAPACITY_COLUMNS = {
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
}

REQUIRED_SCORING_COLUMNS = {
    "rfp",
    "model",
    "field_accuracy_pct",
    "avg_field_similarity_pct",
    "completeness_pct",
    "hallucination_rate_pct",
    "wrong_value_rate_pct",
    "latency_seconds",
    "total_tokens",
    "raw_json_valid_pct",
    "json_after_cleanup_pct",
}


def validate_columns(df, required, name):
    missing = sorted(required - set(df.columns))

    if missing:
        st.error(
            f"{name} is missing required columns: "
            + ", ".join(missing)
        )
        st.stop()


validate_columns(
    summary_df,
    REQUIRED_SUMMARY_COLUMNS,
    "model_summary.csv",
)

validate_columns(
    capacity_df,
    REQUIRED_CAPACITY_COLUMNS,
    "capacity_full_rfp_summary.csv",
)

validate_columns(
    scoring_df,
    REQUIRED_SCORING_COLUMNS,
    "scoring_results.csv",
)


# ============================================================
# BASIC CLEANUP
# ============================================================

summary_df = summary_df.copy()
capacity_df = capacity_df.copy()
scoring_df = scoring_df.copy()

capacity_df = capacity_df.sort_values(
    ["model", "concurrency"]
).reset_index(drop=True)

rfp_order = sorted(
    scoring_df["rfp"].dropna().unique(),
    key=lambda x: (
        int("".join(filter(str.isdigit, str(x))) or 999999),
        str(x),
    ),
)

scoring_df["rfp"] = pd.Categorical(
    scoring_df["rfp"],
    categories=rfp_order,
    ordered=True,
)

scoring_df = scoring_df.sort_values(
    ["rfp", "model"]
).reset_index(drop=True)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1400px;
            padding-top: 4.7rem;
            padding-bottom: 4rem;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .top-nav {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 999999;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: .35rem;
            padding: .55rem 1rem;
            background: rgba(15, 17, 22, .94);
            border-bottom: 1px solid rgba(150,150,150,.18);
            backdrop-filter: blur(12px);
            overflow-x: auto;
            white-space: nowrap;
        }

        .top-nav .brand {
            color: #ffffff;
            font-weight: 800;
            margin-right: .7rem;
        }

        .top-nav a {
            color: #e8eaed !important;
            text-decoration: none !important;
            font-size: .88rem;
            font-weight: 600;
            padding: .48rem .72rem;
            border-radius: .55rem;
        }

        .top-nav a:hover {
            background: rgba(255,255,255,.08);
        }

        .section-anchor {
            scroll-margin-top: 78px;
            height: 1px;
        }

        .section-subtitle {
            color: #8b949e;
            margin-top: -.35rem;
            margin-bottom: 1rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,.20);
            border-radius: .85rem;
            padding: .85rem 1rem;
        }

        .info-card {
            border: 1px solid rgba(128,128,128,.18);
            border-radius: .85rem;
            padding: 1rem 1.1rem;
            margin-bottom: .7rem;
            background: rgba(128,128,128,.035);
        }

        .small-note {
            color: #8b949e;
            font-size: .88rem;
            line-height: 1.5;
        }

        @media (max-width: 900px) {
            .top-nav {
                justify-content: flex-start;
            }

            .top-nav .brand {
                display: none;
            }
        }
    </style>

    <nav class="top-nav">
        <span class="brand">RFP Evaluation</span>
        <a href="#extractor">Test Model</a>
        <a href="#overview">Overview</a>
                <a href="#capacity">Capacity & Scalability</a>
        <a href="#rfp-detail">Detailed RFP Analysis</a>
    </nav>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
}


def anchor(name):
    st.markdown(
        f'<div id="{name}" class="section-anchor"></div>',
        unsafe_allow_html=True,
    )


def section_header(name, title, subtitle):
    anchor(name)
    st.markdown(f"## {title}")
    st.markdown(
        f'<div class="section-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def format_percent(value):
    return f"{value:.2f}%"


def format_seconds(value):
    return f"{value:.2f} s"


def model_colors(models):
    palette = (
        px.colors.qualitative.Set2
        + px.colors.qualitative.Safe
    )

    return {
        model: palette[i % len(palette)]
        for i, model in enumerate(sorted(models))
    }


all_models = sorted(
    set(summary_df["model"].dropna())
    | set(scoring_df["model"].dropna())
    | set(capacity_df["model"].dropna())
)

COLOR_MAP = model_colors(all_models)



# ============================================================
# MODEL TEST / RFP EXTRACTOR
#
# This uses the SAME extraction architecture as the V5 benchmark:
# - 8000-char chunks
# - 150-char overlap
# - 2 chunk workers
# - 650 max output tokens
# - invalid JSON recovery
# - candidate collection
# - deterministic Python field scoring/selection
#
# Only Qwen is exposed in the Streamlit "Test Model" section.
# ============================================================

from concurrent.futures import ThreadPoolExecutor, as_completed
import time


MODEL_BASE_URL = "https://t16.aidc.nadir.sh/v1"
MODEL_ID = "cyankiwi/Qwen3.5-4B-AWQ-4bit"

CHUNK_SIZE_CHARS = 8000
OVERLAP_CHARS = 150
CHUNK_WORKERS = 2
MAX_OUTPUT_TOKENS = 650
REQUEST_TIMEOUT_SECONDS = 180

RECOVERY_OVERLAP_CHARS = 100
MAX_CANDIDATES_PER_FIELD = 4
SOURCE_EXCERPT_CHARS = 900


EXTRACT_FIELDS = [
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


FIELD_LIST = "\n".join(
    f"- {field}"
    for field in EXTRACT_FIELDS
)

EXTRACT_PROMPT = f"""Extract RFP facts from the document chunk below.

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


def extract_pdf_text(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()

    doc = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    pages = []

    for page in doc:
        pages.append(page.get_text())

    doc.close()

    return "\n".join(pages)


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


def split_uploaded_rfp_text(text):
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
                end = search_start + cut
            else:
                end = hard_end

            if end <= start + 1000:
                end = hard_end

        chunk = text[start:end].strip()

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

    midpoint = len(chunk) // 2
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

    local_mid = midpoint - left_bound

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
        cut = left_bound + cut
    else:
        cut = midpoint

    first = chunk[:cut].strip()

    second_start = max(
        0,
        cut - RECOVERY_OVERLAP_CHARS,
    )

    second = chunk[
        second_start:
    ].strip()

    return [
        part
        for part in (first, second)
        if part
    ]


def parse_model_json(text):
    if not text:
        return None, False

    cleaned = text.strip()

    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    cleaned = (
        cleaned
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    try:
        obj = json.loads(cleaned)

        if isinstance(obj, dict):
            return obj, True

    except Exception:
        pass

    decoder = json.JSONDecoder()

    for i, char in enumerate(cleaned):
        if char != "{":
            continue

        try:
            obj, _ = decoder.raw_decode(
                cleaned[i:]
            )

            if isinstance(obj, dict):
                return obj, True

        except Exception:
            pass

    return None, False


def raw_json_valid(text):
    try:
        return isinstance(
            json.loads(
                (text or "").strip()
            ),
            dict,
        )
    except Exception:
        return False


def make_model_client():
    return OpenAI(
        base_url=MODEL_BASE_URL,
        api_key="none",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def chat_request(content, max_tokens):
    client = make_model_client()

    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        temperature=0,
        max_tokens=max_tokens,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    choice = response.choices[0]

    raw = (
        choice.message.content
        or ""
    )

    usage = response.usage

    return {
        "time": elapsed,
        "raw": raw,
        "finish_reason": getattr(
            choice,
            "finish_reason",
            None,
        ),
        "prompt_tokens": (
            usage.prompt_tokens
            if usage
            else 0
        ),
        "completion_tokens": (
            usage.completion_tokens
            if usage
            else 0
        ),
        "total_tokens": (
            usage.total_tokens
            if usage
            else 0
        ),
    }


def run_extraction_once(chunk):
    result = chat_request(
        EXTRACT_PROMPT + chunk,
        MAX_OUTPUT_TOKENS,
    )

    parsed, cleaned_ok = parse_model_json(
        result["raw"]
    )

    return {
        **result,
        "raw_ok": raw_json_valid(
            result["raw"]
        ),
        "cleaned_ok": cleaned_ok,
        "parsed": parsed,
    }


def run_chunk_with_recovery(
    chunk,
    chunk_number,
):
    attempts = []
    parsed_parts = []

    first = run_extraction_once(
        chunk
    )

    attempts.append(first)

    if first["cleaned_ok"]:
        parsed_parts.append(
            {
                "parsed": first["parsed"],
                "source_text": chunk,
                "source_label": (
                    f"chunk {chunk_number}"
                ),
            }
        )

        return {
            "recovered": False,
            "attempts": attempts,
            "parsed_parts": parsed_parts,
            "success": True,
        }

    recovery_parts = split_failed_chunk(
        chunk
    )

    if (
        len(recovery_parts) == 1
        and recovery_parts[0] == chunk
    ):
        return {
            "recovered": False,
            "attempts": attempts,
            "parsed_parts": [],
            "success": False,
        }

    all_recovery_ok = True

    for part_num, part in enumerate(
        recovery_parts,
        1,
    ):
        recovered = run_extraction_once(
            part
        )

        attempts.append(
            recovered
        )

        if recovered["cleaned_ok"]:
            parsed_parts.append(
                {
                    "parsed": recovered["parsed"],
                    "source_text": part,
                    "source_label": (
                        f"chunk "
                        f"{chunk_number}."
                        f"{part_num}"
                    ),
                }
            )
        else:
            all_recovery_ok = False

    return {
        "recovered": bool(
            parsed_parts
        ),
        "attempts": attempts,
        "parsed_parts": parsed_parts,
        "success": (
            bool(parsed_parts)
            and all_recovery_ok
        ),
    }


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
        source_text or ""
    )

    if len(source_text) <= SOURCE_EXCERPT_CHARS:
        return source_text.strip()

    lower = source_text.lower()

    words = keyword_tokens(
        value
    )[:10]

    positions = []

    for word in words:
        pos = lower.find(word)

        if pos >= 0:
            positions.append(pos)

    if positions:
        center = min(
            positions
        )
    else:
        field_words = keyword_tokens(
            field
        )[:5]

        for word in field_words:
            pos = lower.find(word)

            if pos >= 0:
                positions.append(pos)

        center = (
            min(positions)
            if positions
            else len(source_text) // 2
        )

    half = (
        SOURCE_EXCERPT_CHARS // 2
    )

    start = max(
        0,
        center - half,
    )

    end = min(
        len(source_text),
        start + SOURCE_EXCERPT_CHARS,
    )

    start = max(
        0,
        end - SOURCE_EXCERPT_CHARS,
    )

    return (
        source_text[start:end]
        .strip()
    )


def new_candidate_map():
    return {
        field: []
        for field in EXTRACT_FIELDS
    }


def add_candidate(
    candidates,
    field,
    value,
    source_label,
    source_text,
):
    if field not in EXTRACT_FIELDS:
        return

    value = clean_value(value)

    if value is None:
        return

    normalized = normalize_text(
        value
    )

    if not normalized:
        return

    existing = candidates[field]

    for item in existing:
        if (
            normalize_text(
                item["value"]
            )
            == normalized
        ):
            return

    if (
        len(existing)
        >= MAX_CANDIDATES_PER_FIELD
    ):
        return

    excerpt = make_source_excerpt(
        source_text,
        field,
        value,
    )

    existing.append(
        {
            "source": source_label,
            "value": value,
            "excerpt": excerpt,
        }
    )


def add_parsed_candidates(
    candidates,
    parsed,
    source_label,
    source_text,
):
    if not isinstance(
        parsed,
        dict,
    ):
        return

    for field, value in parsed.items():
        add_candidate(
            candidates,
            field,
            value,
            source_label,
            source_text,
        )


def contains_any(text, terms):
    text = text.lower()

    return any(
        term in text
        for term in terms
    )


def count_matches(text, terms):
    text = text.lower()

    return sum(
        1
        for term in terms
        if term in text
    )


def looks_like_heading_only(value):
    v = normalize_text(value)

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

    if len(v) <= 90 and contains_any(
        v,
        heading_terms,
    ):
        has_detail = bool(
            re.search(
                r"\b\d+(?:\.\d+)?\s*%|\$\s*\d|\bminimum\b|\brequired\b|\bmust\b",
                v,
            )
        )

        if not has_detail:
            return True

    return False


def normalize_reference_count(value):
    text = normalize_text(value)

    digit_match = re.search(
        r"\b(\d+)\b",
        text,
    )

    if digit_match:
        return digit_match.group(1)

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
    v = normalize_text(value)
    e = normalize_text(excerpt)
    combined = f"{v} {e}"

    if not v:
        return None

    score = 1.0

    if len(v) < 2:
        return None

    if field == "RFP Name":
        if contains_any(
            v,
            (
                "rfp",
                "request for proposal",
                "erp",
                "lms",
                "software",
                "system",
            ),
        ):
            score += 2

    elif field == "Organization":
        if len(v) <= 150:
            score += 1

    elif field == "Issue Date":
        if contains_any(
            combined,
            (
                "issue date",
                "issued",
                "publication date",
                "published",
            ),
        ):
            score += 3

        if contains_any(
            combined,
            (
                "submission deadline",
                "closing date",
                "questions deadline",
            ),
        ):
            score -= 2

    elif field == "Start Date":
        if contains_any(
            combined,
            (
                "contract start",
                "commence",
                "commencement",
                "effective date",
                "effective from",
            ),
        ):
            score += 4

        if contains_any(
            combined,
            (
                "phase 1",
                "phase 2",
                "phase 3",
                "milestone",
                "expected completion",
                "completion date",
                "vendor selection",
                "submission deadline",
                "closing date",
                "questions deadline",
            ),
        ):
            score -= 4

        if score <= -1:
            return None

    elif field == "End Date":
        if contains_any(
            combined,
            (
                "contract end",
                "remain in effect until",
                "effective until",
                "term shall",
                "agreement until",
            ),
        ):
            score += 4

        if contains_any(
            combined,
            (
                "phase 1",
                "phase 2",
                "phase 3",
                "milestone",
                "expected completion",
                "vendor selection",
                "submission deadline",
                "closing date",
            ),
        ):
            score -= 4

        if score <= -1:
            return None

    elif field == "Submission Deadline (date & time)":
        if contains_any(
            combined,
            (
                "submission deadline",
                "closing date",
                "closing time",
                "proposals due",
                "proposal deadline",
                "submission date",
            ),
        ):
            score += 4

        if contains_any(
            combined,
            (
                "questions",
                "inquiries",
            ),
        ):
            score -= 2

    elif field == "Deadline for Questions / Inquiries":
        if contains_any(
            combined,
            (
                "questions",
                "inquiries",
                "deadline for questions",
                "question deadline",
            ),
        ):
            score += 4

    elif field == "Budget":
        if contains_any(
            combined,
            (
                "budget",
                "estimated budget",
                "project budget",
                "contract budget",
                "not to exceed",
            ),
        ):
            score += 3

        if contains_any(
            combined,
            (
                "pricing schedule",
                "unit price",
                "total bid",
                "rates must",
                "pricing matrix",
            ),
        ) and "budget" not in combined:
            return None

    elif field == "RFP Contact (name and/or email)":
        if re.search(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            value,
        ):
            score += 4

        if contains_any(
            combined,
            (
                "contact",
                "questions",
                "inquiries",
                "procurement",
            ),
        ):
            score += 2

    elif field == "Submission Method / Portal":
        if contains_any(
            combined,
            (
                "submit",
                "submission",
                "electronically",
                "email",
                "portal",
                "bidsandtenders",
                "bonfire",
                "bcbid",
                "upload",
                "pdf",
            ),
        ):
            score += 3

    elif field == "Contract Term / Duration (including renewal options)":
        if contains_any(
            combined,
            (
                "contract term",
                "agreement",
                "renew",
                "renewal",
                "years",
                "months",
                "remain in effect",
                "effective from",
            ),
        ):
            score += 3

    elif field == "Scope of Deliverables / Services Requested":
        if looks_like_heading_only(value):
            return None

        if contains_any(
            combined,
            (
                "scope",
                "deliverables",
                "services",
                "provide",
                "conduct",
                "develop",
                "implementation",
                "assessment",
                "review",
            ),
        ):
            score += 2

    elif field == "Mandatory Submission Requirements":
        if looks_like_heading_only(value):
            return None

        good_terms = (
            "submit",
            "submission",
            "proposal",
            "form",
            "signed",
            "include",
            "pricing schedule",
            "mandatory evaluation table",
            "pdf",
            "authorized representative",
        )

        bad_terms = (
            "years of experience",
            "technical requirement",
            "security control",
            "hosting",
        )

        score += (
            2
            * count_matches(
                combined,
                good_terms,
            )
        )

        score -= (
            2
            * count_matches(
                combined,
                bad_terms,
            )
        )

    elif field == "Mandatory Technical Requirements":
        if looks_like_heading_only(value):
            return None

        technical_terms = (
            "technical",
            "integration",
            "integrate",
            "system",
            "security",
            "encryption",
            "authentication",
            "api",
            "cloud",
            "hosting",
            "connectivity",
            "architecture",
            "compatibility",
            "soc2",
            "wcag",
            "data residency",
            "qa",
            "iso 9001",
            "iso 14001",
            "testing",
            "validation",
        )

        wrong_terms = (
            "unit price",
            "total bid",
            "rates must",
            "canadian funds",
            "pricing schedule",
            "minimum of five years",
            "years of experience",
            "client references",
            "authorized representative",
        )

        score += (
            2
            * count_matches(
                combined,
                technical_terms,
            )
        )

        score -= (
            3
            * count_matches(
                combined,
                wrong_terms,
            )
        )

        if score <= 0:
            return None

    elif field == "Evaluation Criteria & Weighting (points breakdown by category)":
        if looks_like_heading_only(value):
            return None

        has_percent = bool(
            re.search(
                r"\b\d+(?:\.\d+)?\s*%",
                value,
            )
        )

        has_points = bool(
            re.search(
                r"\b\d+(?:\.\d+)?\s*(?:points?|pts?)\b",
                value.lower(),
            )
        )

        has_multiple_numbers = (
            len(
                re.findall(
                    r"\b\d+(?:\.\d+)?\b",
                    value,
                )
            )
            >= 2
        )

        if not (
            has_percent
            or has_points
            or has_multiple_numbers
        ):
            return None

        if contains_any(
            combined,
            (
                "criteria",
                "weight",
                "points",
                "pricing",
                "experience",
                "qualifications",
            ),
        ):
            score += 3

    elif field == "Minimum Score Threshold to Advance (where a pass/fail cutoff is specified)":
        if not re.search(
            r"\b\d+(?:\.\d+)?\s*%",
            value,
        ) and not re.search(
            r"\b\d+(?:\.\d+)?\s*(?:points?|pts?)\b",
            value.lower(),
        ):
            return None

        if contains_any(
            combined,
            (
                "threshold",
                "minimum",
                "advance",
                "pass",
                "proceed",
            ),
        ):
            score += 3

    elif field == "Pricing Structure / Cost Submission Requirements":
        if looks_like_heading_only(value):
            return None

        pricing_terms = (
            "price",
            "pricing",
            "cost",
            "rate",
            "fee",
            "fees",
            "unit price",
            "total bid",
            "currency",
            "canadian",
            "tax",
            "pricing schedule",
            "pricing matrix",
            "all-inclusive",
        )

        score += (
            2
            * count_matches(
                combined,
                pricing_terms,
            )
        )

        if count_matches(
            combined,
            pricing_terms,
        ) == 0:
            return None

    elif field == "Minimum Insurance Coverage Requirements (type and dollar amount)":
        if "insurance" not in combined:
            return None

        score += 3

        if re.search(
            r"\$\s*[\d,]+",
            value,
        ):
            score += 2

    elif field == "Required Vendor Experience / Qualifications (e.g., years of experience)":
        experience_terms = (
            "experience",
            "years",
            "qualification",
            "qualifications",
            "certification",
            "certified",
            "project manager",
            "team member",
            "similar projects",
        )

        score += (
            2
            * count_matches(
                combined,
                experience_terms,
            )
        )

        if count_matches(
            combined,
            experience_terms,
        ) == 0:
            return None

    elif field == "Number of References Required":
        count = normalize_reference_count(
            value
        )

        if count is None:
            return None

        if "reference" not in combined:
            return None

        score += 4

    elif field == "Data Security / Privacy Compliance Requirements":
        terms = (
            "security",
            "privacy",
            "encryption",
            "confidential",
            "cyber",
            "soc2",
            "data protection",
            "access control",
            "compliance",
        )

        score += (
            2
            * count_matches(
                combined,
                terms,
            )
        )

        if count_matches(
            combined,
            terms,
        ) == 0:
            return None

    elif field == "Data Hosting / Residency Requirements (where specified)":
        terms = (
            "hosting",
            "hosted",
            "residency",
            "residence",
            "data centre",
            "data center",
            "stored in canada",
            "hosted in canada",
            "jurisdiction",
        )

        score += (
            2
            * count_matches(
                combined,
                terms,
            )
        )

        if count_matches(
            combined,
            terms,
        ) == 0:
            return None

    elif field == "Vendor Demonstration Requirement":
        if contains_any(
            combined,
            (
                "demonstration",
                "demo",
                "presentation",
                "shortlisted vendors",
                "short-listed",
            ),
        ):
            score += 4
        elif v.lower() in {
            "yes",
            "required",
        }:
            score += 1
        else:
            return None

    elif field == "Summary":
        if len(v) < 30:
            return None

        score += 1

    if len(v) <= 500:
        score += 0.5

    return score


def empty_extraction_result():
    return {
        field: "Not Specified"
        for field in EXTRACT_FIELDS
    }


def select_final_output(
    candidates,
):
    final = empty_extraction_result()

    for field in EXTRACT_FIELDS:
        items = candidates[field]

        if not items:
            continue

        scored = []

        for index, item in enumerate(
            items
        ):
            score = field_candidate_score(
                field,
                item["value"],
                item["excerpt"],
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

        chosen = scored[0][2]
        value = chosen["value"]

        if field == "Number of References Required":
            normalized_count = (
                normalize_reference_count(
                    value
                )
            )

            if normalized_count:
                value = normalized_count

        final[field] = value

    return final


def run_uploaded_document(text):
    chunks = split_uploaded_rfp_text(
        text
    )

    candidates = new_candidate_map()

    if not chunks:
        return {
            "output": empty_extraction_result(),
            "chunks_total": 0,
            "chunks_successful": 0,
            "time_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "raw_json_valid_rate": 0.0,
            "json_valid_after_cleanup_rate": 0.0,
        }

    progress_bar = st.progress(0)
    status_text = st.empty()

    document_start = (
        time.perf_counter()
    )

    chunk_results = {}

    workers = min(
        CHUNK_WORKERS,
        max(
            len(chunks),
            1,
        ),
    )

    completed = 0

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        futures = {
            executor.submit(
                run_chunk_with_recovery,
                chunk,
                i,
            ): i
            for i, chunk in enumerate(
                chunks,
                1,
            )
        }

        for future in as_completed(
            futures
        ):
            i = futures[future]

            chunk_results[i] = (
                future.result()
            )

            completed += 1

            status_text.write(
                f"Processing chunks: "
                f"{completed} of {len(chunks)} completed..."
            )

            progress_bar.progress(
                completed / len(chunks)
            )

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    raw_ok = 0
    cleaned_ok = 0
    successful_original_chunks = 0

    for i in sorted(
        chunk_results
    ):
        result = chunk_results[i]

        if result["success"]:
            successful_original_chunks += 1

        for attempt in result[
            "attempts"
        ]:
            prompt_tokens += (
                attempt[
                    "prompt_tokens"
                ]
            )

            completion_tokens += (
                attempt[
                    "completion_tokens"
                ]
            )

            total_tokens += (
                attempt[
                    "total_tokens"
                ]
            )

        original = (
            result[
                "attempts"
            ][0]
        )

        raw_ok += int(
            original[
                "raw_ok"
            ]
        )

        cleaned_ok += int(
            original[
                "cleaned_ok"
            ]
        )

        for parsed_part in result[
            "parsed_parts"
        ]:
            add_parsed_candidates(
                candidates,
                parsed_part[
                    "parsed"
                ],
                parsed_part[
                    "source_label"
                ],
                parsed_part[
                    "source_text"
                ],
            )

    final = select_final_output(
        candidates
    )

    wall_time = (
        time.perf_counter()
        - document_start
    )

    total_original_chunks = (
        len(chunks)
    )

    status_text.empty()
    progress_bar.empty()

    return {
        "output": final,
        "chunks_total": (
            total_original_chunks
        ),
        "chunks_successful": (
            successful_original_chunks
        ),
        "time_seconds": round(
            wall_time,
            2,
        ),
        "prompt_tokens": (
            prompt_tokens
        ),
        "completion_tokens": (
            completion_tokens
        ),
        "total_tokens": (
            total_tokens
        ),
        "raw_json_valid_rate": (
            round(
                raw_ok
                / total_original_chunks
                * 100,
                2,
            )
            if total_original_chunks
            else 0
        ),
        "json_valid_after_cleanup_rate": (
            round(
                cleaned_ok
                / total_original_chunks
                * 100,
                2,
            )
            if total_original_chunks
            else 0
        ),
    }



# ============================================================
# TITLE
# ============================================================

st.title("RFP AI Extractor & Model Evaluation")

st.caption(
    "Test the deployed Qwen model on an RFP PDF, then review the "
    "five-document benchmark and full-RFP capacity results below."
)


# ============================================================
# TEST THE DEPLOYED MODEL
# ============================================================

anchor("extractor")

st.markdown("## Test the Deployed Model")

st.markdown(
    '<div class="section-subtitle">'
    'Upload an RFP PDF and extract the 24 benchmark fields using the deployed Qwen model.'
    '</div>',
    unsafe_allow_html=True,
)

if "display_result" not in st.session_state:
    st.session_state.display_result = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = None

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if "extraction_metrics" not in st.session_state:
    st.session_state.extraction_metrics = None


uploaded_file = st.file_uploader(
    "Upload RFP PDF",
    type=["pdf"],
    key="rfp_pdf_uploader",
)

st.caption("PDF files only.")

if uploaded_file is not None:
    if (
        st.session_state.last_uploaded_file is not None
        and st.session_state.last_uploaded_file
        != uploaded_file.name
    ):
        st.session_state.display_result = None
        st.session_state.chunk_count = None
        st.session_state.extraction_metrics = None

    st.session_state.last_uploaded_file = (
        uploaded_file.name
    )

    upload_col1, upload_col2 = st.columns([2, 1])

    with upload_col1:
        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

    with upload_col2:
        st.metric(
            "File size",
            f"{uploaded_file.size / 1024:.1f} KB",
        )

    if st.button(
        "Analyze RFP",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Extracting PDF text..."
            ):
                uploaded_text = extract_pdf_text(
                    uploaded_file
                )

            if not uploaded_text.strip():
                st.error(
                    "No readable text was found in the PDF."
                )
            else:
                with st.spinner(
                    "Analyzing RFP with Qwen..."
                ):
                    analysis = run_uploaded_document(
                        uploaded_text
                    )

                st.session_state.display_result = (
                    analysis["output"]
                )

                st.session_state.chunk_count = (
                    analysis["chunks_total"]
                )

                st.session_state.extraction_metrics = {
                    "latency_seconds": analysis["time_seconds"],
                    "chunks_successful": analysis["chunks_successful"],
                    "prompt_tokens": analysis["prompt_tokens"],
                    "completion_tokens": analysis["completion_tokens"],
                    "total_tokens": analysis["total_tokens"],
                    "raw_json_valid_rate": analysis["raw_json_valid_rate"],
                    "json_valid_after_cleanup_rate": analysis[
                        "json_valid_after_cleanup_rate"
                    ],
                }

        except Exception as exc:
            st.error(
                f"Model request failed: {exc}"
            )


if st.session_state.display_result is not None:
    st.success("Analysis complete")

    st.caption(
        f"Document processed in "
        f"{st.session_state.chunk_count} chunks."
    )

    if st.session_state.extraction_metrics is not None:
        metrics = st.session_state.extraction_metrics

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Full document latency",
            f"{metrics['latency_seconds']:.2f} s",
        )

        m2.metric(
            "Successful chunks",
            f"{metrics['chunks_successful']}/{st.session_state.chunk_count}",
        )

        m3.metric(
            "Total tokens",
            f"{metrics['total_tokens']:,}",
        )

        m4.metric(
            "JSON valid after cleanup",
            f"{metrics['json_valid_after_cleanup_rate']:.1f}%",
        )

    st.markdown("#### Extracted Information")

    table_tab, json_tab = st.tabs(
        ["Table", "JSON"]
    )

    with table_tab:
        extraction_table = pd.DataFrame(
            list(
                st.session_state
                .display_result
                .items()
            ),
            columns=[
                "Field",
                "Extracted Value",
            ],
        )

        st.table(
            extraction_table
        )

    with json_tab:
        st.json(
            st.session_state.display_result
        )

    st.download_button(
        label="Download JSON",
        data=json.dumps(
            st.session_state.display_result,
            indent=2,
            ensure_ascii=False,
        ),
        file_name="rfp_result.json",
        mime="application/json",
        use_container_width=True,
    )


st.divider()

st.markdown(
    """
    <div class="small-note">
    Benchmark quality includes Granite, Qwen, and OpenAI across five RFPs.
    Capacity testing covers the deployed local models only: Granite and Qwen.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 1. OVERVIEW
# ============================================================

section_header(
    "overview",
    "1. Overview",
    "The main benchmark and production-capacity results in one place.",
)

best_accuracy = summary_df.loc[
    summary_df["field_accuracy_pct"].idxmax()
]

lowest_latency = summary_df.loc[
    summary_df["avg_latency_seconds"].idxmin()
]

lowest_hallucination = summary_df.loc[
    summary_df["hallucination_rate_pct"].idxmin()
]

peak_capacity = capacity_df.loc[
    capacity_df["throughput_requests_per_min"].idxmax()
]

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "RFPs benchmarked",
    int(summary_df["rfps_evaluated"].max()),
)

k2.metric(
    "Highest avg field accuracy",
    format_percent(best_accuracy["field_accuracy_pct"]),
    best_accuracy["model"],
)

k3.metric(
    "Lowest avg benchmark latency",
    format_seconds(lowest_latency["avg_latency_seconds"]),
    lowest_latency["model"],
)

k4.metric(
    "Highest observed throughput",
    f"{peak_capacity['throughput_requests_per_min']:.2f} RFP/min",
    f"{peak_capacity['model']} @ C={int(peak_capacity['concurrency'])}",
)


quality_snapshot = summary_df[
    [
        "model",
        "field_accuracy_pct",
        "avg_field_similarity_pct",
        "completeness_pct",
    ]
].melt(
    id_vars="model",
    var_name="metric",
    value_name="percent",
)

quality_snapshot["metric"] = quality_snapshot["metric"].map(
    {
        "field_accuracy_pct": "Field accuracy",
        "avg_field_similarity_pct": "Field similarity",
        "completeness_pct": "Completeness",
    }
)

fig = px.bar(
    quality_snapshot,
    x="model",
    y="percent",
    color="metric",
    barmode="group",
    labels={
        "model": "",
        "percent": "Percent",
        "metric": "",
    },
    title="Quality snapshot",
)

fig.update_layout(
    height=410,
    legend_title_text="",
    yaxis_range=[0, 100],
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config=PLOTLY_CONFIG,
)

st.markdown("#### Model Summary")

overview_table = summary_df[
    [
        "model",
        "field_accuracy_pct",
        "completeness_pct",
        "hallucination_rate_pct",
        "wrong_value_rate_pct",
        "avg_latency_seconds",
        "avg_total_tokens",
    ]
].copy()

overview_table.columns = [
    "Model",
    "Accuracy %",
    "Completeness %",
    "Hallucination %",
    "Wrong value %",
    "Avg latency (s)",
    "Avg tokens",
]

overview_display = overview_table.copy()
overview_display["Accuracy %"] = overview_display["Accuracy %"].map(lambda x: f"{x:.2f}")
overview_display["Completeness %"] = overview_display["Completeness %"].map(lambda x: f"{x:.2f}")
overview_display["Hallucination %"] = overview_display["Hallucination %"].map(lambda x: f"{x:.2f}")
overview_display["Wrong value %"] = overview_display["Wrong value %"].map(lambda x: f"{x:.2f}")
overview_display["Avg latency (s)"] = overview_display["Avg latency (s)"].map(lambda x: f"{x:.2f}")
overview_display["Avg tokens"] = overview_display["Avg tokens"].map(lambda x: f"{x:,.0f}")

st.dataframe(
    overview_display,
    use_container_width=True,
    hide_index=True,
)

st.markdown(
    f"""
    <div class="info-card">
    <b>Benchmark note</b><br>
    Lowest average hallucination rate:
    <b>{lowest_hallucination['model']}</b>
    at <b>{lowest_hallucination['hallucination_rate_pct']:.2f}%</b>.
    </div>
    """,
    unsafe_allow_html=True,
)


with st.expander("Metric guide"):
    st.markdown(
        """
        - **Field accuracy** — percentage of the 24 target fields scored as correct.
        - **Field similarity** — average similarity between predicted and ground-truth values.
        - **Completeness** — percentage of expected non-missing fields that were populated.
        - **Hallucination rate** — cases where ground truth is missing but the model produced a value.
        - **Wrong value rate** — expected fields that were populated with an incorrect value.
        - **P95 latency** — latency below which roughly 95% of measured full-RFP jobs completed.
        """
    )


st.divider()


# ============================================================
# EXTRACTION RISK + FIVE-RFP SUMMARY
# ============================================================

st.markdown("#### Extraction risk indicators")

reliability = summary_df[
    [
        "model",
        "hallucination_rate_pct",
        "wrong_value_rate_pct",
    ]
].melt(
    id_vars="model",
    var_name="metric",
    value_name="percent",
)

reliability["metric"] = reliability["metric"].map(
    {
        "hallucination_rate_pct": "Hallucination rate",
        "wrong_value_rate_pct": "Wrong value rate",
    }
)

fig = px.bar(
    reliability,
    x="model",
    y="percent",
    color="metric",
    barmode="group",
    labels={
        "model": "",
        "percent": "Percent",
        "metric": "",
    },
    title="Extraction risk indicators",
)

fig.update_layout(
    height=390,
    legend_title_text="",
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config=PLOTLY_CONFIG,
)


st.divider()


# ============================================================
# 2. CAPACITY & SCALABILITY
# ============================================================

section_header(
    "capacity",
    "2. Capacity & Scalability",
    "Full-RFP production load testing: each request represents one complete RFP extraction.",
)

capacity_models = sorted(
    capacity_df["model"].unique()
)

selected_capacity_models = st.multiselect(
    "Capacity models",
    options=capacity_models,
    default=capacity_models,
    key="capacity_models",
)

cap = capacity_df[
    capacity_df["model"].isin(selected_capacity_models)
].copy()

if cap.empty:
    st.warning("Select at least one capacity model.")
else:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Max tested concurrency",
        int(cap["concurrency"].max()),
    )

    overall_success_rate = (
        cap["requests_successful"].sum()
        / cap["requests_total"].sum()
        * 100
    )

    c2.metric(
        "Overall success rate",
        f"{overall_success_rate:.1f}%",
    )

    c3.metric(
        "Failed full-RFP jobs",
        int(cap["requests_failed"].sum()),
    )

    c4.metric(
        "Peak observed throughput",
        f"{cap['throughput_requests_per_min'].max():.2f} RFP/min",
    )

    throughput_fig = px.line(
        cap,
        x="concurrency",
        y="throughput_requests_per_min",
        color="model",
        markers=True,
        color_discrete_map=COLOR_MAP,
        labels={
            "concurrency": "Concurrent full-RFP jobs",
            "throughput_requests_per_min": "Throughput (RFP/min)",
            "model": "",
        },
        title="Throughput vs concurrency",
    )

    throughput_fig.update_layout(
        height=440,
        legend_title_text="",
    )

    st.plotly_chart(
        throughput_fig,
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    latency_fig = px.line(
        cap,
        x="concurrency",
        y="p95_latency_s",
        color="model",
        markers=True,
        color_discrete_map=COLOR_MAP,
        labels={
            "concurrency": "Concurrent full-RFP jobs",
            "p95_latency_s": "P95 latency (s)",
            "model": "",
        },
        title="P95 latency vs concurrency",
    )

    latency_fig.update_layout(
        height=440,
        legend_title_text="",
    )

    st.plotly_chart(
        latency_fig,
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

    st.markdown("#### Observed capacity points")

    capacity_cards = st.columns(
        len(selected_capacity_models)
        if selected_capacity_models
        else 1
    )

    for i, model in enumerate(
        selected_capacity_models
    ):
        model_cap = cap[
            cap["model"] == model
        ].copy()

        peak_row = model_cap.loc[
            model_cap[
                "throughput_requests_per_min"
            ].idxmax()
        ]

        max_row = model_cap.loc[
            model_cap["concurrency"].idxmax()
        ]

        with capacity_cards[i]:
            st.markdown(
                f"""
                <div class="info-card">
                    <b>{model}</b><br><br>
                    Peak observed throughput:
                    <b>{peak_row['throughput_requests_per_min']:.2f} RFP/min</b>
                    at concurrency
                    <b>{int(peak_row['concurrency'])}</b>.<br><br>
                    Maximum tested concurrency:
                    <b>{int(max_row['concurrency'])}</b><br>
                    P95 at maximum tested load:
                    <b>{max_row['p95_latency_s']:.2f} s</b><br>
                    Success rate at maximum tested load:
                    <b>{max_row['success_rate_pct']:.1f}%</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "Higher concurrency does not automatically mean better capacity. "
        "A throughput plateau combined with rising P95 latency indicates "
        "that extra concurrent work is mainly increasing queueing/wait time."
    )

    st.info(
        "Granite showed strong saturation before the highest tested loads: "
        "as concurrency increased, latency became very large while throughput "
        "stopped improving meaningfully. This is best described as practical "
        "throttling / saturation rather than a hard request failure."
    )

    capacity_table = cap[
        [
            "model",
            "concurrency",
            "success_rate_pct",
            "avg_latency_s",
            "p95_latency_s",
            "throughput_requests_per_min",
        ]
    ].copy()

    capacity_table.columns = [
        "Model",
        "Concurrency",
        "Success %",
        "Avg latency (s)",
        "P95 latency (s)",
        "Throughput (RFP/min)",
    ]

    with st.expander("View capacity data"):
        capacity_display = capacity_table.copy()
        capacity_display["Success %"] = capacity_display["Success %"].map(lambda x: f"{x:.1f}")
        capacity_display["Avg latency (s)"] = capacity_display["Avg latency (s)"].map(lambda x: f"{x:.2f}")
        capacity_display["P95 latency (s)"] = capacity_display["P95 latency (s)"].map(lambda x: f"{x:.2f}")
        capacity_display["Throughput (RFP/min)"] = capacity_display["Throughput (RFP/min)"].map(lambda x: f"{x:.2f}")

        st.dataframe(
            capacity_display,
            use_container_width=True,
            hide_index=True,
        )


st.divider()


# ============================================================
# 3. DETAILED RFP ANALYSIS
# ============================================================

section_header(
    "rfp-detail",
    "3. Detailed RFP Analysis",
    "Drill-down comparison — choose one RFP and compare the models.",
)

rfp_options = rfp_order
model_options = sorted(
    scoring_df["model"].unique()
)

f1, f2 = st.columns(2)

selected_rfp = f1.selectbox(
    "Select RFP",
    options=rfp_options,
    key="dashboard_selected_rfp",
)

selected_models = f2.multiselect(
    "Select models",
    options=model_options,
    default=model_options,
    key="dashboard_selected_models",
)

selected_detail = scoring_df[
    (scoring_df["rfp"] == selected_rfp)
    & (
        scoring_df["model"]
        .isin(selected_models)
    )
].copy()

if selected_detail.empty:
    st.info("Select at least one model.")
else:
    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Best field accuracy",
        f"{selected_detail['field_accuracy_pct'].max():.2f}%",
    )

    d2.metric(
        "Best similarity",
        f"{selected_detail['avg_field_similarity_pct'].max():.2f}%",
    )

    d3.metric(
        "Lowest latency",
        f"{selected_detail['latency_seconds'].min():.2f} s",
    )

    d4.metric(
        "Lowest hallucination",
        f"{selected_detail['hallucination_rate_pct'].min():.2f}%",
    )

    left, right = st.columns(2)

    with left:
        detail_quality = selected_detail[
            [
                "model",
                "field_accuracy_pct",
                "avg_field_similarity_pct",
                "completeness_pct",
            ]
        ].melt(
            id_vars="model",
            var_name="metric",
            value_name="percent",
        )

        detail_quality["metric"] = detail_quality["metric"].map(
            {
                "field_accuracy_pct": "Field accuracy",
                "avg_field_similarity_pct": "Similarity",
                "completeness_pct": "Completeness",
            }
        )

        fig = px.bar(
            detail_quality,
            x="model",
            y="percent",
            color="metric",
            barmode="group",
            labels={
                "model": "Model",
                "percent": "Percent",
                "metric": "",
            },
            title=f"Quality comparison — {selected_rfp}",
        )

        fig.update_layout(
            height=420,
            legend_title_text="",
            yaxis_range=[0, 100],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    with right:
        fig = px.bar(
            selected_detail,
            x="model",
            y="latency_seconds",
            color="model",
            color_discrete_map=COLOR_MAP,
            labels={
                "model": "Model",
                "latency_seconds": "Latency (s)",
            },
            title=f"Latency — {selected_rfp}",
        )

        fig.update_layout(
            height=420,
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "All dashboard values are read directly from the supplied "
    "benchmark summary, scoring results, and full-RFP capacity CSV files."
)
