import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE.parent / "documents"
RESULTS_FILE = BASE / "benchmark_results.json"


# ============================================================
# CHOOSE WHICH DOCUMENT(S) TO RUN
#
# Run one file first, then change this list later.
# Existing benchmark_results.json content is preserved.
# ============================================================

DOCUMENTS_TO_RUN = [
    "rfp1.txt",
    "rfp2.txt",
    "rfp3.txt",
    "rfp4.txt",
    "rfp5.txt",
]


# ============================================================
# MODELS
#
# OpenAI intentionally excluded for now.
#
# Expected port-forward:
# kubectl port-forward -n team deployment/rfp-dual-serving \
#   9000:8000 9001:8001
# ============================================================

MODELS = {
    "Granite-8B-AWQ": {
        "provider": "local",
        "url": "http://localhost:9000/v1",
        "model": "drawais/Granite-4.1-8B-AWQ-INT4",
    },
    "Qwen-4B-AWQ": {
        "provider": "local",
        "url": "http://localhost:9001/v1",
        "model": "cyankiwi/Qwen3.5-4B-AWQ-4bit",
    },
    "OpenAI": {
        "provider": "openai",
        "model": "gpt-5.6-luna",
    },
}


# ============================================================
# BENCHMARK SETTINGS
#
# Based on V3:
# - large chunks
# - low overlap
# - 2 workers
# - invalid JSON recovery
#
# V5 addition:
# - NO extra LLM validation/consolidation calls
# - deterministic Python field checks + candidate scoring
# ============================================================

CHUNK_SIZE_CHARS = 8000
OVERLAP_CHARS = 150
CHUNK_WORKERS = 2

MAX_OUTPUT_TOKENS = 650
REQUEST_TIMEOUT_SECONDS = 180

RECOVERY_OVERLAP_CHARS = 100
MAX_CANDIDATES_PER_FIELD = 4
SOURCE_EXCERPT_CHARS = 900


# ============================================================
# OUTPUT FIELDS
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
# EXTRACTION PROMPT
#
# Unsupported fields MUST be omitted at chunk level.
# Python alone inserts "Not Specified" in the final output.
# ============================================================

FIELD_LIST = "\n".join(f"- {field}" for field in FIELDS)

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
# MISSING-VALUE DEFENSE
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

    if simplified in {"null", "none", "n/a", "na"}:
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

def split_text(text: str):
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

    for i, c in enumerate(cleaned):
        if c != "{":
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


# ============================================================
# MODEL REQUEST
# ============================================================

def make_client(model_name, info):
    """
    Create the correct client.

    OpenAI uses the official API and OPENAI_API_KEY.
    Granite/Qwen use the local OpenAI-compatible vLLM endpoints.
    """
    if model_name == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Run: export OPENAI_API_KEY='your-key'"
            )

        return OpenAI(
            api_key=api_key,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    return OpenAI(
        base_url=info["url"],
        api_key="none",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def chat_request(
    model_name,
    info,
    content,
    max_tokens,
):
    """
    Granite/Qwen:
      local OpenAI-compatible Chat Completions API.

    OpenAI:
      official Responses API using gpt-5.6-luna.

    Both are normalized to the same token/latency fields so the
    existing benchmark JSON and scoring script still work.
    """

    client = make_client(
        model_name,
        info,
    )

    start = time.perf_counter()

    # --------------------------------------------------------
    # OPENAI
    # --------------------------------------------------------
    if model_name == "OpenAI":
        response = client.responses.create(
            model=info["model"],
            input=content,
            max_output_tokens=max_tokens,
            reasoning={
                "effort": "none",
            },
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        raw = (
            response.output_text
            or ""
        )

        usage = getattr(
            response,
            "usage",
            None,
        )

        input_tokens = (
            getattr(
                usage,
                "input_tokens",
                0,
            )
            if usage
            else 0
        )

        output_tokens = (
            getattr(
                usage,
                "output_tokens",
                0,
            )
            if usage
            else 0
        )

        total_tokens = (
            getattr(
                usage,
                "total_tokens",
                input_tokens + output_tokens,
            )
            if usage
            else 0
        )

        return {
            "time": elapsed,
            "raw": raw,
            "finish_reason": getattr(
                response,
                "status",
                None,
            ),
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    # --------------------------------------------------------
    # LOCAL GRANITE / QWEN
    # --------------------------------------------------------
    kwargs = {
        "model": info["model"],
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    if "Qwen" in model_name:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }

    response = (
        client.chat.completions.create(
            **kwargs
        )
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


# ============================================================
# ONE EXTRACTION REQUEST
# ============================================================

def run_extraction_once(
    model_name,
    info,
    chunk,
):
    r = chat_request(
        model_name,
        info,
        PROMPT + chunk,
        MAX_OUTPUT_TOKENS,
    )

    parsed, cleaned_ok = parse_json(
        r["raw"]
    )

    return {
        **r,
        "raw_ok": raw_json_valid(
            r["raw"]
        ),
        "cleaned_ok": cleaned_ok,
        "parsed": parsed,
    }


# ============================================================
# INVALID/TRUNCATED JSON RECOVERY
# ============================================================

def run_chunk_with_recovery(
    model_name,
    info,
    chunk,
    chunk_number,
):
    attempts = []
    parsed_parts = []

    first = run_extraction_once(
        model_name,
        info,
        chunk,
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

    print(
        f"\n--- INVALID CHUNK "
        f"{chunk_number} RESPONSE ---"
    )

    print(
        f"finish_reason="
        f"{first['finish_reason']}"
    )

    print(first["raw"])

    print(
        "----------------------------------"
    )

    recovery_parts = (
        split_failed_chunk(
            chunk
        )
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

    print(
        f"  Recovering chunk "
        f"{chunk_number} as "
        f"{len(recovery_parts)} "
        f"smaller requests..."
    )

    all_recovery_ok = True

    for part_num, part in enumerate(
        recovery_parts,
        1,
    ):
        rr = run_extraction_once(
            model_name,
            info,
            part,
        )

        attempts.append(rr)

        print(
            f"    Recovery "
            f"{chunk_number}.{part_num}"
            f" | {rr['time']:.2f}s"
            f" | raw={rr['raw_ok']}"
            f" | cleaned={rr['cleaned_ok']}"
            f" | finish="
            f"{rr['finish_reason']}"
            f" | tokens="
            f"{rr['total_tokens']}"
        )

        if rr["cleaned_ok"]:
            parsed_parts.append(
                {
                    "parsed": rr["parsed"],
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


# ============================================================
# SOURCE EXCERPTS
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
        source_text or ""
    )

    if len(source_text) <= (
        SOURCE_EXCERPT_CHARS
    ):
        return source_text.strip()

    lower = source_text.lower()

    # Prefer words from the actual candidate value.
    words = keyword_tokens(
        value
    )[:10]

    positions = []

    for word in words:
        pos = lower.find(word)

        if pos >= 0:
            positions.append(pos)

    if positions:
        center = min(positions)
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


# ============================================================
# CANDIDATE COLLECTION
# ============================================================

def new_candidate_map():
    return {
        field: []
        for field in FIELDS
    }


def add_candidate(
    candidates,
    field,
    value,
    source_label,
    source_text,
):
    if field not in FIELDS:
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

    for field, value in (
        parsed.items()
    ):
        add_candidate(
            candidates,
            field,
            value,
            source_label,
            source_text,
        )


# ============================================================
# DETERMINISTIC FIELD RULES
#
# V5: No extra model calls.
#
# Each candidate gets a Python score based on:
# - field meaning
# - value content
# - nearby source excerpt
#
# A candidate can also be rejected completely.
# ============================================================

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

    # Very short section-heading style values.
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
        # If it contains actual numeric/requirement detail,
        # don't reject it automatically.
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
    """
    Convert values like:
      'minimum of three (3) references'
      '3 or more references'
    into:
      '3'
    if a clear count exists.
    """
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

    # --------------------------------------------------------
    # Generic penalties
    # --------------------------------------------------------

    if len(v) < 2:
        return None

    # --------------------------------------------------------
    # Exact field-specific rules
    # --------------------------------------------------------

    if field == "RFP Name":
        if contains_any(
            v,
            ("rfp", "request for proposal", "erp", "lms", "software", "system"),
        ):
            score += 2

    elif field == "Organization":
        if len(v) <= 150:
            score += 1

    elif field == "Issue Date":
        if contains_any(
            combined,
            ("issue date", "issued", "publication date", "published"),
        ):
            score += 3

        if contains_any(
            combined,
            ("submission deadline", "closing date", "questions deadline"),
        ):
            score -= 2

    elif field == "Start Date":
        if contains_any(
            combined,
            ("contract start", "commence", "commencement", "effective date", "effective from"),
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
            ("contract end", "remain in effect until", "effective until", "term shall", "agreement until"),
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
            ("questions", "inquiries"),
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

        # Reject pricing-instruction text masquerading as budget.
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
            ("contact", "questions", "inquiries", "procurement"),
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
            ("criteria", "weight", "points", "pricing", "experience", "qualifications"),
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
            ("threshold", "minimum", "advance", "pass", "proceed"),
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

    # Small preference for concise candidates.
    if len(v) <= 500:
        score += 0.5

    return score


# ============================================================
# FINAL PYTHON-ONLY CANDIDATE SELECTION
# ============================================================

def empty_result():
    return {
        field: "Not Specified"
        for field in FIELDS
    }


def select_final_output(
    candidates,
):
    final = empty_result()

    for field in FIELDS:
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

        # Normalize reference count to the exact count only.
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


# ============================================================
# RUN ONE FULL DOCUMENT
# ============================================================

def run_document(
    model_name,
    info,
    text,
):
    chunks = split_text(text)
    candidates = new_candidate_map()

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    raw_ok = 0
    cleaned_ok = 0
    successful_original_chunks = 0

    print(
        f"\nModel: {model_name}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

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

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {
            executor.submit(
                run_chunk_with_recovery,
                model_name,
                info,
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

            try:
                result = future.result()

                chunk_results[i] = result

                original = (
                    result[
                        "attempts"
                    ][0]
                )

                print(
                    f"  Chunk "
                    f"{i}/{len(chunks)}"
                    f" | "
                    f"{original['time']:.2f}s"
                    f" | raw="
                    f"{original['raw_ok']}"
                    f" | cleaned="
                    f"{original['cleaned_ok']}"
                    f" | finish="
                    f"{original['finish_reason']}"
                    f" | tokens="
                    f"{original['total_tokens']}"
                    + (
                        " | RECOVERED"
                        if result[
                            "recovered"
                        ]
                        else ""
                    )
                )

            except Exception as e:
                print(
                    f"  Chunk "
                    f"{i}/{len(chunks)}"
                    f" ERROR: {e}"
                )

    # Process chunks in original document order.
    for i in sorted(
        chunk_results
    ):
        result = chunk_results[i]

        if result["success"]:
            successful_original_chunks += 1

        # Count ALL actual model requests:
        # original attempts + recovery attempts.
        for attempt in (
            result["attempts"]
        ):
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

        for parsed_part in (
            result[
                "parsed_parts"
            ]
        ):
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

    # IMPORTANT:
    # No additional model call here.
    # Final selection is deterministic Python only.
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

    return {
        "success": (
            successful_original_chunks
            > 0
        ),
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
        "prompt_tokens": (
            prompt_tokens
        ),
        "completion_tokens": (
            completion_tokens
        ),
        "total_tokens": (
            total_tokens
        ),
        "output": final,
    }


# ============================================================
# PERSISTENT RESULTS
#
# Existing results remain across repeated runs.
# Running rfp2 later does NOT erase rfp1.
# Re-running the same document replaces only that document/model.
# ============================================================

def load_existing_results():
    if not RESULTS_FILE.exists():
        return {}

    try:
        with RESULTS_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        return (
            data
            if isinstance(
                data,
                dict,
            )
            else {}
        )

    except Exception as e:
        raise RuntimeError(
            f"Could not read existing results file: "
            f"{RESULTS_FILE}\n{e}"
        )


def save_results(data):
    temp_file = (
        RESULTS_FILE
        .with_suffix(
            ".json.tmp"
        )
    )

    with temp_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    temp_file.replace(
        RESULTS_FILE
    )


# ============================================================
# VALIDATION OF SELECTED FILES
# ============================================================

def validate_selected_documents():
    allowed = {
        "rfp1.txt",
        "rfp2.txt",
        "rfp3.txt",
        "rfp4.txt",
        "rfp5.txt",
    }

    if not DOCUMENTS_TO_RUN:
        raise SystemExit(
            "DOCUMENTS_TO_RUN is empty."
        )

    unknown = [
        name
        for name in DOCUMENTS_TO_RUN
        if name not in allowed
    ]

    if unknown:
        raise SystemExit(
            f"Unknown document(s): "
            f"{unknown}"
        )

    missing = [
        name
        for name in DOCUMENTS_TO_RUN
        if not (
            DOCUMENTS_DIR
            / name
        ).exists()
    ]

    if missing:
        raise SystemExit(
            f"Document file(s) "
            f"not found in "
            f"{DOCUMENTS_DIR}: "
            f"{missing}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    validate_selected_documents()

    results = (
        load_existing_results()
    )

    print(
        "\nRFP BENCHMARK - OPTIMIZED V5 + OPENAI"
    )

    print(
        "V3 architecture + "
        "Python-only field validation/scoring"
    )

    print(
        f"Documents selected: "
        f"{DOCUMENTS_TO_RUN}"
    )

    print(
        f"Results file: "
        f"{RESULTS_FILE}"
    )

    print(
        f"Chunk size="
        f"{CHUNK_SIZE_CHARS} chars"
        f" | overlap="
        f"{OVERLAP_CHARS} chars"
        f" | workers="
        f"{CHUNK_WORKERS}"
        f" | max output="
        f"{MAX_OUTPUT_TOKENS}"
    )

    for doc_num, filename in enumerate(
        DOCUMENTS_TO_RUN,
        1,
    ):
        print(
            "\n"
            + "=" * 70
        )

        print(
            f"DOCUMENT "
            f"{doc_num}/"
            f"{len(DOCUMENTS_TO_RUN)}: "
            f"{filename}"
        )

        print(
            "=" * 70
        )

        text = (
            DOCUMENTS_DIR
            / filename
        ).read_text(
            encoding="utf-8",
            errors="ignore",
        )

        results.setdefault(
            filename,
            {},
        )

        for model_name, info in (
            MODELS.items()
        ):
            print(
                f"\nRunning "
                f"{filename} "
                f"with "
                f"{model_name}..."
            )

            result = run_document(
                model_name,
                info,
                text,
            )

            results[
                filename
            ][
                model_name
            ] = result

            # Save immediately after each model.
            save_results(
                results
            )

            print(
                f"Saved "
                f"{filename} / "
                f"{model_name}"
                f" | latency="
                f"{result['time_seconds']}s"
                f" | chunks="
                f"{result['chunks_total']}"
                f" | tokens="
                f"{result['total_tokens']}"
            )

    print(
        "\nFINISHED ✅"
    )

    print(
        f"Saved/updated: "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()
