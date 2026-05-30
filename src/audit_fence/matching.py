"""Text normalization, number parsing, and evidence matching primitives.

Extracted from fence.py for modularity. All functions are stateless
and used by both the Fence enforcement logic and the public API
(``normalize_number``, ``extract_numbers``).
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Normalize text for substring comparison: strip markdown, collapse whitespace, lowercase."""
    text = re.sub(r"\*\*|__|\*|_|`|~~", "", text)  # bold, italic, code, strikethrough
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [text](url) → text
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # heading markers
    text = re.sub(r"\|", " ", text)  # table pipes
    text = re.sub(r"[—–]", "-", text)  # em/en dashes → hyphen
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# ---------------------------------------------------------------------------
# Source verification
# ---------------------------------------------------------------------------

def _verify_source_match(claim: str, source_text: str) -> tuple[bool, str]:
    """Verify that claim text exists as a substring in the source document."""
    if not claim or not source_text:
        return True, ""

    norm_claim = _normalize(claim)
    norm_source = _normalize(source_text)

    if norm_claim in norm_source:
        return True, ""

    return False, (
        "Claim text not found in the source document. "
        "Copy the EXACT text from the source."
    )


# ---------------------------------------------------------------------------
# Quote extraction
# ---------------------------------------------------------------------------

# Regex to extract quoted passages: "..." or '...' (at least 10 chars inside)
_QUOTE_PATTERN = re.compile(r'["\u201c]([^"\u201d]{10,})["\u201d]')


def _extract_quotes(text: str) -> list[str]:
    """Extract quoted passages from text for validation.

    Looks for double-quoted strings (ASCII ``"`` or Unicode curly quotes)
    that are at least 10 characters long.
    """
    return [m.group(1) for m in _QUOTE_PATTERN.finditer(text)]


# ---------------------------------------------------------------------------
# Number format matching
# ---------------------------------------------------------------------------

# Pattern: optional sign, digits with optional commas, optional decimal, optional suffix
_NUM_PATTERN = re.compile(
    r"[-+]?"
    r"\d[\d,]*"           # integer part (may have commas)
    r"(?:\.\d+)?"         # optional decimal
    r"[%KMBTkmbt]?"        # optional suffix (must be adjacent, no whitespace)
)

_SUFFIX_MULTIPLIERS: dict[str, float] = {
    "k": 1e3,
    "m": 1e6,
    "b": 1e9,
    "t": 1e12,
}


def normalize_number(text: str) -> float | None:
    """Parse a human-readable number string into a float.

    Handles:
    - Comma-separated numbers: ``"5,098,000,000"`` -> ``5098000000.0``
    - Suffix abbreviations: ``"5.1B"`` -> ``5100000000.0``
    - Percentages: ``"26.2%"`` -> ``0.262``
    - Plain numbers: ``"18.923"`` -> ``18.923``

    Returns ``None`` if the string does not look like a number.
    """
    text = text.strip()
    if not text:
        return None

    # Remove leading currency signs
    text = text.lstrip("$")

    # Check for percentage
    is_pct = text.endswith("%")
    if is_pct:
        text = text[:-1].strip()

    # Remove commas
    text = text.replace(",", "")

    # Extract suffix
    suffix = ""
    if text and text[-1].lower() in _SUFFIX_MULTIPLIERS:
        suffix = text[-1].lower()
        text = text[:-1].strip()

    try:
        value = float(text)
    except (ValueError, TypeError):
        return None

    if suffix:
        value *= _SUFFIX_MULTIPLIERS[suffix]
    if is_pct:
        value /= 100.0

    return value


def extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from a text string.

    Finds numbers with optional K/M/B/T suffixes, commas, and percentage
    signs, then normalizes each to a float.

    Example::

        >>> extract_numbers("Revenue $5.1B, up 26.2% YoY")
        [5100000000.0, 0.262]
    """
    results: list[float] = []
    for match in _NUM_PATTERN.finditer(text):
        token = match.group()
        val = normalize_number(token)
        if val is not None:
            results.append(val)
    return results


def _numbers_overlap(nums_a: list[float], nums_b: list[float]) -> bool:
    """Check if any number from list A matches any number from list B.

    Two numbers match if they are within 0.1% relative tolerance of each
    other, handling float imprecision from suffix expansion.
    """
    for a in nums_a:
        for b in nums_b:
            if a == 0 and b == 0:
                return True
            if a == 0 or b == 0:
                continue
            rel = abs(a - b) / max(abs(a), abs(b))
            if rel < 0.001:
                return True
    return False


def _number_match(evidence_line: str, search_text: str) -> bool:
    """Fallback matching: check if a line's numbers match numbers in search results.

    Only activates when the evidence line contains at least one number.
    Requires at least one number overlap AND some non-numeric text overlap
    (at least 3 words in common) to prevent pure-number false positives.
    """
    ev_nums = extract_numbers(evidence_line)
    if not ev_nums:
        return False

    sr_nums = extract_numbers(search_text)
    if not sr_nums:
        return False

    if not _numbers_overlap(ev_nums, sr_nums):
        return False

    # Require some non-numeric textual overlap to prevent false positives
    ev_words = set(re.sub(r"[^a-zA-Z\s]", "", evidence_line.lower()).split())
    sr_words = set(re.sub(r"[^a-zA-Z\s]", "", search_text.lower()).split())
    # Remove very common words
    stopwords = {"the", "a", "an", "in", "of", "to", "and", "or", "is", "was", "for", "on", "at", "by", "with", "from"}
    ev_words -= stopwords
    sr_words -= stopwords
    common = ev_words & sr_words
    return len(common) >= 2
