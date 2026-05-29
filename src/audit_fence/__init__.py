"""audit-fence: Programmatic enforcement for LLM agent evidence."""

from .fence import (
    Fence,
    SearchRecord,
    ValidationResult,
    extract_numbers,
    normalize_number,
)
from .group import FenceGroup
from .prompts import PROMPTS
from .tools import SandboxedSearch
from .workflow import ClaimRecord, create_record_tool

__all__ = [
    "ClaimRecord",
    "Fence",
    "FenceGroup",
    "PROMPTS",
    "SandboxedSearch",
    "SearchRecord",
    "ValidationResult",
    "create_record_tool",
    "extract_numbers",
    "normalize_number",
]
__version__ = "0.1.0"
