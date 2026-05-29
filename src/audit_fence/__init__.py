"""audit-fence: Programmatic enforcement for LLM agent evidence."""

from .fence import (
    Fence,
    SearchRecord,
    ValidationResult,
    extract_numbers,
    normalize_number,
)
from .group import FenceGroup

__all__ = [
    "Fence",
    "FenceGroup",
    "SearchRecord",
    "ValidationResult",
    "extract_numbers",
    "normalize_number",
]
__version__ = "0.1.0"
