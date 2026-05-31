"""audit-fence: Programmatic enforcement for LLM agent evidence."""

from .fence import (
    Fence,
    SearchRecord,
)
from .matching import extract_numbers, normalize_number
from .group import FenceGroup
from .prompts import PROMPTS
from .agent import AuditResult
from .snapshot import Snapshot, ToolCallRecord
from .tools import RipgrepBackend, SandboxedSearch
from .workflow import ClaimRecord, create_record_tool

__all__ = [
    "AuditResult",
    "ClaimRecord",
    "Fence",
    "FenceGroup",
    "PROMPTS",
    "RipgrepBackend",
    "SandboxedSearch",
    "SearchRecord",
    "Snapshot",
    "ToolCallRecord",
    "create_record_tool",
    "extract_numbers",
    "normalize_number",
]
__version__ = "0.9.0"
