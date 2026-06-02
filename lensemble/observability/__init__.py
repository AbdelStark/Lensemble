"""lensemble.observability — structured logging, metrics, redaction (docs/rfcs/RFC-0015).

The redaction guard (``INV-RESIDENCY``) is the single allow-list every log/metric/diagnostic record
passes before a sink write. Structured logging and the metric taxonomy land with #57 / #58.
"""

from __future__ import annotations

from lensemble.observability.redaction import redact, redact_record

__all__ = ["redact", "redact_record"]
