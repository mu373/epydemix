"""Format library events; applications choose handlers, levels and destinations.

Run counters and timers belong to calibration, not this output module. Events
come from the parent process and contain no model inputs or particle values.
"""

import json
import logging
import math
from datetime import datetime, timezone

logger = logging.getLogger("epydemix.calibration")
logger.addHandler(logging.NullHandler())


def emit(event, *, level=logging.INFO, **fields):
    """Send an event using standard logging.extra without configuring root logging."""
    logger.log(level, event, extra={"epydemix": {"event": event, **fields}})


def _json_values(value):
    """Use explicit strings for nonfinite thresholds in strict JSON Lines."""
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


class JSONFormatter(logging.Formatter):
    """Render UTC timestamps and event fields as one strict JSON object per line."""

    def format(self, record):
        fields = getattr(record, "epydemix", {"message": record.getMessage()})
        return json.dumps(
            _json_values(
                {
                    "timestamp": datetime.fromtimestamp(
                        record.created, timezone.utc
                    ).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    **fields,
                }
            ),
            allow_nan=False,
        )
