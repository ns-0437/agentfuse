"""Application-owned progress evidence; tool text alone is not task success."""
from typing import Any, Callable, Optional

ProgressValidator = Callable[[str, Any], Optional[dict]]


def progress_state(validator: Optional[ProgressValidator], name: str,
                   result: Any) -> Optional[dict]:
    """Return stable milestone data, or abstain when no validator is supplied.

    Validators should return the same dictionary for the same completed work.
    Avoid timestamps and request IDs: changing identity is not new progress.
    A validator failure propagates rather than inventing successful evidence.
    """
    if validator is None:
        return None
    state = validator(name, result)
    if state is not None and not isinstance(state, dict):
        raise TypeError("progress_validator must return a milestone dict or None")
    return state
