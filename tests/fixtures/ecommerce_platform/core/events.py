"""Domain event pub-sub subsystem for auditing and checkout workflow telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from typing import Callable


@dataclass
class DomainEvent:
    event_type: str
    payload: dict
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)


class EventDispatcher:
    """Dispatches domain events to registered synchronous observers."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[DomainEvent], None]]] = {}
        self.audit_log: list[DomainEvent] = []

    def subscribe(self, event_type: str, handler: Callable[[DomainEvent], None]) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def emit(self, event: DomainEvent) -> None:
        self.audit_log.append(event)
        for handler in self._handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception:
                pass
