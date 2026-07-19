"""Thread-safe emergency-stop state shared by every command session."""

from __future__ import annotations

from threading import RLock


class EmergencyStopState:
    """A small process-local, latched emergency-stop flag.

    Controllers keep their own hardware-level latch as a second line of
    defence.  This object is the application-level source of truth consumed by
    every AgentLoop and the LAN runtime.
    """

    def __init__(self, active: bool = False):
        self._active = active
        self._lock = RLock()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def activate(self) -> bool:
        """Latch the stop.  Returns True only on the inactive -> active edge."""
        with self._lock:
            changed = not self._active
            self._active = True
            return changed

    def release(self) -> bool:
        """Release the latch.  Returns True only on the active -> inactive edge."""
        with self._lock:
            changed = self._active
            self._active = False
            return changed
