"""
app/services/progress_manager.py
Simple thread-safe singleton for tracking long-running task progress.
"""
import threading
import logging

log = logging.getLogger('visiocraft.progress')

class ProgressManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ProgressManager, cls).__new__(cls)
                cls._instance.progress = {}
                cls._instance.system_ready = False
        return cls._instance

    def set_system_ready(self, ready: bool):
        """Update global system readiness flag."""
        with self._lock:
            self.system_ready = ready
            log.info("System Ready: %s", ready)

    def set_progress(self, session_id: str, percentage: int, message: str = ""):
        """Update progress for a session."""
        with self._lock:
            self.progress[session_id] = {
                "percentage": int(percentage),
                "message": message
            }

    def get_progress(self, session_id: str):
        """Retrieve progress for a session."""
        with self._lock:
            # We want to return a copy or a new dict to avoid external mutation
            base = self.progress.get(session_id, {"percentage": 0, "message": "Initializing..."})
            return {
                "percentage": base["percentage"],
                "message": base["message"],
                "system_ready": self.system_ready
            }

    def clear_progress(self, session_id: str):
        """Remove progress data after completion."""
        with self._lock:
            if session_id in self.progress:
                del self.progress[session_id]

progress_manager = ProgressManager()
