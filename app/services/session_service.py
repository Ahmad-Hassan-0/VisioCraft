"""
app/services/session_service.py
Thread-safe in-memory session management.
"""
import logging
import uuid
import time
import threading
from typing import Dict, Optional, Any
from pathlib import Path

import config

log = logging.getLogger('visiocraft.session')


class SessionService:
    """Manages user sessions for the web application."""

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._start_cleanup_thread()
        log.info("SessionService initialized (timeout=%ds, cleanup=%ds)",
                 config.SESSION_TIMEOUT, config.CLEANUP_INTERVAL)

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self.lock:
            self.sessions[session_id] = {
                'id': session_id,
                'created_at': time.time(),
                'last_accessed': time.time(),
                'image_path': None,
                'mask_path': None,
                'extracted_path': None,
                'inpainted_path': None,
                'canvas_state': None,
                'layers': [],
                'history': [],
            }
        log.info("Session CREATED: %s  (total active: %d)",
                 session_id, len(self.sessions))
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]['last_accessed'] = time.time()
                log.debug("Session GET: %s  (keys: %s)",
                          session_id, list(self.sessions[session_id].keys()))
                return self.sessions[session_id]
            log.warning("Session NOT FOUND: %s  (active: %d)",
                        session_id, len(self.sessions))
            return None

    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].update(data)
                self.sessions[session_id]['last_accessed'] = time.time()
                log.info("Session UPDATED: %s  (updated keys: %s)",
                         session_id, list(data.keys()))
                return True
            log.warning("Session UPDATE failed — not found: %s", session_id)
            return False

    def delete_session(self, session_id: str) -> bool:
        with self.lock:
            if session_id in self.sessions:
                self._cleanup_session_files(self.sessions[session_id])
                del self.sessions[session_id]
                log.info("Session DELETED: %s  (remaining: %d)",
                         session_id, len(self.sessions))
                return True
            log.warning("Session DELETE failed — not found: %s", session_id)
            return False

    def get_session_count(self) -> int:
        with self.lock:
            return len(self.sessions)

    def session_exists(self, session_id: str) -> bool:
        with self.lock:
            return session_id in self.sessions

    def add_layer_to_session(self, session_id: str, layer_data: Dict[str, Any]) -> bool:
        with self.lock:
            if session_id in self.sessions:
                if 'layers' not in self.sessions[session_id]:
                    self.sessions[session_id]['layers'] = []
                self.sessions[session_id]['layers'].append(layer_data)
                log.debug("Layer added to session %s (total: %d)",
                          session_id, len(self.sessions[session_id]['layers']))
                return True
            return False

    def add_to_history(self, session_id: str, action: str, data: Any = None):
        with self.lock:
            if session_id in self.sessions:
                if 'history' not in self.sessions[session_id]:
                    self.sessions[session_id]['history'] = []
                history = self.sessions[session_id]['history']
                history.append({
                    'action': action,
                    'data': data,
                    'timestamp': time.time(),
                })
                if len(history) > config.UNDO_HISTORY_SIZE:
                    history.pop(0)
                log.debug("History added to session %s: action=%s", session_id, action)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _cleanup_session_files(self, session: Dict[str, Any]):
        file_keys = ['image_path', 'mask_path', 'extracted_path', 'inpainted_path']
        for key in file_keys:
            if key in session and session[key]:
                path = Path(session[key])
                if path.exists():
                    try:
                        path.unlink()
                        log.debug("Cleaned up file: %s", path)
                    except Exception as e:
                        log.warning("Could not delete %s: %s", path, e)

    def _cleanup_expired_sessions(self):
        current_time = time.time()
        expired = []
        with self.lock:
            for session_id, session in self.sessions.items():
                if current_time - session['last_accessed'] > config.SESSION_TIMEOUT:
                    expired.append(session_id)
            for session_id in expired:
                self._cleanup_session_files(self.sessions[session_id])
                del self.sessions[session_id]
        if expired:
            log.info("Cleaned up %d expired session(s)", len(expired))

    def _start_cleanup_thread(self):
        def cleanup_loop():
            while True:
                time.sleep(config.CLEANUP_INTERVAL)
                self._cleanup_expired_sessions()
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
