"""
Client_Side/SessionManager.py
Manages user sessions and temporary data storage
"""

import uuid
import time
from typing import Dict, Optional, Any
import threading

import config


class SessionManager:
    """Manages user sessions for the web application"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._start_cleanup_thread()
    
    def create_session(self) -> str:
        """Create a new session and return session ID"""
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
                'history': []
            }
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data by ID"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]['last_accessed'] = time.time()
                return self.sessions[session_id]
            return None
    
    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session with new data"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].update(data)
                self.sessions[session_id]['last_accessed'] = time.time()
                return True
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        with self.lock:
            if session_id in self.sessions:
                # Clean up associated files
                session = self.sessions[session_id]
                self._cleanup_session_files(session)
                del self.sessions[session_id]
                return True
            return False
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions"""
        with self.lock:
            return self.sessions.copy()
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session exists"""
        with self.lock:
            return session_id in self.sessions
    
    def _cleanup_session_files(self, session: Dict[str, Any]):
        """Clean up files associated with a session"""
        from pathlib import Path
        
        file_keys = ['image_path', 'mask_path', 'extracted_path', 'inpainted_path']
        
        for key in file_keys:
            if key in session and session[key]:
                path = Path(session[key])
                if path.exists():
                    try:
                        path.unlink()
                    except Exception as e:
                        print(f"Warning: Could not delete {path}: {e}")
    
    def _cleanup_expired_sessions(self):
        """Remove sessions that have exceeded timeout"""
        current_time = time.time()
        expired = []
        
        with self.lock:
            for session_id, session in self.sessions.items():
                if current_time - session['last_accessed'] > config.SESSION_TIMEOUT:
                    expired.append(session_id)
            
            for session_id in expired:
                session = self.sessions[session_id]
                self._cleanup_session_files(session)
                del self.sessions[session_id]
        
        if expired:
            print(f"🧹 Cleaned up {len(expired)} expired session(s)")
    
    def _start_cleanup_thread(self):
        """Start background thread for periodic cleanup"""
        def cleanup_loop():
            while True:
                time.sleep(config.CLEANUP_INTERVAL)
                self._cleanup_expired_sessions()
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        with self.lock:
            return len(self.sessions)
    
    def add_layer_to_session(self, session_id: str, layer_data: Dict[str, Any]) -> bool:
        """Add a layer to session's canvas"""
        with self.lock:
            if session_id in self.sessions:
                if 'layers' not in self.sessions[session_id]:
                    self.sessions[session_id]['layers'] = []
                self.sessions[session_id]['layers'].append(layer_data)
                return True
            return False
    
    def add_to_history(self, session_id: str, action: str, data: Any = None):
        """Add action to session history for undo/redo"""
        with self.lock:
            if session_id in self.sessions:
                if 'history' not in self.sessions[session_id]:
                    self.sessions[session_id]['history'] = []
                
                history = self.sessions[session_id]['history']
                history.append({
                    'action': action,
                    'data': data,
                    'timestamp': time.time()
                })
                
                # Limit history size
                if len(history) > config.UNDO_HISTORY_SIZE:
                    history.pop(0)