"""
app/routes/session_routes.py
Session creation endpoint.
"""
import logging
from flask import Blueprint, jsonify, current_app

log = logging.getLogger('visiocraft.routes.session')

session_bp = Blueprint('sessions', __name__)


@session_bp.route('/session/create', methods=['POST'])
def create_session():
    log.info("POST /api/session/create")
    sm = current_app.config['SESSION_SVC']
    try:
        session_id = sm.create_session()
        log.info("Session created: %s", session_id)
        return jsonify({'success': True, 'session_id': session_id})
    except Exception as e:
        log.error("Failed to create session: %s", e, exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@session_bp.route('/session/progress/<session_id>', methods=['GET'])
def get_progress(session_id):
    from app.services.progress_manager import progress_manager
    progress = progress_manager.get_progress(session_id)
    return jsonify(progress)
