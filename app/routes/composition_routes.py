"""
app/routes/composition_routes.py
Image composition endpoint — fixes the missing /api/compose route.
"""
import logging
import traceback
from flask import Blueprint, request, jsonify, send_file, current_app

log = logging.getLogger('visiocraft.routes.composition')

composition_bp = Blueprint('composition', __name__)


@composition_bp.route('/compose', methods=['POST'])
def compose_image():
    sm = current_app.config['SESSION_SVC']
    comp = current_app.config['COMPOSITION_SVC']
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            log.error("POST /api/compose — no JSON body received")
            return jsonify({'success': False, 'error': 'No JSON body'}), 400

        session_id = data.get('session_id')
        transform = data.get('transform', {})
        log.info("POST /api/compose — session=%s, transform=%s",
                 session_id, transform)

        s = sm.get_session(session_id)
        if not s:
            log.error("Compose: session not found: %s", session_id)
            return jsonify({'success': False, 'error': 'Session not found'}), 404

        composite_path = comp.compose(s, transform)
        sm.update_session(session_id, {'composite_path': composite_path})

        log.info("Compose: SUCCESS — session=%s, output=%s",
                 session_id, composite_path)
        return jsonify({
            'success': True,
            'composite_url': f'/api/composite/{session_id}',
        })
    except Exception as e:
        log.error("Compose: FAILED — %s\n%s", e, traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@composition_bp.route('/composite/<session_id>')
def get_composite(session_id):
    sm = current_app.config['SESSION_SVC']
    s = sm.get_session(session_id)
    if not s or 'composite_path' not in s or not s['composite_path']:
        log.warning("GET /api/composite/%s — not found", session_id)
        return jsonify({'error': 'Not found'}), 404
    log.debug("Serving composite for session %s", session_id)
    return send_file(s['composite_path'])
