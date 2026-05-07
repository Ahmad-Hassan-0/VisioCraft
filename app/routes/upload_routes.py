"""
app/routes/upload_routes.py
File upload and image serving endpoints.
"""
import logging
from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename
from pathlib import Path

import config

log = logging.getLogger('visiocraft.routes.upload')

upload_bp = Blueprint('uploads', __name__)


@upload_bp.route('/upload', methods=['POST'])
def upload_image():
    sm = current_app.config['SESSION_SVC']
    try:
        file = request.files.get('file')
        session_id = request.form.get('session_id')
        log.info("POST /api/upload — session=%s, file=%s",
                 session_id, file.filename if file else 'None')

        if not file or not session_id or file.filename == '':
            log.warning("Upload rejected: missing file or session_id")
            return jsonify({'success': False, 'error': 'Missing file or session_id'}), 400
        if not config.validate_image_format(file.filename):
            log.warning("Upload rejected: unsupported format '%s'", file.filename)
            return jsonify({'success': False, 'error': 'Unsupported format'}), 400

        filename = secure_filename(file.filename)
        save_path = config.get_temp_path(f"{session_id}_upload{Path(filename).suffix}")
        file.save(save_path)
        log.info("File saved: %s (size: %d bytes)", save_path, save_path.stat().st_size)

        sm.update_session(session_id, {'image_path': str(save_path)})
        log.info("Upload complete — session=%s, url=/api/image/%s", session_id, session_id)
        return jsonify({'success': True, 'image_url': f'/api/image/{session_id}'})
    except Exception as e:
        log.error("Upload failed: %s", e, exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@upload_bp.route('/image/<session_id>')
def get_image(session_id):
    sm = current_app.config['SESSION_SVC']
    s = sm.get_session(session_id)
    if not s or 'image_path' not in s or not s['image_path']:
        log.warning("GET /api/image/%s — not found", session_id)
        return jsonify({'error': 'Not found'}), 404
    log.debug("Serving image for session %s: %s", session_id, s['image_path'])
    return send_file(s['image_path'])


@upload_bp.route('/upload-generated', methods=['POST'])
def upload_generated():
    sm = current_app.config['SESSION_SVC']
    try:
        session_id = request.json.get('session_id')
        log.info("POST /api/upload-generated — session=%s", session_id)
        s = sm.get_session(session_id)
        if not s:
            log.warning("Session not found: %s", session_id)
            return jsonify({'error': 'Session not found'}), 404
        generated_path = config.get_output_path(f'generated_{session_id}.png')
        if not Path(generated_path).exists():
            log.warning("Generated image not found: %s", generated_path)
            return jsonify({'error': 'Generated image not found'}), 404
        sm.update_session(session_id, {'image_path': str(generated_path)})
        log.info("Generated image linked to session %s", session_id)
        return jsonify({'success': True})
    except Exception as e:
        log.error("upload-generated failed: %s", e, exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
