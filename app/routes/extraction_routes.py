"""
app/routes/extraction_routes.py
Object extraction endpoints.
"""
import logging
import time
import traceback
from flask import Blueprint, request, jsonify, send_file, current_app
from PIL import Image

log = logging.getLogger('visiocraft.routes.extraction')

extraction_bp = Blueprint('extraction', __name__)


@extraction_bp.route('/extract', methods=['POST'])
def extract_object():
    sm = current_app.config['SESSION_SVC']
    ext = current_app.config['EXTRACTION_SVC']
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            log.error("POST /api/extract — no JSON body received")
            return jsonify({'success': False, 'error': 'No JSON body'}), 400

        session_id = data.get('session_id')
        log.info("POST /api/extract — session=%s", session_id)

        s = sm.get_session(session_id)
        if not s:
            log.error("Extract: session not found: %s", session_id)
            return jsonify({'success': False, 'error': 'Session not found'}), 404

        image_path, mask_path = s.get('image_path'), s.get('mask_path')
        if not image_path or not mask_path:
            log.error("Extract: missing image_path=%s or mask_path=%s",
                      image_path, mask_path)
            return jsonify({'success': False, 'error': 'Image or mask missing'}), 400

        t0 = time.time()
        extracted_path = ext.extract_object(image_path, mask_path, session_id)
        img = Image.open(extracted_path)
        metadata = {
            'session_id': session_id,
            'dimensions': {'width': img.width, 'height': img.height},
            'format': 'PNG',
            'has_alpha': img.mode == 'RGBA',
            'processing_time': round(time.time() - t0, 2),
        }
        sm.update_session(session_id, {
            'extracted_path': extracted_path,
            'extraction_metadata': metadata,
        })

        log.info("Extract: SUCCESS — session=%s, size=%dx%d, time=%.2fs",
                 session_id, img.width, img.height, metadata['processing_time'])
        return jsonify({
            'success': True,
            'object_url': f'/api/object/{session_id}',
            'metadata': metadata,
        })
    except Exception as e:
        log.error("Extract: FAILED — %s\n%s", e, traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@extraction_bp.route('/object/<session_id>')
def get_extracted_object(session_id):
    sm = current_app.config['SESSION_SVC']
    s = sm.get_session(session_id)
    if not s or 'extracted_path' not in s or not s['extracted_path']:
        log.warning("GET /api/object/%s — not found", session_id)
        return jsonify({'error': 'Not found'}), 404
    log.debug("Serving extracted object for session %s", session_id)
    return send_file(s['extracted_path'])
