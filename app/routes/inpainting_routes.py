"""
app/routes/inpainting_routes.py
Inpainting endpoints.
"""
import logging
import time
import traceback
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file, current_app
from PIL import Image

log = logging.getLogger('visiocraft.routes.inpainting')

inpainting_bp = Blueprint('inpainting', __name__)


@inpainting_bp.route('/inpaint', methods=['POST'])
def inpaint_background():
    sm = current_app.config['SESSION_SVC']
    inp = current_app.config['INPAINTING_SVC']
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            log.error("POST /api/inpaint — no JSON body received")
            return jsonify({'success': False, 'error': 'No JSON body'}), 400

        session_id = data.get('session_id')
        log.info("POST /api/inpaint — session=%s", session_id)

        if not inp or not inp.available:
            log.error("Inpainting service not available")
            return jsonify({'success': False, 'error': 'Inpainting not ready'}), 500

        s = sm.get_session(session_id)
        if not s:
            log.error("Inpaint: session not found: %s", session_id)
            return jsonify({'success': False, 'error': 'Session not found'}), 404

        image_path, mask_path = s.get('image_path'), s.get('mask_path')
        if not image_path or not mask_path:
            log.error("Inpaint: missing image=%s or mask=%s",
                      image_path, mask_path)
            return jsonify({'success': False, 'error': 'Image or mask missing'}), 400
        if not Path(image_path).exists() or not Path(mask_path).exists():
            log.error("Inpaint: files not found on disk — image=%s, mask=%s",
                      Path(image_path).exists(), Path(mask_path).exists())
            return jsonify({'success': False, 'error': 'File not found on disk'}), 400

        t0 = time.time()
        inpainted_path = inp.inpaint_background(image_path, mask_path,
                                                session_id=session_id)
        img = Image.open(inpainted_path)
        metadata = {
            'session_id': session_id,
            'method': inp.primary.__class__.__name__.replace('Inpainter', '').lower() if inp.primary else 'unknown',
            'dimensions': {'width': img.width, 'height': img.height},
            'processing_time': round(time.time() - t0, 2),
        }
        sm.update_session(session_id, {
            'inpainted_path': inpainted_path,
            'inpainting_metadata': metadata,
        })

        log.info("Inpaint: SUCCESS — session=%s, size=%dx%d, time=%.2fs",
                 session_id, img.width, img.height, metadata['processing_time'])
        return jsonify({
            'success': True,
            'inpainted_url': f'/api/inpainted/{session_id}',
            'metadata': metadata,
        })
    except Exception as e:
        log.error("Inpaint: FAILED — %s\n%s", e, traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@inpainting_bp.route('/inpainted/<session_id>')
def get_inpainted(session_id):
    sm = current_app.config['SESSION_SVC']
    s = sm.get_session(session_id)
    if not s or 'inpainted_path' not in s or not s['inpainted_path']:
        log.warning("GET /api/inpainted/%s — not found", session_id)
        return jsonify({'error': 'Not found'}), 404
    log.debug("Serving inpainted for session %s", session_id)
    return send_file(s['inpainted_path'])
