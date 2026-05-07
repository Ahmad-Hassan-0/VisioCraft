"""
app/routes/segmentation_routes.py
Segmentation and preview endpoints.
"""
import logging
import traceback
from flask import Blueprint, request, jsonify, send_file, current_app

log = logging.getLogger('visiocraft.routes.segmentation')

segmentation_bp = Blueprint('segmentation', __name__)


@segmentation_bp.route('/segment', methods=['POST'])
def segment_object():
    sm = current_app.config['SESSION_SVC']
    seg = current_app.config['SEGMENTATION_SVC']
    ext = current_app.config['EXTRACTION_SVC']
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            log.error("POST /api/segment — no JSON body received")
            return jsonify({'success': False, 'error': 'No JSON body'}), 400

        session_id = data.get('session_id')
        points = data.get('points', [])
        log.info("POST /api/segment — session=%s, points=%d", session_id, len(points))

        if not session_id:
            log.error("Segment: missing session_id in request body")
            return jsonify({'success': False, 'error': 'Missing session_id'}), 400

        s = sm.get_session(session_id)
        if not s:
            log.error("Segment: session not found: %s", session_id)
            return jsonify({'success': False, 'error': 'Session not found'}), 404

        image_path = s.get('image_path')
        if not image_path:
            log.error("Segment: no image uploaded for session %s", session_id)
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400

        log.info("Segment: starting segmentation (image=%s)", image_path)
        mask_path = seg.segment_from_points(image_path, points, session_id)
        log.info("Segment: mask created at %s", mask_path)

        log.info("Segment: creating preview...")
        preview_path = ext.create_preview(image_path, mask_path, session_id)
        log.info("Segment: preview created at %s", preview_path)

        sm.update_session(session_id, {
            'mask_path': mask_path,
            'preview_path': preview_path,
        })

        log.info("Segment: SUCCESS — session=%s", session_id)
        return jsonify({
            'success': True,
            'mask_url': f'/api/mask/{session_id}',
            'preview_url': f'/api/preview/{session_id}',
        })
    except Exception as e:
        log.error("Segment: FAILED — %s\n%s", e, traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@segmentation_bp.route('/preview/<session_id>')
def get_preview(session_id):
    sm = current_app.config['SESSION_SVC']
    s = sm.get_session(session_id)
    if not s or 'preview_path' not in s or not s['preview_path']:
        log.warning("GET /api/preview/%s — not found", session_id)
        return jsonify({'error': 'Not found'}), 404
    log.debug("Serving preview for session %s", session_id)
    return send_file(s['preview_path'])


@segmentation_bp.route('/mask/<session_id>')
def get_mask(session_id):
    sm = current_app.config['SESSION_SVC']
    s = sm.get_session(session_id)
    if not s or 'mask_path' not in s or not s['mask_path']:
        log.warning("GET /api/mask/%s — not found", session_id)
        return jsonify({'error': 'Not found'}), 404
    log.debug("Serving mask for session %s", session_id)
    return send_file(s['mask_path'])
