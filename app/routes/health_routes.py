"""
app/routes/health_routes.py
Health check, methods, and metadata endpoints.
"""
import logging
import time
import json
from flask import Blueprint, request, jsonify, send_file, current_app

import config

log = logging.getLogger('visiocraft.routes.health')

health_bp = Blueprint('health', __name__)


@health_bp.route('/methods')
def get_methods():
    log.debug("GET /api/methods")
    return jsonify({'success': True, 'methods': ['opencv']})


@health_bp.route('/health')
def health_check():
    sm = current_app.config['SESSION_SVC']
    seg = current_app.config['SEGMENTATION_SVC']
    inp = current_app.config['INPAINTING_SVC']
    result = {
        'status': 'healthy',
        'version': '2.0.0',
        'inpainting_available': inp.available if inp else False,
        'sam_available': seg.sam_available if seg else False,
        'active_sessions': sm.get_session_count() if sm else 0,
    }
    log.info("GET /api/health → %s", result)
    return jsonify(result)


@health_bp.route('/metadata/<session_id>')
def get_metadata(session_id):
    sm = current_app.config['SESSION_SVC']
    log.info("GET /api/metadata/%s", session_id)
    try:
        s = sm.get_session(session_id)
        if not s:
            log.warning("Metadata: session not found: %s", session_id)
            return jsonify({'error': 'Session not found'}), 404
        metadata = {
            'project': 'VisioCraft AI',
            'session_id': session_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'extraction': s.get('extraction_metadata', {}),
            'inpainting': s.get('inpainting_metadata', {}),
        }
        meta_path = config.get_output_path(f'metadata_{session_id}.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        log.info("Metadata exported: %s", meta_path)
        return send_file(
            meta_path, as_attachment=True,
            download_name=f'visiocraft_metadata_{session_id}.json',
            mimetype='application/json',
        )
    except Exception as e:
        log.error("Metadata export failed: %s", e, exc_info=True)
        return jsonify({'error': str(e)}), 500
