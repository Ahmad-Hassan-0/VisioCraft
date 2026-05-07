"""
app/routes/generation_routes.py
AI image generation endpoints.
"""
import logging
import traceback
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file, current_app

import config

log = logging.getLogger('visiocraft.routes.generation')

generation_bp = Blueprint('generation', __name__)


@generation_bp.route('/generate-image', methods=['POST'])
def generate_image():
    gen = current_app.config['GENERATION_SVC']
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            log.error("POST /api/generate-image — no JSON body received")
            return jsonify({'success': False, 'error': 'No JSON body'}), 400

        prompt = data.get('prompt')
        session_id = data.get('session_id')
        log.info("POST /api/generate-image — session=%s, prompt='%s'",
                 session_id, (prompt or '')[:80])

        if not prompt:
            log.warning("Generate: no prompt provided")
            return jsonify({'success': False, 'error': 'No prompt provided'}), 400

        gen.generate(prompt, session_id)
        log.info("Generate: SUCCESS — session=%s", session_id)
        return jsonify({'success': True, 'image_url': f'/api/generated/{session_id}'})
    except Exception as e:
        log.error("Generate: FAILED — %s\n%s", e, traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@generation_bp.route('/generated/<session_id>')
def get_generated(session_id):
    path = config.get_output_path(f'generated_{session_id}.png')
    if not Path(path).exists():
        log.warning("GET /api/generated/%s — not found at %s", session_id, path)
        return jsonify({'error': 'Not found'}), 404
    log.debug("Serving generated image for session %s", session_id)
    return send_file(path)
