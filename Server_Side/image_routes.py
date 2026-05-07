"""
Server_Side/image_routes.py
Responsible ONLY for image processing API endpoints.
"""
from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename
from pathlib import Path
import traceback, time, json
import config

image_bp = Blueprint('images', __name__)

# Helper to get managers from app config
def mgr():
    return (
        current_app.config['SESSION_MGR'],
        current_app.config['PROCESSING_MGR'],
        current_app.config['OBJECT_MASKER']
    )

def get_session_or_404(session_mgr, session_id):
    s = session_mgr.get_session(session_id)
    if not s:
        raise LookupError('Session not found')
    return s

# ── Session ──────────────────────────────────────────
@image_bp.route('/session/create', methods=['POST'])
def create_session():
    sm, _, _ = mgr()
    try:
        return jsonify({'success': True, 'session_id': sm.create_session()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ── Upload ────────────────────────────────────────────
@image_bp.route('/upload', methods=['POST'])
def upload_image():
    sm, _, _ = mgr()
    try:
        file       = request.files.get('file')
        session_id = request.form.get('session_id')
        if not file or not session_id or file.filename == '':
            return jsonify({'success': False, 'error': 'Missing file or session_id'}), 400
        if not config.validate_image_format(file.filename):
            return jsonify({'success': False, 'error': 'Unsupported format'}), 400
        filename  = secure_filename(file.filename)
        save_path = config.get_temp_path(f"{session_id}_upload{Path(filename).suffix}")
        file.save(save_path)
        sm.update_session(session_id, {'image_path': str(save_path)})
        return jsonify({'success': True, 'image_url': f'/api/image/{session_id}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@image_bp.route('/image/<session_id>')
def get_image(session_id):
    sm, _, _ = mgr()
    s = sm.get_session(session_id)
    if not s or 'image_path' not in s:
        return jsonify({'error': 'Not found'}), 404
    return send_file(s['image_path'])

# ── Segmentation ──────────────────────────────────────
@image_bp.route('/segment', methods=['POST'])
def segment_object():
    sm, pm, om = mgr()
    try:
        data       = request.json
        session_id = data.get('session_id')
        points     = data.get('points', [])
        s          = get_session_or_404(sm, session_id)
        image_path = s.get('image_path')
        if not image_path:
            return jsonify({'error': 'No image uploaded'}), 400
        mask_path    = om.segment_from_points(image_path, points, session_id)
        preview_path = pm.create_preview(image_path, mask_path, session_id)
        sm.update_session(session_id, {'mask_path': mask_path, 'preview_path': preview_path})
        return jsonify({'success': True, 'mask_url': f'/api/mask/{session_id}', 'preview_url': f'/api/preview/{session_id}'})
    except LookupError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@image_bp.route('/preview/<session_id>')
def get_preview(session_id):
    sm, _, _ = mgr()
    s = sm.get_session(session_id)
    if not s or 'preview_path' not in s:
        return jsonify({'error': 'Not found'}), 404
    return send_file(s['preview_path'])

@image_bp.route('/mask/<session_id>')
def get_mask(session_id):
    sm, _, _ = mgr()
    s = sm.get_session(session_id)
    if not s or 'mask_path' not in s:
        return jsonify({'error': 'Not found'}), 404
    return send_file(s['mask_path'])

# ── Extraction ────────────────────────────────────────
@image_bp.route('/extract', methods=['POST'])
def extract_object():
    sm, pm, _ = mgr()
    try:
        session_id = request.json.get('session_id')
        s          = get_session_or_404(sm, session_id)
        image_path, mask_path = s.get('image_path'), s.get('mask_path')
        if not image_path or not mask_path:
            return jsonify({'error': 'Image or mask missing'}), 400
        t0             = time.time()
        extracted_path = pm.extract_object(image_path, mask_path, session_id)
        from PIL import Image
        img      = Image.open(extracted_path)
        metadata = {'session_id': session_id, 'dimensions': {'width': img.width, 'height': img.height},
                    'format': 'PNG', 'has_alpha': img.mode == 'RGBA', 'processing_time': round(time.time()-t0, 2)}
        sm.update_session(session_id, {'extracted_path': extracted_path, 'extraction_metadata': metadata})
        return jsonify({'success': True, 'object_url': f'/api/object/{session_id}', 'metadata': metadata})
    except LookupError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@image_bp.route('/object/<session_id>')
def get_extracted_object(session_id):
    sm, _, _ = mgr()
    s = sm.get_session(session_id)
    if not s or 'extracted_path' not in s:
        return jsonify({'error': 'Not found'}), 404
    return send_file(s['extracted_path'])

# ── Inpainting ────────────────────────────────────────
@image_bp.route('/inpaint', methods=['POST'])
def inpaint_background():
    sm, pm, _ = mgr()
    try:
        data       = request.json
        session_id = data.get('session_id')
        # method = data.get('method', config.DEFAULT_INPAINT_METHOD)  # We ignore method now
        if not pm or not pm.inpainter:  # FIXED: check inpainter instead of methods
            return jsonify({'success': False, 'error': 'Inpainting not ready (Flux not loaded)'}), 500
        
        s          = get_session_or_404(sm, session_id)
        image_path, mask_path = s.get('image_path'), s.get('mask_path')
        if not image_path or not mask_path:
            return jsonify({'error': 'Image or mask missing'}), 400
        if not Path(image_path).exists() or not Path(mask_path).exists():
            return jsonify({'error': 'File not found on disk'}), 400
        
        t0             = time.time()
        inpainted_path = pm.inpaint_background(image_path, mask_path, session_id=session_id)  # FIXED: removed method arg
        from PIL import Image
        img      = Image.open(inpainted_path)
        metadata = {'session_id': session_id, 'method': 'flux',  # Hardcoded flux
                    'dimensions': {'width': img.width, 'height': img.height},
                    'processing_time': round(time.time()-t0, 2)}
        sm.update_session(session_id, {'inpainted_path': inpainted_path, 'inpainting_metadata': metadata})
        return jsonify({'success': True, 'inpainted_url': f'/api/inpainted/{session_id}', 'metadata': metadata})
    except LookupError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@image_bp.route('/inpainted/<session_id>')
def get_inpainted(session_id):
    sm, _, _ = mgr()
    s = sm.get_session(session_id)
    if not s or 'inpainted_path' not in s:
        return jsonify({'error': 'Not found'}), 404
    return send_file(s['inpainted_path'])

# ── Generation ────────────────────────────────────────
@image_bp.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        data       = request.json
        prompt     = data.get('prompt')
        session_id = data.get('session_id')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        from Server_Side.Image_Processing.Image_Generation import generate_image_from_prompt
        generate_image_from_prompt(prompt, session_id)
        return jsonify({'success': True, 'image_url': f'/api/generated/{session_id}'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@image_bp.route('/generated/<session_id>')
def get_generated(session_id):
    path = config.get_output_path(f'generated_{session_id}.png')
    if not Path(path).exists():
        return jsonify({'error': 'Not found'}), 404
    return send_file(path)

# ── Metadata ──────────────────────────────────────────
@image_bp.route('/metadata/<session_id>')
def get_metadata(session_id):
    sm, _, _ = mgr()
    try:
        s = get_session_or_404(sm, session_id)
        metadata = {
            'project': 'VisioCraft AI', 'session_id': session_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'extraction': s.get('extraction_metadata', {}),
            'inpainting': s.get('inpainting_metadata', {}),
        }
        meta_path = config.get_output_path(f'metadata_{session_id}.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        return send_file(meta_path, as_attachment=True,
                         download_name=f'visiocraft_metadata_{session_id}.json',
                         mimetype='application/json')
    except LookupError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Health & Debug ────────────────────────────────────
@image_bp.route('/methods')
def get_methods():
    return jsonify({'success': True, 'methods': ['flux']})  # Fixed: only flux

@image_bp.route('/health')
def health_check():
    sm, pm, om = mgr()
    return jsonify({
        'status': 'healthy', 'version': '1.0.0',
        'inpainting_methods': ['flux'] if pm and pm.inpainter else [],  # Fixed: no pm.methods
        'sam_available': om.sam_available if om else False,
        'active_sessions': sm.get_session_count() if sm else 0
    })

@image_bp.route('/upload-generated', methods=['POST'])
def upload_generated():
    sm, _, _ = mgr()
    try:
        session_id = request.json.get('session_id')
        s = get_session_or_404(sm, session_id)
        
        # Find the generated image
        generated_path = config.get_output_path(f'generated_{session_id}.png')
        if not Path(generated_path).exists():
            return jsonify({'error': 'Generated image not found'}), 404
        
        # Set it as the session's image_path so segmentation works
        sm.update_session(session_id, {'image_path': str(generated_path)})
        return jsonify({'success': True})
    except LookupError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500