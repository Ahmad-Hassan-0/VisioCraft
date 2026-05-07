"""
app/__init__.py
Flask application factory — single point of initialization.
"""
import logging
import time
from flask import Flask, request as flask_request, jsonify
from pathlib import Path
import firebase_admin
from firebase_admin import credentials

import config

BASE_DIR = Path(__file__).resolve().parent.parent
log = logging.getLogger('visiocraft.app')


def create_app() -> Flask:
    """Create and configure the Flask application."""

    # ── Logging first ────────────────────────────────────────────────────────
    from app.logging_config import setup_logging
    setup_logging()
    log.info("Creating Flask application...")

    # ── Create Flask app ─────────────────────────────────────────────────────
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "Client_Side" / "Front_End" / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
    log.info("Flask app created (templates=%s, static=%s)",
             app.template_folder, app.static_folder)

    # ── Ensure directories exist ─────────────────────────────────────────────
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.TEMP_DIR.mkdir(exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)
    log.info("Directories ensured: outputs=%s, temp=%s, models=%s",
             config.OUTPUT_DIR, config.TEMP_DIR, config.MODELS_DIR)

    # ── Firebase ─────────────────────────────────────────────────────────────
    log.info("=" * 50)
    log.info("INITIALIZING VISIOCRAFT SERVER")
    log.info("=" * 50)

    key_path = BASE_DIR / "serviceAccountKey.json"
    if not firebase_admin._apps:
        if not key_path.exists():
            log.critical("serviceAccountKey.json NOT FOUND at %s", key_path)
            raise FileNotFoundError(f"serviceAccountKey.json not found at {key_path}")
        firebase_admin.initialize_app(credentials.Certificate(str(key_path)))
    log.info("Firebase initialized OK")

    # ── Create services (Dependency Injection) ───────────────────────────────
    log.info("Creating services...")

    from app.services.session_service import SessionService
    from app.services.segmentation_service import SegmentationService
    from app.services.extraction_service import ExtractionService
    from app.services.inpainting_service import InpaintingService
    from app.services.composition_service import CompositionService
    from app.services.generation_service import GenerationService

    session_svc = SessionService()
    log.info("  SessionService created")

    segmentation_svc = SegmentationService()
    log.info("  SegmentationService created (SAM=%s)", segmentation_svc.sam_available)

    extraction_svc = ExtractionService()
    log.info("  ExtractionService created")

    inpainting_svc = InpaintingService()
    log.info("  InpaintingService created (available=%s)", inpainting_svc.available)

    composition_svc = CompositionService()
    log.info("  CompositionService created")

    generation_svc = GenerationService()
    log.info("  GenerationService created")

    log.info("All services initialized successfully")

    # ── Inject services into app config ──────────────────────────────────────
    app.config['SESSION_SVC'] = session_svc
    app.config['SEGMENTATION_SVC'] = segmentation_svc
    app.config['EXTRACTION_SVC'] = extraction_svc
    app.config['INPAINTING_SVC'] = inpainting_svc
    app.config['COMPOSITION_SVC'] = composition_svc
    app.config['GENERATION_SVC'] = generation_svc

    # ── Register blueprints ──────────────────────────────────────────────────
    from app.routes.page_routes import page_bp
    from app.routes.auth_routes import auth_bp
    from app.routes.session_routes import session_bp
    from app.routes.upload_routes import upload_bp
    from app.routes.segmentation_routes import segmentation_bp
    from app.routes.extraction_routes import extraction_bp
    from app.routes.inpainting_routes import inpainting_bp
    from app.routes.composition_routes import composition_bp
    from app.routes.generation_routes import generation_bp
    from app.routes.health_routes import health_bp

    app.register_blueprint(page_bp)
    app.register_blueprint(auth_bp, url_prefix='/api')
    app.register_blueprint(session_bp, url_prefix='/api')
    app.register_blueprint(upload_bp, url_prefix='/api')
    app.register_blueprint(segmentation_bp, url_prefix='/api')
    app.register_blueprint(extraction_bp, url_prefix='/api')
    app.register_blueprint(inpainting_bp, url_prefix='/api')
    app.register_blueprint(composition_bp, url_prefix='/api')
    app.register_blueprint(generation_bp, url_prefix='/api')
    app.register_blueprint(health_bp, url_prefix='/api')

    log.info("All blueprints registered (10 total)")

    # ── Request/response logging middleware ───────────────────────────────────
    req_log = logging.getLogger('visiocraft.http')

    @app.before_request
    def log_request():
        flask_request._start_time = time.time()
        if not flask_request.path.startswith('/static'):
            req_log.info(">>> %s %s  (from %s)",
                         flask_request.method, flask_request.path,
                         flask_request.remote_addr)
            if flask_request.content_type and 'json' in flask_request.content_type:
                try:
                    body = flask_request.get_json(silent=True)
                    if body:
                        # Truncate large payloads (e.g. point arrays)
                        summary = str(body)[:300]
                        req_log.debug("    Body: %s", summary)
                except Exception:
                    pass

    @app.after_request
    def log_response(response):
        if not flask_request.path.startswith('/static'):
            elapsed = time.time() - getattr(flask_request, '_start_time', time.time())
            req_log.info("<<< %s %s → %d  (%.0fms)",
                         flask_request.method, flask_request.path,
                         response.status_code, elapsed * 1000)
        return response

    # ── Error handlers ───────────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        log.warning("404 Not Found: %s %s", flask_request.method, flask_request.path)
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def server_error(e):
        log.error("500 Server Error on %s %s: %s",
                  flask_request.method, flask_request.path, e)
        return jsonify({'error': 'Server error', 'detail': str(e)}), 500

    log.info("=" * 50)
    log.info("VisioCraft server ready")
    log.info("=" * 50)

    return app
