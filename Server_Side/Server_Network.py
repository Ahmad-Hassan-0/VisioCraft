"""
Server_Side/Server_Network.py
App entry point: initializes Flask, registers blueprints, starts components.
"""
from flask import Flask
from pathlib import Path
import firebase_admin
from firebase_admin import credentials
import config
from Client_Side.SessionManager import SessionManager
from Server_Side.Image_Processing.Image_Processing_Manager import ImageProcessingManager
from Server_Side.Object_Extraction.Object_Masking import ObjectMasking

BASE_DIR = Path(__file__).resolve().parent.parent

app = Flask(__name__,
    template_folder=str(BASE_DIR / "Client_Side" / "Front_End" / "templates"),
    static_folder=str(BASE_DIR / "static"))
app.config['MAX_CONTENT_LENGTH'] = getattr(config, 'MAX_CONTENT_LENGTH', 16 * 1024 * 1024)

session_manager = None # Shared global instances 
processing_manager = None
object_masker = None

def init_server():
    global session_manager, processing_manager, object_masker
    print("\n" + "="*50)
    print("🚀 INITIALIZING VISIOCRAFT SERVER")
    print("="*50)

    key_path = BASE_DIR / "serviceAccountKey.json" # Firebase
    if not firebase_admin._apps:
        if not key_path.exists():
            raise FileNotFoundError(f"serviceAccountKey.json not found at {key_path}")
        firebase_admin.initialize_app(credentials.Certificate(str(key_path)))
    print("✅ Firebase initialized")

    session_manager   = SessionManager() # Managers
    processing_manager = ImageProcessingManager()
    object_masker     = ObjectMasking()
    print("✅ All managers initialized\n")

    app.config['SESSION_MGR']    = session_manager # Inject into app config so blueprints can access
    app.config['PROCESSING_MGR'] = processing_manager
    app.config['OBJECT_MASKER']  = object_masker

    from Server_Side.page_routes  import page_bp  # Register blueprints
    from Server_Side.auth_routes  import auth_bp
    from Server_Side.image_routes import image_bp
    app.register_blueprint(page_bp)
    app.register_blueprint(auth_bp,   url_prefix='/api')
    app.register_blueprint(image_bp,  url_prefix='/api')

    print("✅ Blueprints registered")
    print("="*50 + "\n")
    
    @app.errorhandler(404) # Error handlers
    def not_found(e):
        from flask import jsonify
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def server_error(e):
        from flask import jsonify
        return jsonify({'error': 'Server error'}), 500