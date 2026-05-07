"""
Server_Side/auth_routes.py
Responsible ONLY for Firebase authentication endpoints.
"""
from flask import Blueprint, request, jsonify, current_app
from firebase_admin import auth

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/verify-token', methods=['POST'])
def verify_token():
    data = request.json
    id_token = data.get('idToken')
    if not id_token:
        return jsonify({"status": "error", "message": "No token provided"}), 400
    try:
        decoded = auth.verify_id_token(id_token)
        uid   = decoded['uid']
        email = decoded.get('email', 'unknown')
        print(f"✅ Verified: {email} (UID: {uid})")
        return jsonify({"status": "success", "uid": uid, "email": email}), 200
    except Exception as e:
        print(f"❌ Auth failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 401