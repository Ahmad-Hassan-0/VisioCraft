"""
app/routes/auth_routes.py
Firebase authentication endpoints.
"""
import logging
from flask import Blueprint, request, jsonify
from firebase_admin import auth

log = logging.getLogger('visiocraft.routes.auth')

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/verify-token', methods=['POST'])
def verify_token():
    data = request.get_json(force=True, silent=True)
    if not data:
        log.error("POST /api/verify-token — no JSON body")
        return jsonify({"status": "error", "message": "No body"}), 400

    id_token = data.get('idToken')
    if not id_token:
        log.warning("verify-token: no idToken in body")
        return jsonify({"status": "error", "message": "No token provided"}), 400

    try:
        decoded = auth.verify_id_token(id_token)
        uid = decoded['uid']
        email = decoded.get('email', 'unknown')
        log.info("AUTH SUCCESS — email=%s, uid=%s", email, uid)
        return jsonify({"status": "success", "uid": uid, "email": email}), 200
    except Exception as e:
        log.error("AUTH FAILED — %s", e)
        return jsonify({"status": "error", "message": str(e)}), 401
