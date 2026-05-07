"""
app/routes/page_routes.py
HTML page rendering — presentation only.
"""
from flask import Blueprint, render_template

page_bp = Blueprint('pages', __name__)


@page_bp.route('/')
def index():
    return render_template('signup.html')


@page_bp.route('/login')
def login_page():
    return render_template('login.html')


@page_bp.route('/home')
@page_bp.route('/Home')
def home_page():
    return render_template('home.html', active_page='home', show_logout=True)


@page_bp.route('/forget-password')
def forget_password_page():
    return render_template('forget_password.html')


@page_bp.route('/canvas')
def canvas_page():
    return render_template('canvas.html', active_page='canvas', show_logout=True)


@page_bp.route('/generate')
def generate_page():
    return render_template('generate.html', active_page='generate', show_logout=True)


@page_bp.route('/about')
def about_page():
    return render_template('about_us.html', active_page='about', show_logout=False)


@page_bp.route('/contact')
def contact_page():
    return render_template('contact_us.html', active_page='contact', show_logout=False)
