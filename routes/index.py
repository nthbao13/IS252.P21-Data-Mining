from flask import render_template, session
from utils.decorators import cleanup_session

def register_routes(app):
    @app.route('/')
    @cleanup_session
    def index():
        return render_template('index.html')