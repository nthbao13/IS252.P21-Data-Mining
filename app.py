from flask import Flask
import os
import logging
from routes import register_all_routes
from algorithm.preprocess import preprocess_page, download_file

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define directories
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key'

# App configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Custom template filter for list difference
@app.template_filter('list_difference')
def list_difference(list1, list2):
    return list(set(list1) - set(list2))

# Register routes
register_all_routes(app)

# Preprocess routes
app.route('/preprocess', methods=['GET', 'POST'])(preprocess_page)
app.route('/download/<filename>')(download_file)

# Clean up old files on startup
@app.before_request
def setup_cleanup():
    if not getattr(setup_cleanup, 'has_run', False):
        for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            logger.debug(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting {file_path}: {e}")
        setup_cleanup.has_run = True

if __name__ == '__main__':
    app.run(debug=True)