from functools import wraps
from flask import session, current_app
import os
import logging

logger = logging.getLogger(__name__)

def cleanup_session(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'filename' in session:
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], session['filename'])
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"Deleted file: {filepath}")
        
        if 'result_id' in session:
            from .file_utils import delete_result_file
            delete_result_file(session['result_id'], current_app.config['RESULT_FOLDER'])
            logger.debug(f"Deleted result file: {session['result_id']}")
        
        if 'reduct_file_path' in session:
            filepath = session['reduct_file_path']
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"Deleted file: {filepath}")
        
        session.clear()
        return f(*args, **kwargs)
    return decorated_function