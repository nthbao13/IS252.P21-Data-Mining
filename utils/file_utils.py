import os
import json
import uuid
from werkzeug.utils import secure_filename
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from functools import wraps
from flask import request, session, current_app
import os
import logging

logger = logging.getLogger(__name__)

def handle_file_upload(request, upload_folder, allow_missing=False):
    """Handle file upload and return DataFrame, filename, and filepath or error"""
    if 'file' not in request.files:
        logger.error("No file part in request")
        return None, None, "Không có file được tải lên"
    
    file = request.files['file']
    if file.filename == '':
        if allow_missing:
            logger.debug("No file selected, proceeding with allow_missing")
            return None, None, None  # Return None for all values to indicate no file
        logger.error("No file selected")
        return None, None, "Vui lòng chọn một file để tải lên"
    
    if file and file.filename.endswith('.csv'):
        filename = file.filename
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        logger.debug(f"Saved file to {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            return df, filename, filepath
        except Exception as e:
            logger.error(f"Error reading CSV file {filepath}: {str(e)}")
            return None, filename, f"Lỗi đọc file CSV: {str(e)}"
    else:
        return None, file.filename, "File phải có định dạng .csv"

def save_result_to_file(result, result_folder):
    result_id = str(uuid.uuid4())
    result_path = os.path.join(result_folder, f"{result_id}.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    return result_id

def load_result_from_file(result_id, result_folder):
    if not result_id:
        return None
    result_path = os.path.join(result_folder, f"{result_id}.json")
    if not os.path.exists(result_path):
        return None
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def delete_result_file(result_id, result_folder):
    if not result_id:
        return False
    result_path = os.path.join(result_folder, f"{result_id}.json")
    if os.path.exists(result_path):
        os.remove(result_path)
        return True
    return False

def get_decision_values(filename, decision_col):
    try:
        df = pd.read_csv(filename)
        return df[decision_col].unique().tolist()
    except Exception as e:
        logger.error(f"Error getting decision values: {str(e)}")
        return []