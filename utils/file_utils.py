import os
import json
import uuid
from werkzeug.utils import secure_filename
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def handle_file_upload(request, upload_folder):
    logger.debug("Starting handle_file_upload")
    
    if 'file' not in request.files:
        logger.error("No file part in request")
        return None, None, "Không có file được tải lên"
    
    file = request.files['file']
    if not file or file.filename == '':
        logger.error("No file selected")
        return None, None, "Chưa chọn file"
    
    if not file.filename.endswith('.csv'):
        logger.error(f"Invalid file format: {file.filename}")
        return None, None, "Chỉ hỗ trợ file CSV"
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    
    try:
        file.save(filepath)
        logger.debug(f"File saved: {filepath}")
    except Exception as e:
        logger.error(f"Error saving file {filepath}: {str(e)}")
        return None, None, f"Lỗi lưu file: {str(e)}"
    
    try:
        df = pd.read_csv(filepath, header=0)  # Giả định luôn có header
        logger.debug(f"File read successfully: {filepath}")
        return df, filename, filepath
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {str(e)}")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.debug(f"Deleted invalid file: {filepath}")
            except Exception as ex:
                logger.error(f"Error deleting invalid file {filepath}: {str(ex)}")
        return None, None, f"Lỗi đọc file: {str(e)}"

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