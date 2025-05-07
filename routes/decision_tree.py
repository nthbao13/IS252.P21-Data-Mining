from flask import render_template, request, redirect, url_for, session
from utils import handle_file_upload, save_result_to_file, load_result_from_file, delete_result_file
from algorithm.decision_tree_id3 import run_decision_tree, predict
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def handle_decision_tree_upload(request, session, upload_folder, result_folder):
    if 'file' not in request.files:
        logger.error("No file part in request")
        return None, "Không có file được tải lên"
    
    file = request.files['file']
    if not file or file.filename == '':
        logger.error("No file selected")
        return None, "Chưa chọn file"
    
    # Clean up old file and result
    if 'filename' in session:
        old_filepath = os.path.join(upload_folder, session['filename'])
        try:
            if os.path.exists(old_filepath):
                os.remove(old_filepath)
                logger.debug(f"Deleted old file: {old_filepath}")
        except Exception as e:
            logger.error(f"Error deleting old file {old_filepath}: {str(e)}")
    
    if 'result_id' in session:
        try:
            delete_result_file(session['result_id'], result_folder)
            logger.debug(f"Deleted old result file: {session['result_id']}")
        except Exception as e:
            logger.error(f"Error deleting result file {session['result_id']}: {str(e)}")
        session.pop('result_id', None)
    
    # Handle new file upload
    df, filename, filepath = handle_file_upload(request, upload_folder)
    if df is None:
        logger.error(f"File upload failed: {filename}")
        return None, filename  # filename contains error message
    
    session['filename'] = filename
    logger.debug(f"New file uploaded: {filename}")
    return df, None

def handle_decision_tree_build(request, session, upload_folder, result_folder):
    filename = session.get('filename')
    if not filename:
        logger.warning("No filename in session")
        return None, "Vui lòng tải lên file dữ liệu trước"
    
    filepath = os.path.join(upload_folder, filename)
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        session.pop('filename', None)
        session.pop('result_id', None)
        return None, "File không tồn tại, vui lòng upload lại"
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {str(e)}")
        return None, f"Lỗi đọc file: {str(e)}"
    
    target_col = request.form.get('target_col')
    if not target_col:
        logger.error("No target column selected")
        return None, "Vui lòng chọn cột đích"
    
    criterion = request.form.get('criterion', 'information_gain')
    result = run_decision_tree(df, target_col, criterion)
    if 'error' in result:
        logger.error(f"Error building tree: {result['error']}")
        return None, result['error']
    
    result_id = save_result_to_file(result, result_folder)
    session['result_id'] = result_id
    logger.debug(f"Decision tree built, result saved: {result_id}")
    return result, None

def handle_decision_tree_predict(request, session, result_folder):
    result_id = session.get('result_id')
    if not result_id:
        logger.warning("No result_id in session")
        return None, "Cây quyết định không tồn tại, vui lòng xây dựng lại"
    
    result = load_result_from_file(result_id, result_folder)
    if not result:
        logger.warning(f"Result file not found: {result_id}")
        session.pop('result_id', None)
        return None, "Không tìm thấy dữ liệu cây quyết định, vui lòng xây dựng lại"
    
    instance = {feature: request.form.get(feature) for feature in result['features']}
    if None in instance.values() or '' in instance.values():
        logger.error("Missing values in prediction instance")
        return None, "Vui lòng chọn giá trị cho tất cả các thuộc tính"
    
    try:
        prediction = predict(result['tree'], instance)
        logger.debug(f"Prediction made: {prediction}")
        return {
            'prediction': prediction,
            'result': result,
            'criterion': result.get('criterion', 'information_gain')
        }, None
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None, f"Lỗi dự đoán: {str(e)}"

def register_routes(app):
    @app.route('/decision-tree', methods=['GET', 'POST'])
    def decision_tree_page():
        session.permanent = True
        
        # Handle reset
        if request.method == 'GET' and request.args.get('reset') == 'true':
            logger.debug("Reset requested")
            if 'filename' in session:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['filename'])
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        logger.debug(f"Deleted file on reset: {filepath}")
                except Exception as e:
                    logger.error(f"Error deleting file on reset {filepath}: {str(e)}")
            
            if 'result_id' in session:
                try:
                    delete_result_file(session['result_id'], app.config['RESULT_FOLDER'])
                    logger.debug(f"Deleted result file on reset: {session['result_id']}")
                except Exception as e:
                    logger.error(f"Error deleting result file on reset: {str(e)}")
            
            session.clear()
            logger.debug("Session cleared")
            return redirect(url_for('decision_tree_page'))
        
        # Handle POST requests
        if request.method == 'POST':
            # File upload
            if 'file' in request.files and 'target_col' not in request.form:
                logger.debug("Handling file upload")
                df, error = handle_decision_tree_upload(request, session, app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER'])
                if error:
                    return render_template('decision_tree.html', error=error)
                return render_template('decision_tree.html', columns=df.columns.tolist(), filename=session['filename'])
            
            # Build decision tree
            elif 'target_col' in request.form and 'predict' not in request.form:
                logger.debug("Building decision tree")
                result, error = handle_decision_tree_build(request, session, app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER'])
                if error:
                    return render_template('decision_tree.html', error=error)
                return render_template('decision_tree.html', result=result, filename=session.get('filename'), criterion=result.get('criterion'))
            
            # Predict
            elif 'predict' in request.form:
                logger.debug("Handling prediction")
                data, error = handle_decision_tree_predict(request, session, app.config['RESULT_FOLDER'])
                if error:
                    return render_template('decision_tree.html', error=error)
                return render_template('decision_tree.html', **data, filename=session.get('filename'))
        
        # Handle GET with existing result
        if 'result_id' in session:
            result = load_result_from_file(session['result_id'], app.config['RESULT_FOLDER'])
            if result:
                logger.debug("Loaded existing result")
                return render_template('decision_tree.html', result=result, filename=session.get('filename'), 
                                     criterion=result.get('criterion', 'information_gain'))
        
        logger.debug("Rendering decision tree page with no data")
        return render_template('decision_tree.html')