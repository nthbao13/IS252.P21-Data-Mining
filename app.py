from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import os
import json
import uuid
import shutil
from werkzeug.utils import secure_filename
import logging
from datetime import timedelta
from algorithm.reduct import run_reduct, get_decision_values
from algorithm.preprocess import preprocess_page, download_file
from algorithm.association_rules import run_association_rules
from algorithm.decision_tree_id3 import run_decision_tree, predict

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Định nghĩa các thư mục
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'

# Tạo thư mục nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Required for session management
app.permanent_session_lifetime = timedelta(minutes=30)  # Set session timeout to 30 minutes

# Cấu hình ứng dụng
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Các hàm tiện ích cho lưu trữ kết quả
def save_result_to_file(result):
    """Lưu kết quả vào file và trả về tên file"""
    result_id = str(uuid.uuid4())
    result_path = os.path.join(app.config['RESULT_FOLDER'], f"{result_id}.json")
    
    # Lưu kết quả vào file JSON
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    
    return result_id

def load_result_from_file(result_id):
    """Nạp kết quả từ file"""
    if not result_id:
        return None
    
    result_path = os.path.join(app.config['RESULT_FOLDER'], f"{result_id}.json")
    if not os.path.exists(result_path):
        return None
    
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def delete_result_file(result_id):
    """Xóa file kết quả"""
    if not result_id:
        return False
    
    result_path = os.path.join(app.config['RESULT_FOLDER'], f"{result_id}.json")
    if os.path.exists(result_path):
        os.remove(result_path)
        return True
    return False

# Custom template filter for list difference
@app.template_filter('list_difference')
def list_difference(list1, list2):
    """Filter to get difference between two lists."""
    return list(set(list1) - set(list2))

# Trang chủ
@app.route('/')
def index():
    logger.debug("Accessing index route, clearing session")
    # Clear session data when returning to the home page
    if 'filename' in session:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['filename'])
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.debug(f"Deleted file: {filepath}")
    # Xóa file kết quả nếu có
    if 'result_id' in session:
        delete_result_file(session['result_id'])
        logger.debug(f"Deleted result file: {session.get('result_id')}")
    
    session.clear()
    return render_template('index.html')

# Route cho preprocessing
app.route('/preprocess', methods=['GET', 'POST'])(preprocess_page)
app.route('/download/<filename>')(download_file)

# Route cho association rules
@app.route('/association-rules', methods=['GET', 'POST'])
def association_rules_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('association_rules.html', error='Không có file được tải lên')

        file = request.files['file']
        if not file or file.filename == '':
            return render_template('association_rules.html', error='Chưa chọn file')

        if not file.filename.endswith('.csv'):
            return render_template('association_rules.html', error='Chỉ hỗ trợ file CSV')

        try:
            minsup = float(request.form.get('minsup', 0.4))
            minconf = float(request.form.get('minconf', 0.4))
            if not (0 <= minsup <= 1 and 0 <= minconf <= 1):
                raise ValueError("minsup và minconfidence phải từ 0 đến 1")
        except ValueError as e:
            return render_template('association_rules.html', error=str(e))

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            result = run_association_rules(df, minsup, minconf)
            if 'error' in result:
                return render_template('association_rules.html', error=result['error'])
            return render_template('association_rules.html', result=result)
        except Exception as e:
            return render_template('association_rules.html', error=f"Lỗi xử lý file: {str(e)}")
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"Deleted file: {filepath}")

    return render_template('association_rules.html')

# Route cho decision tree
@app.route('/decision-tree', methods=['GET', 'POST'])
def decision_tree_page():
    session.permanent = True

    # Xử lý Reset GET: /decision-tree?reset=true
    if request.method == 'GET' and request.args.get('reset') == 'true':
        # Xóa file dữ liệu
        if 'filename' in session:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['filename'])
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"Deleted file on reset request: {filepath}")
        
        # Xóa file kết quả
        if 'result_id' in session:
            delete_result_file(session['result_id'])
            logger.debug(f"Deleted result file: {session['result_id']}")
        
        session.clear()
        logger.debug("Session cleared on reset request")
        return redirect(url_for('decision_tree_page'))

    if request.method == 'POST':
        # Người dùng vừa upload file
        if 'file' in request.files and 'target_col' not in request.form:
            file = request.files['file']
            if not file or file.filename == '':
                return render_template('decision_tree.html', error='Chưa chọn file')

            if not file.filename.endswith('.csv'):
                return render_template('decision_tree.html', error='Chỉ hỗ trợ file CSV')

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Nếu đã có file cũ thì xóa
            if 'filename' in session:
                old_filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['filename'])
                if os.path.exists(old_filepath):
                    os.remove(old_filepath)
                    logger.debug(f"Deleted old file: {old_filepath}")
            
            # Nếu đã có kết quả cũ thì xóa
            if 'result_id' in session:
                delete_result_file(session['result_id'])
                session.pop('result_id', None)

            file.save(filepath)
            session['filename'] = filename
            logger.debug(f"File uploaded and saved: {filename}")

            try:
                df = pd.read_csv(filepath)
                columns = df.columns.tolist()
                return render_template('decision_tree.html', columns=columns, filename=filename)
            except Exception as e:
                logger.error(f"Error reading file: {str(e)}")
                return render_template('decision_tree.html', error=f"Lỗi đọc file: {str(e)}")

        # Người dùng chọn cột đích và tiêu chí xây dựng cây (khởi tạo ban đầu)
        elif 'target_col' in request.form and 'predict' not in request.form:
            filename = session.get('filename')
            if not filename:
                return render_template('decision_tree.html', error='Vui lòng tải lên file dữ liệu trước')

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                session.pop('filename', None)
                if 'result_id' in session:
                    delete_result_file(session['result_id'])
                    session.pop('result_id', None)
                return render_template('decision_tree.html', error='File không tồn tại, vui lòng upload lại')

            df = pd.read_csv(filepath)
            target_col = request.form.get('target_col')
            criterion = request.form.get('criterion', 'information_gain')

            result = run_decision_tree(df, target_col, criterion)
            if 'error' in result:
                logger.error(f"Error building tree: {result['error']}")
                return render_template('decision_tree.html', error=result['error'])
            
            # Lưu kết quả vào file và lưu ID trong session
            result_id = save_result_to_file(result)
            session['result_id'] = result_id
            logger.debug(f"Decision tree built and stored in file: {result_id}")
            
            return render_template('decision_tree.html', result=result, filename=filename, criterion=criterion)
        
        # Người dùng thực hiện dự đoán (predict)
        elif 'predict' in request.form:
            result_id = session.get('result_id')
            if not result_id:
                return render_template('decision_tree.html', error='Cây quyết định không tồn tại, vui lòng xây dựng lại.')
            
            # Nạp kết quả từ file
            result = load_result_from_file(result_id)
            if not result:
                session.pop('result_id', None)
                return render_template('decision_tree.html', error='Không tìm thấy dữ liệu cây quyết định, vui lòng xây dựng lại.')

            # Lấy giá trị các thuộc tính từ form
            instance = {feature: request.form.get(feature) for feature in result['features']}
            if None in instance.values() or '' in instance.values():
                return render_template('decision_tree.html', result=result, filename=session.get('filename'), 
                                     error='Vui lòng chọn giá trị cho tất cả các thuộc tính.')

            prediction = predict(result['tree'], instance)
            logger.debug(f"Prediction result: {prediction}")
            return render_template('decision_tree.html', result=result, prediction=prediction, 
                                 filename=session.get('filename'), criterion=result.get('criterion', 'information_gain'))

    # Request GET thông thường (không reset)
    if 'result_id' in session:
        result = load_result_from_file(session['result_id'])
        if result:
            logger.debug("Loaded result from file")
            return render_template('decision_tree.html', result=result, filename=session.get('filename'), 
                                 criterion=result.get('criterion', 'information_gain'))
        else:
            logger.warning(f"Result file not found: {session['result_id']}")
            session.pop('result_id', None)

    return render_template('decision_tree.html')

@app.route('/reduct', methods=['GET', 'POST'])
def reduct_page():
    session.permanent = True  # Ensure session persists
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('reduct.html', error='Không có file được tải lên')

        file = request.files['file']
        if not file or file.filename == '':
            return render_template('reduct.html', error='Chưa chọn file')

        if not file.filename.endswith('.csv'):
            return render_template('reduct.html', error='Chỉ hỗ trợ file CSV')

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        session['reduct_file_path'] = filepath  # Store in session
        logger.debug(f"File saved and session set: {filepath}")

        try:
            df = pd.read_csv(filepath)
            return render_template('select_attributes.html', columns=df.columns.tolist(), filename=file.filename)
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            return render_template('reduct.html', error=f"Lỗi đọc file: {str(e)}")

    return render_template('reduct.html')

@app.route('/reduct/analyze', methods=['POST'])
def reduct_analyze():
    file_path = session.get('reduct_file_path')

    if not file_path or not os.path.exists(file_path):
        return render_template('reduct.html', error='Vui lòng tải lên file dữ liệu trước')

    try:
        # Get form data
        decision_col = request.form.get('decision_col')
        decision_value = request.form.get('decision_value')
        index_col = request.form.get('index_col', None)

        # Get attributes (all columns except decision and index)
        all_columns = request.form.getlist('attributes')

        if not all_columns:
            return render_template('reduct.html', error='Vui lòng chọn ít nhất một thuộc tính')

        if not decision_col:
            return render_template('reduct.html', error='Vui lòng chọn cột quyết định')

        # Run the analysis
        result = run_reduct(file_path, all_columns, decision_col, decision_value, index_col)

        if 'error' in result:
            return render_template('reduct.html', error=result['error'])

        return render_template('results.html', **result)
    except Exception as e:
        logger.error(f"Error in reduct analysis: {str(e)}")
        return render_template('reduct.html', error=f"Lỗi phân tích: {str(e)}")
    finally:
        # Clean up the file after analysis
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Deleted file: {file_path}")
        if 'reduct_file_path' in session:
            session.pop('reduct_file_path')

# Route to get decision values dynamically
@app.route('/get_decision_values', methods=['POST'])
def get_decision_values_route():
    filename = session.get('reduct_file_path')
    decision_col = request.form.get('decision_col')

    logger.debug(f"Fetching decision values for file: {filename}, column: {decision_col}")
    if filename and decision_col and os.path.exists(filename):
        values = get_decision_values(filename, decision_col)
        logger.debug(f"Returning values: {values}")
        return {'values': values}

    logger.warning("No filename or decision column provided")
    return {'values': []}

# Thêm handler dọn dẹp khi ứng dụng khởi động
@app.before_request
def setup_cleanup():
    """Thiết lập dọn dẹp trước mỗi request"""
    # Xóa các file cũ trong thư mục uploads và results (chỉ chạy một lần khi khởi động)
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