from flask import render_template, request, session
from utils.file_utils import handle_file_upload, get_decision_values
from algorithm.reduct import run_reduct
import os
import logging

logger = logging.getLogger(__name__)

def register_routes(app):
    @app.route('/reduct', methods=['GET', 'POST'])
    def reduct_page():
        session.permanent = True
        if request.method == 'POST':
            df, filename, filepath = handle_file_upload(request, app.config['UPLOAD_FOLDER'])
            if df is None:
                return render_template('reduct.html', error=filename)
            
            session['reduct_file_path'] = filepath
            return render_template('select_attributes.html', columns=df.columns.tolist(), filename=filename)
        
        return render_template('reduct.html')

    @app.route('/reduct/analyze', methods=['POST'])
    def reduct_analyze():
        file_path = session.get('reduct_file_path')
        if not file_path or not os.path.exists(file_path):
            return render_template('reduct.html', error='Vui lòng tải lên file dữ liệu trước')
        
        try:
            decision_col = request.form.get('decision_col')
            decision_value = request.form.get('decision_value')
            index_col = request.form.get('index_col', None)
            all_columns = request.form.getlist('attributes')
            
            if not all_columns:
                return render_template('reduct.html', error='Vui lòng chọn ít nhất một thuộc tính')
            if not decision_col:
                return render_template('reduct.html', error='Vui lòng chọn cột quyết định')
            
            result = run_reduct(file_path, all_columns, decision_col, decision_value, index_col)
            if 'error' in result:
                return render_template('reduct.html', error=result['error'])
            
            return render_template('results.html', **result)
        except Exception as e:
            logger.error(f"Error in reduct analysis: {str(e)}")
            return render_template('reduct.html', error=f"Lỗi phân tích: {str(e)}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Deleted file: {file_path}")
            if 'reduct_file_path' in session:
                session.pop('reduct_file_path')

    @app.route('/get_decision_values', methods=['POST'])
    def get_decision_values_route():
        filename = session.get('reduct_file_path')
        decision_col = request.form.get('decision_col')
        
        if filename and decision_col and os.path.exists(filename):
            values = get_decision_values(filename, decision_col)
            return {'values': values}
        
        logger.warning("No filename or decision column provided")
        return {'values': []}