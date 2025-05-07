from flask import render_template, request
from utils.file_utils import handle_file_upload
from algorithm.association_rules import run_association_rules
import os
import logging

logger = logging.getLogger(__name__)

def register_routes(app):
    @app.route('/association-rules', methods=['GET', 'POST'])
    def association_rules_page():
        if request.method == 'POST':
            df, filename, filepath = handle_file_upload(request, app.config['UPLOAD_FOLDER'])
            if df is None:
                return render_template('association_rules.html', error=filename)
            
            try:
                minsup = float(request.form.get('minsup', 0.4))
                minconf = float(request.form.get('minconf', 0.4))
                if not (0 <= minsup <= 1 and 0 <= minconf <= 1):
                    raise ValueError("minsup và minconfidence phải từ 0 đến 1")
                result = run_association_rules(df, minsup, minconf)
                if 'error' in result:
                    return render_template('association_rules.html', error=result['error'])
                return render_template('association_rules.html', result=result)
            except ValueError as e:
                return render_template('association_rules.html', error=str(e))
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug(f"Deleted file: {filepath}")
        
        return render_template('association_rules.html')