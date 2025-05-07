from flask import render_template, request
from utils import handle_file_upload
from algorithm.clustering_kmeans import run_kmeans
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def register_routes(app):
    @app.route('/clustering', methods=['GET', 'POST'])
    def clustering_page():
        logger.debug("Accessing clustering route")
        
        if request.method == 'POST':
            logger.debug(f"POST request received: {request.form.keys()}")
            
            # Lấy tham số từ form
            try:
                K = int(request.form.get('K', 3))
                if K <= 0:
                    raise ValueError("Số cụm phải lớn hơn 0")
            except ValueError as e:
                logger.error(f"Invalid K value: {str(e)}")
                return render_template('clustering.html', error=str(e))
            
            use_sample_data = request.form.get('use_sample_data') == 'true'
            
            # Xử lý file upload
            df = None
            filepath = None
            if not use_sample_data:
                if 'file' not in request.files:
                    logger.error("No file part in request")
                    return render_template('clustering.html', error="Không có file được tải lên")
                
                result = handle_file_upload(request, app.config['UPLOAD_FOLDER'])
                df, filename, error = result if len(result) == 3 else (None, None, result[1])
                if df is None:
                    logger.error(f"File upload failed: {error}")
                    return render_template('clustering.html', error=error)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Chạy K-Means
            try:
                result = run_kmeans(df, K=K, use_sample_data=use_sample_data)
                if result.get('error'):
                    logger.error(f"K-Means error: {result['error']}")
                    return render_template('clustering.html', error=result['error'])
                # Kiểm tra result trước khi truyền vào template
                logger.debug(f"Result keys: {list(result.keys())}")
                if not isinstance(result.get('data'), pd.DataFrame):
                    logger.error("Result 'data' is not a DataFrame")
                    return render_template('clustering.html', error="Dữ liệu kết quả không hợp lệ")
                logger.debug("K-Means completed successfully")
                return render_template('clustering.html', result=result)
            except Exception as e:
                logger.error(f"Error running K-Means: {str(e)}")
                return render_template('clustering.html', error=f"Lỗi xử lý: {str(e)}")
            finally:
                if filepath and os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        logger.debug(f"Deleted file: {filepath}")
                    except Exception as e:
                        logger.error(f"Error deleting file {filepath}: {str(e)}")
        
        logger.debug("Rendering clustering page")
        return render_template('clustering.html')