from flask import render_template, request, session, jsonify
from utils import handle_file_upload
from algorithm.kmeans import run_kmeans
import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def register_routes(app):
    @app.route('/kmeans', methods=['GET', 'POST'])
    def kmeans_page():
        logger.debug("Accessing clustering route")
        dimensions = session.get('dimensions', 0)
        
        if request.method == 'POST':
            logger.debug(f"POST request received: {request.form.keys()}")
            
            # Lấy tham số từ form
            try:
                K = int(request.form.get('K', 3))
                if K <= 0:
                    raise ValueError("Số cụm phải lớn hơn 0")
            except ValueError as e:
                logger.error(f"Invalid K value: {str(e)}")
                return render_template('kmeans.html', error=str(e), dimensions=dimensions)
            
            use_sample_data = request.form.get('use_sample_data') == 'true'
            
            # Xử lý file upload
            df = None
            filepath = None
            if not use_sample_data:
                if 'file' not in request.files or not request.files['file'].filename:
                    logger.error("No file part in request")
                    return render_template('kmeans.html', error="Không có file được tải lên", dimensions=dimensions)
                
                result = handle_file_upload(request, app.config['UPLOAD_FOLDER'])
                df, filename, error = result if len(result) == 3 else (None, None, result[1])
                if df is None:
                    logger.error(f"File upload failed: {error}")
                    return render_template('kmeans.html', error=error, dimensions=dimensions)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Lấy số chiều
            if use_sample_data:
                dimensions = 3  # Dữ liệu mẫu là 3D
            elif df is not None:
                dimensions = len(df.select_dtypes(include=[np.number]).columns)
            
            # Lưu vào session
            if df is not None:
                session['df'] = df.to_json()
            session['dimensions'] = dimensions
            
            # Xử lý centroid khởi tạo từ bảng
            initial_centroids = None
            if any(f'centroid_0_{j}' in request.form for j in range(dimensions)):
                try:
                    centroid_list = []
                    for i in range(K):
                        row = [float(request.form.get(f'centroid_{i}_{j}', 0)) for j in range(dimensions)]
                        centroid_list.append(row)
                    initial_centroids = np.array(centroid_list)
                except Exception as e:
                    logger.error(f"Invalid centroid input: {str(e)}")
                    return render_template('kmeans.html', error=f"Dữ liệu centroid không hợp lệ: {str(e)}", dimensions=dimensions)
            
            # Chạy K-Means
            try:
                result = run_kmeans(df, K=K, use_sample_data=use_sample_data, initial_centroids=initial_centroids)
                if result.get('error'):
                    logger.error(f"K-Means error: {result['error']}")
                    return render_template('kmeans.html', error=result['error'], dimensions=dimensions)
                logger.debug(f"Result keys: {list(result.keys())}")
                if not isinstance(result.get('data'), pd.DataFrame):
                    logger.error("Result 'data' is not a DataFrame")
                    return render_template('kmeans.html', error="Dữ liệu kết quả không hợp lệ", dimensions=dimensions)
                logger.debug("K-Means completed successfully")
                return render_template('kmeans.html', result=result, dimensions=dimensions)
            except Exception as e:
                logger.error(f"Error running K-Means: {str(e)}")
                return render_template('kmeans.html', error=f"Lỗi xử lý: {str(e)}", dimensions=dimensions)
            finally:
                if filepath and os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        logger.debug(f"Deleted file: {filepath}")
                    except Exception as e:
                        logger.error(f"Error deleting file {filepath}: {str(e)}")
        
        logger.debug("Rendering clustering page")
        return render_template('kmeans.html', dimensions=dimensions)
    
    @app.route('/get_dimensions', methods=['POST'])
    def get_dimensions():
        """Trả về số chiều của file CSV"""
        try:
            result = handle_file_upload(request, app.config['UPLOAD_FOLDER'])
            df, filename, error = result if len(result) == 3 else (None, None, result[1])
            if df is None:
                logger.error(f"File upload failed: {error}")
                return jsonify({'error': error})
            dimensions = len(df.select_dtypes(include=[np.number]).columns)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'dimensions': dimensions})
        except Exception as e:
            logger.error(f"Error getting dimensions: {str(e)}")
            return jsonify({'error': str(e)})