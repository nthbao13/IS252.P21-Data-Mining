import pandas as pd
import os
import logging
from flask import render_template, request, session
from werkzeug.utils import secure_filename
from algorithm.kohonen import run_kohonen

logger = logging.getLogger(__name__)

def handle_file_upload(request, upload_folder):
    """Handle file upload and return DataFrame"""
    if 'file' not in request.files:
        return None, "No file part in the request", None
    file = request.files['file']
    if file.filename == '':
        return None, "No file selected", None
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        try:
            df = pd.read_csv(filepath)
            return df, None, filepath
        except Exception as e:
            return None, f"Error reading CSV: {str(e)}", filepath
    return None, "Invalid file format. Please upload a CSV file", None

def register_routes(app):
    @app.route('/kohonen', methods=['GET', 'POST'])
    def kohonen_page():
        df = None
        error = None
        columns = []
        
        if 'df' in session:
            try:
                df = pd.read_json(session['df'])
                columns = df.columns.tolist()
            except Exception as e:
                error = f"Lỗi khi đọc dữ liệu: {str(e)}"
                session.pop('df', None)
        
        if request.method == 'POST':
            action = request.form.get('action', '')
            
            if action == 'upload' or 'file' in request.files:
                uploaded_df, error, filepath = handle_file_upload(request, app.config['UPLOAD_FOLDER'])
                
                if uploaded_df is None:
                    return render_template('kohonen.html', df=df, columns=columns, error=error)
                
                try:
                    session['df'] = uploaded_df.to_json()
                    session.pop('num_clusters', None)
                    session.pop('map_height', None)
                    session.pop('map_width', None)
                    session.pop('learning_rate', None)
                    session.pop('radius', None)
                    session.pop('iterations', None)
                    df = uploaded_df
                    columns = uploaded_df.columns.tolist()
                    
                    if filepath and os.path.exists(filepath):
                        os.remove(filepath)
                        logger.debug(f"Deleted file: {filepath}")
                    
                    return render_template('kohonen.html', df=df, columns=columns)
                except Exception as e:
                    error = f"Lỗi khi xử lý file: {str(e)}"
                    if filepath and os.path.exists(filepath):
                        os.remove(filepath)
                    return render_template('kohonen.html', df=df, columns=columns, error=error)
            
            elif action == 'train_som':
                if 'df' not in session:
                    return render_template('kohonen.html', error="Vui lòng tải dữ liệu trước khi huấn luyện")
                
                try:
                    num_clusters = int(request.form.get('num_clusters', 3))
                    map_height = request.form.get('map_height')
                    map_width = request.form.get('map_width')
                    map_height = int(map_height) if map_height else None
                    map_width = int(map_width) if map_width else None
                    learning_rate = float(request.form.get('learning_rate', 0.4))
                    radius = float(request.form.get('radius', 0.0))
                    iterations = int(request.form.get('iterations', 5))
                    
                    if num_clusters < 1:
                        raise ValueError("Number of clusters must be positive")
                    if learning_rate <= 0 or learning_rate > 1:
                        raise ValueError("Learning rate must be between 0 and 1")
                    if radius < 0:
                        raise ValueError("Neighborhood radius cannot be negative")
                    if iterations < 1:
                        raise ValueError("Number of iterations must be positive")
                    if map_height and map_width and (map_height < 1 or map_width < 1):
                        raise ValueError("Map dimensions must be positive")
                    
                    session['num_clusters'] = num_clusters
                    session['map_height'] = map_height
                    session['map_width'] = map_width
                    session['learning_rate'] = learning_rate
                    session['radius'] = radius
                    session['iterations'] = iterations
                    
                    df = pd.read_json(session['df'])
                    result = run_kohonen(df, num_clusters, map_height, map_width, learning_rate, radius, iterations)
                    
                    if not result['success']:
                        return render_template('kohonen.html', df=df, columns=columns, error=result['error'])
                    
                    return render_template('kohonen.html', df=df, columns=columns, result=result)
                except ValueError as e:
                    return render_template('kohonen.html', df=df, columns=columns, error=str(e))
                except Exception as e:
                    logger.error(f"Error in SOM training: {str(e)}")
                    return render_template('kohonen.html', df=df, columns=columns, error=f"Lỗi khi huấn luyện SOM: {str(e)}")
        
        return render_template('kohonen.html', df=df, columns=columns, error=error)