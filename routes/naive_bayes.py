import pandas as pd
import os
import logging
from flask import render_template, request, session
from werkzeug.utils import secure_filename
from algorithm.naive_bayes import run_naive_bayes

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
            logger.info(f"Successfully loaded CSV file: {filepath}")
            return df, None, filepath
        except Exception as e:
            logger.error(f"Error reading CSV: {str(e)}")
            return None, f"Error reading CSV: {str(e)}", filepath
    return None, "Invalid file format. Please upload a CSV file", None

def register_routes(app):
    @app.route('/naive-bayes', methods=['GET', 'POST'])
    def naive_bayes_page():
        df = None
        error = None
        columns = []
        
        if 'df' in session:
            try:
                df = pd.read_json(session['df'])
                columns = df.columns.tolist()
                logger.debug("Loaded DataFrame from session")
            except Exception as e:
                error = f"Lỗi khi đọc dữ liệu từ session: {str(e)}"
                session.pop('df', None)
                logger.error(error)
        
        if request.method == 'POST':
            action = request.form.get('action', '')
            
            if action == 'upload' or 'file' in request.files:
                uploaded_df, error, filepath = handle_file_upload(request, app.config['UPLOAD_FOLDER'])
                
                if uploaded_df is None:
                    logger.warning(f"File upload failed: {error}")
                    return render_template('naive_bayes.html', 
                                          df=df, 
                                          columns=columns,
                                          error=error)
                
                try:
                    session['df'] = uploaded_df.to_json()
                    session.pop('selected_attributes', None)
                    session.pop('target_attribute', None)
                    session.pop('attribute_values', None)
                    session['laplace'] = False
                    df = uploaded_df
                    columns = uploaded_df.columns.tolist()
                    
                    if filepath and os.path.exists(filepath):
                        os.remove(filepath)
                        logger.debug(f"Deleted file: {filepath}")
                    
                    logger.info("Successfully processed uploaded file")
                    return render_template('naive_bayes.html', 
                                          df=df, 
                                          columns=columns)
                except Exception as e:
                    error = f"Lỗi khi xử lý file: {str(e)}"
                    logger.error(error)
                    if filepath and os.path.exists(filepath):
                        os.remove(filepath)
                    return render_template('naive_bayes.html', 
                                          df=df, 
                                          columns=columns, 
                                          error=error)
            
            elif action == 'select_attributes':
                if 'df' not in session:
                    error = "Vui lòng tải dữ liệu trước khi chọn thuộc tính"
                    logger.warning(error)
                    return render_template('naive_bayes.html', 
                                          error=error)
                
                target_attribute = request.form.get('target_attribute')
                selected_attributes = request.form.getlist('selected_attributes')
                laplace = 'laplace' in request.form
                
                if not target_attribute:
                    error = "Vui lòng chọn thuộc tính mục tiêu"
                    logger.warning(error)
                    return render_template('naive_bayes.html', 
                                          df=df, 
                                          columns=columns,
                                          error=error)
                
                if not selected_attributes:
                    error = "Vui lòng chọn ít nhất một thuộc tính đặc trưng"
                    logger.warning(error)
                    return render_template('naive_bayes.html', 
                                          df=df, 
                                          columns=columns,
                                          error=error)
                
                if target_attribute in selected_attributes:
                    selected_attributes = [attr for attr in selected_attributes if attr != target_attribute]
                
                if not selected_attributes:
                    error = "Vui lòng chọn ít nhất một thuộc tính đặc trưng khác với thuộc tính mục tiêu"
                    logger.warning(error)
                    return render_template('naive_bayes.html', 
                                          df=df, 
                                          columns=columns,
                                          error=error)
                
                session['target_attribute'] = target_attribute
                session['selected_attributes'] = selected_attributes
                session['laplace'] = laplace
                attribute_values = {attr: sorted(df[attr].astype(str).unique().tolist()) for attr in selected_attributes}
                session['attribute_values'] = attribute_values
                
                logger.info(f"Selected attributes: {selected_attributes}, Target: {target_attribute}, Laplace: {laplace}")
                return render_template('naive_bayes.html', 
                                      df=df, 
                                      columns=columns,
                                      target_attribute=target_attribute,
                                      selected_attributes=selected_attributes,
                                      attribute_values=attribute_values,
                                      show_prediction_form=True)
        
        target_attribute = session.get('target_attribute')
        selected_attributes = session.get('selected_attributes', [])
        attribute_values = session.get('attribute_values', {})
        
        return render_template('naive_bayes.html', 
                              df=df, 
                              columns=columns,
                              target_attribute=target_attribute,
                              selected_attributes=selected_attributes,
                              attribute_values=attribute_values,
                              error=error,
                              show_prediction_form=bool(target_attribute and selected_attributes))

    @app.route('/naive-bayes/predict', methods=['POST'])
    def naive_bayes_predict():
        if 'df' not in session:
            error = "Vui lòng tải dữ liệu trước khi dự đoán"
            logger.warning(error)
            return render_template('naive_bayes.html', error=error)
        
        df = pd.read_json(session['df'])
        columns = df.columns.tolist()
        
        try:
            target_attribute = session.get('target_attribute')
            selected_attributes = session.get('selected_attributes', [])
            laplace = session.get('laplace', False)
            
            if not target_attribute or not selected_attributes:
                error = "Vui lòng chọn thuộc tính trước khi dự đoán"
                logger.warning(error)
                return render_template('naive_bayes.html', 
                                      df=df, 
                                      columns=columns,
                                      error=error)
            
            sample_for_probability = {}
            for attr in selected_attributes:
                if attr in request.form:
                    sample_for_probability[attr] = request.form[attr]
                else:
                    error = f"Giá trị cho thuộc tính '{attr}' bị thiếu"
                    logger.warning(error)
                    return render_template('naive_bayes.html', 
                                          df=df, 
                                          columns=columns,
                                          target_attribute=target_attribute,
                                          selected_attributes=selected_attributes,
                                          attribute_values=session.get('attribute_values', {}),
                                          error=error,
                                          show_prediction_form=True)
            
            logger.info(f"Running Naive Bayes prediction with sample: {sample_for_probability}, Laplace: {laplace}")
            result = run_naive_bayes(df, selected_attributes, target_attribute, laplace, sample_for_probability)
            
            if not result['success']:
                logger.error(f"Naive Bayes prediction failed: {result['error']}")
                return render_template('naive_bayes.html', 
                                      df=df, 
                                      columns=columns,
                                      target_attribute=target_attribute,
                                      selected_attributes=selected_attributes,
                                      attribute_values=session.get('attribute_values', {}),
                                      error=result['error'],
                                      show_prediction_form=True)
            
            # Log detailed results including unnormalized scores
            logger.info(f"Prediction result: {result['predicted_class']}, Probabilities: {result['probabilities']}, Unnormalized Scores: {result.get('unnormalized_scores', {})}")
            return render_template('naive_bayes.html', 
                                  df=df, 
                                  columns=columns,
                                  target_attribute=target_attribute,
                                  selected_attributes=selected_attributes,
                                  attribute_values=session['attribute_values'],
                                  result=result,
                                  show_prediction_form=True)
        except Exception as e:
            logger.error(f"Error in prediction route: {str(e)}")
            return render_template('naive_bayes.html', 
                                  df=df, 
                                  columns=columns,
                                  target_attribute=session.get('target_attribute'),
                                  selected_attributes=session.get('selected_attributes', []),
                                  attribute_values=session.get('attribute_values', {}),
                                  error=f"Lỗi khi dự đoán: {str(e)}",
                                  show_prediction_form=True)