import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import os
from flask import render_template, request, send_file
from werkzeug.utils import secure_filename

# Đường dẫn thư mục mặc định
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_column(df, col):
    """Phân loại cột dữ liệu"""
    unique_values = df[col].nunique()
    dtype = df[col].dtype.name
    if unique_values == 2 and dtype in ['int64', 'float64']:
        return 'Binary'
    elif dtype in ['object', 'category']:
        return 'Nominal'
    elif dtype in ['int64', 'float64']:
        return 'Numeric'
    return 'Unknown'

def preprocess_outlier_numeric_data(df, col):
    """Xử lý giá trị ngoại lai cho cột số"""
    if classify_column(df, col) == 'Numeric':
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        median_value = df[col].median()
        df[col] = df[col].apply(lambda x: median_value if (x < lower_bound or x > upper_bound) else x)
        return True
    return False

def preprocess_missing_value(df, col):
    if df[col].isnull().sum() == 0:
        return False  # Không có dữ liệu thiếu

    try:
        if classify_column(df, col) == 'Numeric':
            if df[col].dropna().empty:
                df[col].fillna(0, inplace=True)  # Nếu toàn bộ cột là NaN
            else:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)  # Chỉ thay thế NaN
        else:  # Xử lý dữ liệu Nominal (Categorical)
            mode_value = df[col].mode().iloc[0]
            df[col].fillna(mode_value, inplace=True)
        return True
    except Exception as e:
        print(f"Lỗi khi xử lý cột {col}: {str(e)}")
        return False


def get_boxplot(df, col):
    """Tạo boxplot cho cột số"""
    if classify_column(df, col) != 'Numeric':
        return None
    
    with io.BytesIO() as buf:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot của {col}')
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close()
        return plot_url

def reduce_correlated_features(df, threshold=0.9):
    """Giảm các đặc trưng tương quan cao"""
    numeric_cols = [col for col in df.columns if classify_column(df, col) == 'Numeric']
    if not numeric_cols:
        return [], []
    
    corr_matrix = df[numeric_cols].corr()
    columns_to_drop = set()
    correlations = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value > threshold:
                correlations.append({
                    'col1': corr_matrix.columns[i],
                    'col2': corr_matrix.columns[j],
                    'value': corr_value
                })
                columns_to_drop.add(corr_matrix.columns[j])
    
    return correlations, list(columns_to_drop)

def preprocess_page():
    """Xử lý trang tiền xử lý dữ liệu"""
    if request.method != 'POST':
        return render_template('preprocess.html')
    
    if 'file' not in request.files:
        return render_template('preprocess.html', error='Không có file được tải lên')
    
    file = request.files['file']
    if not file or file.filename == '':
        return render_template('preprocess.html', error='Chưa chọn file')
    
    if not allowed_file(file.filename):
        return render_template('preprocess.html', error='Chỉ hỗ trợ file CSV')

    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        df = pd.read_csv(filepath)
        original_df = df.copy()
        
        # Phân loại cột
        column_types = {col: classify_column(df, col) for col in df.columns}
        missing_values = df.isnull().sum().to_dict()
        
        # Xử lý giá trị thiếu
        missing_methods = {}
        for col in df.columns:
            preprocess_missing_value(df, col)
        
        # Xử lý ngoại lai
        outlier_cols = []
        boxplots = {}
        for col in df.columns:
            if preprocess_outlier_numeric_data(df, col):
                outlier_cols.append(col)
                boxplot = get_boxplot(original_df, col)
                if boxplot:
                    boxplots[col] = boxplot
        
        # Xử lý tương quan
        correlations, cols_to_drop = reduce_correlated_features(df)
        selected_drop = request.form.get('drop_correlated')
        if selected_drop in df.columns:  # Kiểm tra cột có tồn tại
            df.drop(selected_drop, axis=1, inplace=True)
        
        # Lưu file đã xử lý
        output_filename = f'processed_{filename}'
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        df.to_csv(output_path, index=False)
        
        return render_template('preprocess.html',
                            column_types=column_types,
                            missing_values=missing_values,
                            missing_methods=missing_methods,
                            boxplots=boxplots,
                            correlations=correlations,
                            filename=output_filename,
                            processed=True)
    
    except pd.errors.EmptyDataError:
        return render_template('preprocess.html', error='File CSV rỗng')
    except Exception as e:
        return render_template('preprocess.html', error=f'Lỗi xử lý file: {str(e)}')
    finally:
        # Xóa file tạm nếu cần
        if os.path.exists(filepath):
            os.remove(filepath)

def download_file(filename):
    """Tải file đã xử lý"""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return render_template('preprocess.html', error='File không tồn tại')