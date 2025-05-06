import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import os
from flask import render_template, request, send_file
from werkzeug.utils import secure_filename
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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

def encode_categorical_data(df, encoding_method):
    """Chuyển đổi dữ liệu categorical thành vector biểu diễn"""
    nominal_cols = [col for col in df.columns if classify_column(df, col) == 'Nominal']
    binary_cols = [col for col in df.columns if classify_column(df, col) == 'Binary']
    
    encoded_cols = {}
    
    # Xử lý các cột nominal
    if nominal_cols:
        if encoding_method == 'one-hot':
            # One-hot encoding
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(df[nominal_cols])
            encoded_feature_names = encoder.get_feature_names_out(nominal_cols)
            
            # Thêm các cột mới vào DataFrame
            for i, feature_name in enumerate(encoded_feature_names):
                df[feature_name] = encoded_data[:, i]
            
            # Lưu thông tin các cột đã encode
            for col in nominal_cols:
                related_cols = [name for name in encoded_feature_names if name.startswith(col + '_')]
                encoded_cols[col] = related_cols
                
            # Xóa các cột gốc
            df.drop(columns=nominal_cols, inplace=True)
            
        elif encoding_method == 'label':
            # Label encoding
            for col in nominal_cols:
                encoder = LabelEncoder()
                df[col + '_encoded'] = encoder.fit_transform(df[col].astype(str))
                encoded_cols[col] = [col + '_encoded']
                df.drop(columns=[col], inplace=True)
    
    # Xử lý các cột binary (chuyển về 0-1)
    for col in binary_cols:
        if df[col].dtype in ['int64', 'float64']:
            # Đảm bảo các cột binary chỉ có giá trị 0-1
            unique_vals = sorted(df[col].unique())
            if len(unique_vals) == 2:
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                df[col] = df[col].map(mapping)
                encoded_cols[col] = [col]  # Không đổi tên nhưng đánh dấu là đã encode
    
    return encoded_cols

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
        
        # Chuyển đổi categorical data thành vector
        encoding_method = request.form.get('encoding_method', 'one-hot')
        encoded_columns = encode_categorical_data(df, encoding_method)
        
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
                            encoded_columns=encoded_columns,
                            encoding_method=encoding_method,
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