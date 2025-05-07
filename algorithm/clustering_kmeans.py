import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import logging

logger = logging.getLogger(__name__)

def run_kmeans(df=None, K=None, max_iterations=100, use_sample_data=False):
    """
    Chạy thuật toán K-Means trên dữ liệu đầu vào.
    
    Args:
        df (pd.DataFrame): Dữ liệu đầu vào (CSV hoặc mẫu).
        K (int): Số cụm, bắt buộc.
        max_iterations (int): Số vòng lặp tối đa.
        use_sample_data (bool): Sử dụng dữ liệu mẫu nếu True.
    
    Returns:
        dict: Kết quả bao gồm DataFrame, centroids, số vòng lặp, biểu đồ (nếu 2D), và các bước giải thuật.
    """
    logger.debug("Starting K-Means algorithm")
    
    # Kiểm tra K
    if K is None or not isinstance(K, int) or K <= 0:
        logger.error(f"Invalid K value: {K}")
        return {'error': f'Số cụm K phải là số nguyên dương: {K}'}
    
    # Nếu sử dụng dữ liệu mẫu hoặc không có df
    if use_sample_data or df is None:
        logger.debug("Generating sample data")
        df = pd.DataFrame(np.random.rand(100, 2) * 100, columns=['X', 'Y'])
    
    # Kiểm tra df có phải là DataFrame
    logger.debug(f"Input df type: {type(df)}, value: {df}")
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Input is not a DataFrame: {type(df)}")
        return {'error': f'Dữ liệu đầu vào không hợp lệ: {type(df)}'}
    
    # Chỉ lấy các cột số
    try:
        df_numeric = df.select_dtypes(include=[np.number]).copy()
        logger.debug(f"Numeric columns: {df_numeric.columns.tolist()}")
        if df_numeric.empty:
            logger.error("No numeric columns in data")
            return {'error': 'Dữ liệu không chứa cột số nào'}
    except Exception as e:
        logger.error(f"Error processing numeric columns: {str(e)}")
        return {'error': f'Lỗi xử lý dữ liệu: {str(e)}'}
    
    # Số chiều
    n = df_numeric.shape[1]
    if n < 1:
        logger.error("No valid columns after filtering")
        return {'error': 'Không có cột dữ liệu hợp lệ'}
    
    logger.debug(f"Data shape: {df_numeric.shape}, K: {K}")
    
    # Khởi tạo K centroid ngẫu nhiên
    max_values = df_numeric.max().values
    centroids = np.random.rand(K, n) * max_values
    
    # Gắn nhãn cụm cho từng điểm
    df_numeric['cluster'] = -1
    
    # Lưu các bước giải thuật
    steps = []
    steps.append(f"Vòng 1: Khởi tạo {K} centroids: {np.round(centroids, 2).tolist()}")
    
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))
    
    # Lặp cho đến khi hội tụ hoặc đạt max_iterations
    iteration = 0
    while iteration < max_iterations:
        prev_clusters = df_numeric['cluster'].copy()
        
        # Bước 1: Gán điểm vào cụm gần nhất
        for i in range(len(df_numeric)):
            distances = [euclidean_distance(df_numeric.iloc[i, :-1].values, centroid) 
                        for centroid in centroids]
            df_numeric.at[i, 'cluster'] = np.argmin(distances)
        
        # Ghi lại các điểm thuộc mỗi cụm
        cluster_points = {}
        for k in range(K):
            indices = df_numeric[df_numeric['cluster'] == k].index.tolist()
            cluster_points[k] = indices
            steps.append(f"Vòng {iteration + 1}: Cụm {k} chứa các điểm: {indices}")
        
        # Bước 2: Cập nhật lại centroid
        for k in range(K):
            cluster_points = df_numeric[df_numeric['cluster'] == k].iloc[:, :-1]
            if not cluster_points.empty:
                centroids[k] = cluster_points.mean().values
        steps.append(f"Vòng {iteration + 1}: Cập nhật centroids: {np.round(centroids, 2).tolist()}")
        
        # Kiểm tra hội tụ
        if df_numeric['cluster'].equals(prev_clusters):
            logger.debug(f"Converged after {iteration + 1} iterations")
            break
        
        iteration += 1
    
    if iteration >= max_iterations:
        logger.warning("Stopped due to max iterations reached")
    
    # Tạo biểu đồ nếu dữ liệu là 2D
    plot_base64 = None
    if n == 2:
        try:
            fig, ax = plt.subplots()
            colors = ['red', 'green', 'blue']
            for k in range(min(K, len(colors))):  # Giới hạn số màu
                cluster_data = df_numeric[df_numeric['cluster'] == k]
                ax.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], 
                          c=colors[k], label=f'Cluster {k}')
            ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', 
                      s=100, label='Centroids')
            ax.legend()
            ax.set_title("K-Means Clustering")
            ax.set_xlabel(df_numeric.columns[0])
            ax.set_ylabel(df_numeric.columns[1])
            
            # Lưu biểu đồ thành base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
        except Exception as e:
            logger.error(f"Error generating plot: {str(e)}")
            plot_base64 = None
    
    logger.debug(f"K-Means completed with {iteration + 1} iterations")
    return {
        'data': df_numeric,
        'centroids': centroids,
        'iterations': iteration + 1,
        'plot': plot_base64,
        'steps': steps,
        'error': None
    }