import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import io
import base64

def preprocess_transactions(df):
    """Chuyển DataFrame thành từ điển giao dịch"""
    if df.empty:
        raise ValueError("Không có giao dịch nào")
    if df.shape[1] < 2:
        raise ValueError("DataFrame cần ít nhất 2 cột: Transaction ID và Item")
    
    transactions = {}
    for index, row in df.iterrows():
        trans_id = row[0]
        item = row[1]
        if pd.isna(trans_id) or pd.isna(item):
            continue  # Bỏ qua nếu có giá trị NaN
        if trans_id not in transactions:
            transactions[trans_id] = set()
        transactions[trans_id].add(str(item))  # Chuyển thành chuỗi để đồng nhất
    
    if not transactions:
        raise ValueError("Không có giao dịch hợp lệ sau khi tiền xử lý")
    return transactions

def calculate_support(itemset, transactions):
    """Tính support cho một tập hợp"""
    count = sum(1 for items in transactions.values() if itemset.issubset(items))
    return count / len(transactions)

def apriori(transactions, minsup):
    """Thuật toán Apriori để tìm tập hợp thường xuyên"""
    if not 0 <= minsup <= 1:
        raise ValueError("min_support phải nằm trong khoảng [0, 1]")
    
    items = set(item for sublist in transactions.values() for item in sublist)
    current_itemsets = [{item} for item in items]
    frequent_itemsets = {}
    
    k = 1
    while current_itemsets:
        # Tính support cho các tập hợp hiện tại
        candidates = {}
        for itemset in current_itemsets:
            support = calculate_support(itemset, transactions)
            if support >= minsup:
                candidates[frozenset(itemset)] = support
        
        # Thêm vào frequent_itemsets
        frequent_itemsets.update(candidates)
        
        # Tạo tập hợp ứng viên mới
        current_itemsets = []
        candidate_keys = list(candidates.keys())
        for i, j in combinations(range(len(candidate_keys)), 2):
            union_set = set(candidate_keys[i]).union(candidate_keys[j])
            if len(union_set) == k + 1 and union_set not in current_itemsets:
                current_itemsets.append(union_set)
        
        k += 1
    
    return sorted(frequent_itemsets.items(), key=lambda x: len(x[0]))

def find_maximal_frequent_itemset(frequent_itemsets):
    """Tìm tập hợp thường xuyên tối đại"""
    if not frequent_itemsets:
        return []
    
    # Sắp xếp theo kích thước giảm dần để tối ưu
    sorted_items = sorted(frequent_itemsets, key=lambda x: len(x[0]), reverse=True)
    maximal_itemsets = []
    seen = set()
    
    for itemset, _ in sorted_items:
        itemset = frozenset(itemset)
        if not any(itemset.issubset(maximal) for maximal in maximal_itemsets):
            maximal_itemsets.append(itemset)
            seen.add(itemset)
    
    return maximal_itemsets

def calculate_confidence(A, B, transactions):
    """Tính độ tin cậy của luật kết hợp"""
    union_set = A.union(B)
    support_union = calculate_support(union_set, transactions)
    support_A = calculate_support(A, transactions)
    
    if support_A == 0:
        raise ValueError(f"Support của {A} bằng 0, không thể tính confidence")
    return support_union / support_A

def generate_association_rules(frequent_itemsets, transactions, min_confidence):
    """Tạo luật kết hợp"""
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence phải nằm trong khoảng [0, 1]")
    
    rules = []
    for itemset, _ in frequent_itemsets:
        itemset = set(itemset)
        for r in range(1, len(itemset)):
            for A in combinations(itemset, r):
                A = set(A)
                B = itemset - A
                if B:
                    try:
                        confidence = calculate_confidence(A, B, transactions)
                        if confidence >= min_confidence:
                            rules.append((A, B, confidence))
                    except ValueError:
                        continue
    return rules

def plot_association_rules(frequent_itemsets):
    """Vẽ biểu đồ cột của support"""
    if not frequent_itemsets:
        return None
    
    with io.BytesIO() as buffer:
        plt.figure(figsize=(10, 6))
        itemsets = [', '.join(map(str, itemset)) for itemset, _ in frequent_itemsets]
        supports = [support for _, support in frequent_itemsets]
        
        plt.bar(itemsets, supports)
        plt.title('Frequent Itemsets Support')
        plt.xlabel('Itemsets')
        plt.ylabel('Support')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
    
    return plot_base64

def run_association_rules(df, min_support=0.4, min_confidence=0.4):
    """Chạy phân tích luật kết hợp hoàn chỉnh"""
    try:
        transactions = preprocess_transactions(df)
        frequent_itemsets = apriori(transactions, min_support)
        maximal_frequent_itemsets = find_maximal_frequent_itemset(frequent_itemsets)
        association_rules = generate_association_rules(frequent_itemsets, transactions, min_confidence)
        plot = plot_association_rules(frequent_itemsets)
        
        return {
            'frequent_itemsets': frequent_itemsets,
            'maximal_frequent_itemsets': maximal_frequent_itemsets,
            'association_rules': association_rules,
            'plot': plot
        }
    except Exception as e:
        return {'error': str(e)}