# algorithm/decision_tree_id3.py
import pandas as pd
import math
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import pydotplus
from IPython.display import Image

# Use a non-interactive backend for matplotlib to avoid display issues in Flask
matplotlib.use('Agg')

# Utility function to convert numpy types to Python types
def to_json_serializable(obj):
    """Recursively convert numpy types to JSON-serializable Python types"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {to_json_serializable(k): to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    return obj

# Functions for entropy, information gain, and Gini index
def entropy(data, target_col):
    """Tính entropy của tập dữ liệu"""
    labels = data[target_col]
    label_count = Counter(labels)
    total_samples = len(labels)
    
    if total_samples == 0:
        return 0
    
    entropy_val = 0
    for _, count in label_count.items():
        p_i = count / total_samples
        entropy_val -= p_i * math.log2(p_i)
    return entropy_val

def information_gain(data, feature, target_col):
    """Tính information gain của một thuộc tính"""
    total_entropy = entropy(data, target_col)
    total_samples = len(data)
    
    weighted_entropy = 0
    for value in data[feature].unique():
        subset = data[data[feature] == value]
        subset_size = len(subset)
        subset_entropy = entropy(subset, target_col)
        weighted_entropy += (subset_size / total_samples) * subset_entropy
    
    return total_entropy - weighted_entropy

def gini_index(data, target_col):
    """Tính Gini Index trên tập giá trị"""
    total_samples = len(data)
    labels = data[target_col]
    label_count = Counter(labels)

    if total_samples == 0:
        return 0
    
    gini_val = 0
    for _, count in label_count.items():
        p_i = count / total_samples
        gini_val += p_i ** 2

    return 1 - gini_val

def gini_index_split(data, feature, target_col):
    """Tính Gini Index cho một thuộc tính"""
    total_samples = len(data)
    weighted_gini = 0

    unique_values = data[feature].unique()

    for value in unique_values:
        subset = data[data[feature] == value]
        subset_size = len(subset)
        if subset_size > 0:  
            subset_gini = gini_index(subset, target_col)
            weighted_gini += (subset_size / total_samples) * subset_gini

    return weighted_gini

def find_best_feature(data, target_col, features, criterion="information_gain"):
    """Tìm thuộc tính tốt nhất dựa trên tiêu chí được chọn"""
    best_feature = None
    best_score = None
    scores = {}
    
    print(f"Tìm thuộc tính tốt nhất theo tiêu chí: {criterion}")
    
    if criterion == "information_gain":
        for feature in features:
            score = information_gain(data, feature, target_col)
            scores[feature] = score
            print(f"  - {feature}: Information Gain = {score:.6f}")
            if best_score is None or score > best_score:
                best_score = score
                best_feature = feature
    elif criterion == "gini_index":
        for feature in features:
            score = gini_index_split(data, feature, target_col)
            scores[feature] = score
            print(f"  - {feature}: Gini Index = {score:.6f}")
            if best_score is None or score < best_score:  # Gini Index chọn giá trị nhỏ nhất
                best_score = score
                best_feature = feature
    
    print(f"Chọn thuộc tính tốt nhất: {best_feature} với điểm số: {best_score:.6f}")
    return best_feature, scores

# Class to represent a node in the decision tree
class Node:
    def __init__(self, feature=None, value=None, label=None, branches=None):
        self.feature = feature  # Feature to split on
        self.value = value      # Value of the feature (for branches)
        self.label = label      # Leaf node label (if applicable)
        self.branches = branches if branches is not None else {}  # Dictionary of branches

    def to_dict(self):
        """Convert the node to a dictionary for JSON serialization"""
        branches = {to_json_serializable(key): child.to_dict() for key, child in self.branches.items()}
        return {
            'feature': self.feature,
            'value': to_json_serializable(self.value),
            'label': to_json_serializable(self.label),
            'branches': branches
        }

    @staticmethod
    def from_dict(data):
        """Reconstruct a Node from a dictionary"""
        node = Node(
            feature=data.get('feature'),
            value=data.get('value'),
            label=data.get('label')
        )
        node.branches = {key: Node.from_dict(child) for key, child in data.get('branches', {}).items()}
        return node

# Function to extract rules from the decision tree
def extract_rules(node, path=None, rules=None):
    """Trích xuất luật từ cây quyết định"""
    if rules is None:
        rules = []
    if path is None:
        path = []
    
    if node.label is not None:
        # Leaf node: create a rule
        if path:
            rule = "IF " + " AND ".join([f"{condition[0]} = {condition[1]}" for condition in path]) + f" THEN {node.label}"
        else:
            rule = f"THEN {node.label}"
        rules.append(rule)
        return rules
    
    # Non-leaf node: recurse through branches
    for value, child in node.branches.items():
        new_path = path + [(node.feature, value)]
        extract_rules(child, new_path, rules)
    
    return rules

def id3(data, target_col, features, criterion="information_gain", steps=None):
    """Xây dựng cây quyết định theo thuật toán ID3"""
    if steps is None:
        steps = []
    
    # Trường hợp cơ bản
    labels = data[target_col]
    if len(labels.unique()) == 1:  
        unique_label = labels.iloc[0]
        steps.append(f"Tất cả mẫu có cùng nhãn: {unique_label}")
        return Node(label=unique_label)
    
    if not features:  # Nếu không còn thuộc tính để phân chia
        majority_label = Counter(labels).most_common(1)[0][0]
        steps.append(f"Không còn thuộc tính để phân chia. Sử dụng nhãn đa số: {majority_label}")
        return Node(label=majority_label)
    
    # Tìm thuộc tính tốt nhất để phân chia
    best_feature, scores = find_best_feature(data, target_col, features, criterion)
    
    # Log thông tin chi tiết về điểm số
    score_desc = scores[best_feature]
    if criterion == "information_gain":
        score_label = "Độ lợi thông tin"
    else:
        score_label = "Chỉ số Gini"
        
    steps.append(f"Thuộc tính tốt nhất để phân chia: {best_feature} ({score_label}: {score_desc:.6f})")
    
    # Thêm thông tin về tất cả các thuộc tính đã xem xét
    for feature, score in scores.items():
        steps.append(f"  - {feature}: {score_label} = {score:.6f}")
    
    # Tạo nút gốc cho thuộc tính này
    root = Node(feature=best_feature)
    
    # Loại bỏ thuộc tính tốt nhất khỏi danh sách thuộc tính
    remaining_features = [f for f in features if f != best_feature]
    
    # Phân chia dữ liệu dựa trên thuộc tính tốt nhất
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subset_size = len(subset)
        # Log số mẫu cho giá trị của thuộc tính
        steps.append(f"Phân chia trên {best_feature} = {value}: có {subset_size} mẫu")
        print(f"Phân chia trên {best_feature} = {value}: có {subset_size} mẫu")
        
        # Bổ sung log chi tiết về các thuộc tính khác trong tập con (ví dụ: thu nhập)
        if len(subset) > 0 and remaining_features:
            steps.append(f"Chi tiết tập con cho {best_feature} = {value}:")
            for feature in remaining_features:
                value_counts = subset[feature].value_counts()
                steps.append(f"  - {feature}:")
                for val, count in value_counts.items():
                    steps.append(f"    - {val}: có {count} mẫu")
                    print(f"    - {feature} = {val}: có {count} mẫu")
        
        if len(subset) == 0:
            majority_label = Counter(labels).most_common(1)[0][0]
            steps.append(f"Không có mẫu nào cho {best_feature} = {value}. Sử dụng nhãn đa số: {majority_label}")
            root.branches[value] = Node(label=majority_label)
        else:
            root.branches[value] = id3(subset, target_col, remaining_features, criterion, steps)
    
    return root
    """Xây dựng cây quyết định theo thuật toán ID3"""
    if steps is None:
        steps = []
    
    # Trường hợp cơ bản
    labels = data[target_col]
    if len(labels.unique()) == 1:  
        unique_label = labels.iloc[0]
        steps.append(f"Tất cả mẫu có cùng nhãn: {unique_label}")
        return Node(label=unique_label)
    
    if not features:  # Nếu không còn thuộc tính để phân chia
        majority_label = Counter(labels).most_common(1)[0][0]
        steps.append(f"Không còn thuộc tính để phân chia. Sử dụng nhãn đa số: {majority_label}")
        return Node(label=majority_label)
    
    # Tìm thuộc tính tốt nhất để phân chia
    best_feature, scores = find_best_feature(data, target_col, features, criterion)
    
    # Log thông tin chi tiết hơn để debug
    score_desc = scores[best_feature]
    if criterion == "information_gain":
        score_label = "Độ lợi thông tin"
    else:
        score_label = "Chỉ số Gini"
        
    steps.append(f"Thuộc tính tốt nhất để phân chia: {best_feature} ({score_label}: {score_desc:.6f})")
    
    # Thêm thông tin về tất cả các thuộc tính đã xem xét
    for feature, score in scores.items():
        steps.append(f"  - {feature}: {score_label} = {score:.6f}")
    
    # Tạo nút gốc cho thuộc tính này
    root = Node(feature=best_feature)
    
    # Loại bỏ thuộc tính tốt nhất khỏi danh sách thuộc tính
    remaining_features = [f for f in features if f != best_feature]
    
    # Phân chia dữ liệu dựa trên thuộc tính tốt nhất
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        subset_size = len(subset)
        # In số mẫu cho từng giá trị của thuộc tính
        print(f"Giá trị {value} trong thuộc tính {best_feature} có {subset_size} mẫu")
        
        steps.append(f"Phân chia trên {best_feature} = {value}")
        
        if len(subset) == 0:
            majority_label = Counter(labels).most_common(1)[0][0]
            steps.append(f"Không có mẫu nào cho {best_feature} = {value}. Sử dụng nhãn đa số: {majority_label}")
            root.branches[value] = Node(label=majority_label)
        else:
            root.branches[value] = id3(subset, target_col, remaining_features, criterion, steps)
    
    return root

# Function to visualize the decision tree using pydotplus
def visualize_tree(node, graph=None):
    if graph is None:
        graph = pydotplus.Dot(graph_type='digraph')
    
    if node.label is not None:
        graph.add_node(pydotplus.Node(str(id(node)), label=str(node.label), shape='ellipse', fillcolor='lightgreen', style='filled'))
        return graph
    
    # Create a node for the current feature
    graph.add_node(pydotplus.Node(str(id(node)), label=node.feature, shape='box', fillcolor='lightblue', style='filled'))
    
    # Recursively add branches
    for value, child in node.branches.items():
        child_graph = visualize_tree(child, graph)
        edge = pydotplus.Edge(str(id(node)), str(id(child)), label=str(value))
        graph.add_edge(edge)
    
    return graph

# Function to convert the graph to a base64-encoded image
def graph_to_base64(graph):
    img_data = graph.create_png()
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    return img_base64

# Function to run the decision tree algorithm
def run_decision_tree(df, target_col, criterion="information_gain"):
    try:
        if target_col not in df.columns:
            return {'error': f"Cột mục tiêu '{target_col}' không tồn tại trong dữ liệu"}
        
        features = [col for col in df.columns if col != target_col]
        if not features:
            return {'error': "Không có thuộc tính nào để xây dựng cây quyết định"}
        
        
        # Build the decision tree
        steps = []
        tree = id3(df, target_col, features, criterion, steps)
        
        # Extract rules from the tree
        rules = extract_rules(tree)
        
        # Visualize the tree
        graph = visualize_tree(tree)
        tree_image = graph_to_base64(graph)
        
        # Get unique values for each feature (for prediction form)
        feature_values = {feature: to_json_serializable(df[feature].unique().tolist()) for feature in features}
        
        # Convert the tree to a dictionary for serialization
        tree_dict = to_json_serializable(tree.to_dict())
        
        return {
            'tree': tree_dict,  # Store the serializable tree
            'tree_image': tree_image,
            'target': target_col,
            'features': features,
            'feature_values': feature_values,
            'steps': steps,
            'criterion': criterion,
            'rules': rules  # Add extracted rules
        }
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return {'error': f"Lỗi xây dựng cây quyết định: {str(e)}\n{traceback_str}"}


# Function to predict using the decision tree
def predict(tree_dict, instance):
    # Reconstruct the tree from the dictionary
    tree = Node.from_dict(tree_dict)
    node = tree
    while node.label is None:
        feature = node.feature
        value = instance.get(feature)
        if value not in node.branches:
            return "Không thể dự báo: Giá trị không tồn tại trong cây"
        node = node.branches[value]
    return node.label