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

    for value in data[feature].unique():
        subset = data[data[feature] == value]
        subset_size = len(subset)
        subset_gini = gini_index(subset, target_col)
        weighted_gini += (subset_size / total_samples) * subset_gini

    return weighted_gini

def find_best_feature(data, target_col, features, criterion="information_gain"):
    """Tìm thuộc tính tốt nhất dựa trên tiêu chí được chọn"""
    best_feature = None
    scores = {}  # Lưu điểm số của từng thuộc tính
    
    if criterion == "information_gain":
        max_score = -float('inf')
        for feature in features:
            score = information_gain(data, feature, target_col)
            scores[feature] = score
            if score > max_score:
                max_score = score
                best_feature = feature
    elif criterion == "gini_index":
        min_score = float('inf')
        for feature in features:
            score = gini_index_split(data, feature, target_col)
            scores[feature] = score
            if score < min_score:  # Gini Index chọn giá trị nhỏ nhất
                min_score = score
                best_feature = feature
    
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

# Function to build the ID3 decision tree
def id3(data, target_col, features, criterion="information_gain", steps=None):
    if steps is None:
        steps = []
    
    # Base cases
    labels = data[target_col]
    if len(labels.unique()) == 1:  # If all examples have the same label
        steps.append(f"All examples have the same label: {labels.iloc[0]}")
        return Node(label=labels.iloc[0])
    
    if not features:  # If no features left to split on
        majority_label = Counter(labels).most_common(1)[0][0]
        steps.append(f"No features left to split on. Using majority label: {majority_label}")
        return Node(label=majority_label)
    
    # Find the best feature to split on
    best_feature, scores = find_best_feature(data, target_col, features, criterion)
    score_label = "Information Gain" if criterion == "information_gain" else "Gini Index"
    steps.append(f"Best feature to split on: {best_feature} ({score_label}: {scores[best_feature]:.4f})")
    
    # Create a root node for this feature
    root = Node(feature=best_feature)
    
    # Remove the best feature from the list of features
    remaining_features = [f for f in features if f != best_feature]
    
    # Split the data based on the best feature
    for value in data[best_feature].unique():
        steps.append(f"Splitting on {best_feature} = {value}")
        subset = data[data[best_feature] == value]
        if len(subset) == 0:
            majority_label = Counter(labels).most_common(1)[0][0]
            steps.append(f"No examples for {best_feature} = {value}. Using majority label: {majority_label}")
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
        # Validate inputs
        if target_col not in df.columns:
            return {'error': f"Cột mục tiêu '{target_col}' không tồn tại trong dữ liệu"}
        
        # Get features (all columns except the target column)
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
        return {'error': f"Lỗi xây dựng cây quyết định: {str(e)}"}

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