import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import io
import base64
import logging

logger = logging.getLogger(__name__)

def calculate_prior_probabilities(labels, laplace=False):
    """Calculate prior probabilities for each class with optional Laplace smoothing"""
    if len(labels) == 0:
        raise ValueError("Label array is empty")
    unique_classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    if laplace:
        # Laplace smoothing: add 1 to each class count and number of classes to denominator
        return dict(zip(unique_classes, (counts + 1) / (total + len(unique_classes))))
    return dict(zip(unique_classes, counts / total))

def calculate_likelihoods(data, labels, laplace=False):
    """Calculate likelihood probabilities with optional Laplace smoothing for categorical data"""
    if data.shape[0] != len(labels):
        raise ValueError("Data and labels must have the same number of rows")
    n_features = data.shape[1]
    likelihoods = defaultdict(lambda: defaultdict(lambda: 0))
    classes = np.unique(labels)
    
    for feature in range(n_features):
        feature_values = np.unique(data[:, feature])  # All possible values of the feature
        for cls in classes:
            class_data = data[labels == cls, feature]
            unique_values, counts = np.unique(class_data, return_counts=True)
            total = len(class_data)
            
            if laplace:
                # Laplace smoothing: add number of possible feature values to denominator
                total += len(feature_values)
                # Assign probability for all possible feature values
                for val in feature_values:
                    count = counts[unique_values == val][0] if val in unique_values else 0
                    likelihoods[(feature, str(val))][cls] = (count + 1) / total
            else:
                # No smoothing: only assign probabilities for observed values
                if total > 0:
                    for val, count in zip(unique_values, counts):
                        likelihoods[(feature, str(val))][cls] = count / total
                else:
                    # If no data for this class, assign 0 probability
                    for val in feature_values:
                        likelihoods[(feature, str(val))][cls] = 0
    
    return likelihoods

def calculate_class_probabilities(sample, priors, likelihoods, selected_attributes):
    """Calculate probabilities and unnormalized scores for all classes"""
    class_scores = {}
    unnormalized_scores = {}
    calculation_steps = []
    
    # Prior probabilities
    prior_steps = [f"P(Lớp = {cls}) = {priors[cls]:.4f}" for cls in priors]
    calculation_steps.append({"title": "Prior Probabilities:", "steps": prior_steps})
    
    # Likelihood and unnormalized scores
    likelihood_steps = []
    unnormalized_steps = []
    
    for cls in priors:
        score = priors[cls]
        likelihood_step = f"P({', '.join([f'{attr} = {val}' for attr, val in zip(selected_attributes, sample)])} | Lớp = {cls}) = "
        factors = []
        
        for feature_idx, (attr, value) in enumerate(zip(selected_attributes, sample)):
            key = (feature_idx, str(value))
            prob = likelihoods[key][cls] if key in likelihoods and cls in likelihoods[key] else 0
            score *= prob
            factors.append(f"P({attr} = {value} | Lớp = {cls}) = {prob:.4f}" + (" (unseen value)" if prob == 0 else ""))
        
        likelihood_step += " × ".join(factors)
        likelihood_steps.append(likelihood_step)
        unnormalized_scores[cls] = score
        unnormalized_steps.append(f"P({', '.join([f'{attr} = {val}' for attr, val in zip(selected_attributes, sample)])} | Lớp = {cls}) × P(Lớp = {cls}) = {score:.4f}")
        class_scores[cls] = score
    
    calculation_steps.append({"title": "Likelihood Calculation:", "steps": likelihood_steps})
    calculation_steps.append({"title": "Unnormalized Scores:", "steps": unnormalized_steps})
    
    # Normalize probabilities
    total = sum(class_scores.values())
    probabilities = {cls: score / total if total > 0 else 0.0 for cls, score in class_scores.items()}
    
    # Posterior probabilities
    posterior_steps = []
    for cls, prob in probabilities.items():
        posterior_steps.append(f"P(Lớp = {cls} | {', '.join([f'{attr} = {val}' for attr, val in zip(selected_attributes, sample)])}) = {prob:.4f}")
    calculation_steps.append({"title": "Posterior Probabilities:", "steps": posterior_steps})
    
    return probabilities, calculation_steps, unnormalized_scores

def plot_feature_distributions(df, selected_attributes, target_attribute):
    """Generate feature distribution plots for each attribute"""
    plots = []
    classes = df[target_attribute].unique()
    
    for attr in selected_attributes:
        with io.BytesIO() as buffer:
            fig, ax = plt.subplots(figsize=(10, 6))
            is_numeric = pd.api.types.is_numeric_dtype(df[attr])
            
            if is_numeric:
                for cls in classes:
                    subset = df[df[target_attribute] == cls][attr]
                    ax.hist(subset, alpha=0.5, label=f'Lớp {cls}', bins=min(10, len(subset.unique())), density=True)
            else:
                attr_values = df[attr].unique()
                bar_width = 0.8 / len(classes)
                positions = np.arange(len(attr_values))
                
                for i, cls in enumerate(classes):
                    counts = [len(df[(df[target_attribute] == cls) & (df[attr] == val)]) for val in attr_values]
                    ax.bar(positions + i * bar_width, counts, bar_width, alpha=0.7, label=f'Lớp {cls}')
                
                ax.set_xticks(positions + bar_width * (len(classes) - 1) / 2)
                ax.set_xticklabels(attr_values, rotation=45 if len(attr_values) > 5 else 0)
            
            ax.set_title(f'Phân phối {attr} theo lớp')
            ax.set_xlabel(attr)
            ax.set_ylabel('Số lượng')
            ax.legend()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            plots.append({"attribute": attr, "plot": plot_base64})
    
    return plots

def run_naive_bayes(df, selected_attributes, target_attribute, laplace=False, sample_for_probability=None):
    """Run complete Naive Bayes analysis with attribute selection"""
    try:
        if df.empty or df.shape[1] < 2:
            raise ValueError("DataFrame phải có ít nhất 2 cột: đặc trưng và nhãn")
        
        all_columns = df.columns.tolist()
        if not selected_attributes:
            raise ValueError("Vui lòng chọn ít nhất một thuộc tính")
        
        if target_attribute not in all_columns:
            raise ValueError(f"Thuộc tính mục tiêu '{target_attribute}' không tồn tại trong dữ liệu")
        
        if not all(attr in all_columns for attr in selected_attributes):
            invalid_attrs = [attr for attr in selected_attributes if attr not in all_columns]
            raise ValueError(f"Thuộc tính không tồn tại trong dữ liệu: {', '.join(invalid_attrs)}")
        
        if target_attribute in selected_attributes:
            selected_attributes = [attr for attr in selected_attributes if attr != target_attribute]
        
        if not selected_attributes:
            raise ValueError("Vui lòng chọn ít nhất một thuộc tính đặc trưng (khác với thuộc tính mục tiêu)")
        
        X = df[selected_attributes].values
        y = df[target_attribute].values
        X = np.array([[str(val) for val in row] for row in X])
        y = np.array([str(val) for val in y])
        
        attribute_values = {attr: sorted(df[attr].astype(str).unique().tolist()) for attr in selected_attributes}
        class_values = sorted(df[target_attribute].astype(str).unique().tolist())
        
        priors = calculate_prior_probabilities(y, laplace=laplace)
        likelihoods = calculate_likelihoods(X, y, laplace=laplace)
        
        result = {
            'success': True,
            'laplace_used': laplace,
            'target': target_attribute,
            'selected_attributes': selected_attributes,
            'attribute_values': attribute_values,
            'class_values': class_values
        }
        
        if sample_for_probability is not None:
            try:
                sample = np.array([str(sample_for_probability[attr]) for attr in selected_attributes])
                probabilities, calculation_steps, unnormalized_scores = calculate_class_probabilities(sample, priors, likelihoods, selected_attributes)
                predicted_class = max(probabilities, key=probabilities.get)
                result.update({
                    'predicted_class': predicted_class,
                    'probabilities': {cls: prob for cls, prob in probabilities.items()},
                    'calculation_steps': calculation_steps,
                    'unnormalized_scores': unnormalized_scores,
                    'feature_distributions': plot_feature_distributions(df, selected_attributes, target_attribute)
                })
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                result.update({
                    'prediction_error': f"Lỗi khi dự đoán: {str(e)}"
                })
        
        return result
    except Exception as e:
        logger.error(f"Error in Naive Bayes: {str(e)}")
        return {'success': False, 'error': str(e)}