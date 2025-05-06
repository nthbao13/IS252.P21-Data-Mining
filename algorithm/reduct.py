import pandas as pd
from collections import defaultdict
from itertools import chain, combinations

def equipvalence_classes(df, attributes):
    eq_classes = defaultdict(list)
    for index, row in df.iterrows():
        key = tuple(row[attr] for attr in attributes)
        eq_classes[key].append(index)
    return eq_classes

def rough_approximations(df, attributes, decision_col, decision_value):
    eq_classes = equipvalence_classes(df, attributes)
    target_object = set(df[df[decision_col] == decision_value].index)
    lower_approx = set()
    upper_approx = set()
    
    # Store step information for visualization
    steps = []
    
    for key, obj_set in eq_classes.items():
        obj_set = set(obj_set)
        is_subset = obj_set.issubset(target_object)
        has_intersection = bool(obj_set & target_object)
        
        steps.append({
            'class': key,
            'objects': list(obj_set),
            'target_intersection': list(obj_set & target_object),
            'is_subset': is_subset,
            'has_intersection': has_intersection
        })
        
        if is_subset:  # Lower approximation
            lower_approx.update(obj_set)
        if has_intersection:  # Upper approximation
            upper_approx.update(obj_set)
    
    return lower_approx, upper_approx, steps, eq_classes

def discernibility_matrix(df, attributes, decision_col):
    n = len(df)
    matrix = [[set() for _ in range(n)] for _ in range(n)]
    
    # Store step information for visualization
    steps = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if df.iloc[i][decision_col] != df.iloc[j][decision_col]:
                for attr in attributes:
                    if df.iloc[i][attr] != df.iloc[j][attr]:
                        matrix[i][j].add(attr)
                        
                step = {
                    'i': df.index[i],
                    'j': df.index[j],
                    'i_decision': df.iloc[i][decision_col],
                    'j_decision': df.iloc[j][decision_col],
                    'attributes': list(matrix[i][j])
                }
                steps.append(step)
    
    return matrix, steps

def reduct(df, attributes, decision_col):
    matrix, _ = discernibility_matrix(df, attributes, decision_col)
    
    # Extract non-empty discernibility sets
    discernibility_sets = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j]:
                discernibility_sets.append(set(matrix[i][j]))
    
    # Get all possible subsets of attributes
    all_subsets = []
    for r in range(1, len(attributes) + 1):
        all_subsets.extend(combinations(attributes, r))
    
    # Find minimal reducts
    valid_reducts = []
    steps = []
    
    for subset in all_subsets:
        subset_set = set(subset)
        valid = True
        
        step_info = {
            'subset': list(subset),
            'checks': []
        }
        
        for disc_set in discernibility_sets:
            intersection = subset_set.intersection(disc_set)
            is_valid = len(intersection) > 0
            
            step_info['checks'].append({
                'disc_set': list(disc_set),
                'intersection': list(intersection),
                'is_valid': is_valid
            })
            
            if not is_valid:
                valid = False
                break
        
        step_info['is_valid'] = valid
        steps.append(step_info)
        
        if valid:
            valid_reducts.append(subset)
    
    # Filter out supersets
    minimal_reducts = []
    for r1 in valid_reducts:
        is_minimal = True
        r1_set = set(r1)
        
        for r2 in valid_reducts:
            if r1 != r2 and set(r2).issubset(r1_set):
                is_minimal = False
                break
                
        if is_minimal:
            minimal_reducts.append(list(r1))
    
    return minimal_reducts, steps, discernibility_sets

def run_reduct(file_path, all_columns, decision_col, decision_value, index_col=None):
    try:
        # Read the CSV
        df = pd.read_csv(file_path)
        
        # Set index if specified
        if index_col and index_col in df.columns:
            df.set_index(index_col, inplace=True)
        
        # Filter attributes (exclude decision column)
        attributes = [col for col in all_columns if col != decision_col]
        
        # Perform rough set analysis
        lower_approx, upper_approx, rough_steps, eq_classes = rough_approximations(df, attributes, decision_col, decision_value)
        reducts, reduct_steps, disc_sets = reduct(df, attributes, decision_col)
        
        # Prepare data for display
        lower_approx_list = list(lower_approx)
        upper_approx_list = list(upper_approx)
        
        # Convert DataFrame to HTML for display
        df_html = df.to_html(classes=['table', 'table-striped', 'table-bordered'])
        
        # Store equivalence classes in a more readable format
        eq_classes_display = {}
        for key, indices in eq_classes.items():
            # Convert attribute values tuple to a readable string
            key_str = ", ".join([f"{attr}={val}" for attr, val in zip(attributes, key)])
            eq_classes_display[key_str] = indices
        
        return {
            'df_html': df_html,
            'attributes': attributes,
            'decision_col': decision_col,
            'decision_value': decision_value,
            'lower_approx': lower_approx_list,
            'upper_approx': upper_approx_list,
            'reducts': reducts,
            'eq_classes': eq_classes_display,
            'rough_steps': rough_steps,
            'reduct_steps': reduct_steps,
            'disc_sets': disc_sets
        }
    except Exception as e:
        return {'error': str(e)}

def get_decision_values(file_path, decision_col):
    try:
        df = pd.read_csv(file_path)
        unique_values = df[decision_col].unique().tolist()
        return unique_values
    except Exception:
        return []