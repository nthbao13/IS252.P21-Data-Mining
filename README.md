# Course Project: Custom Machine Learning Toolkit

## 🎯 Project Objectives
The goals of this project are:
- To develop a custom machine learning toolkit with algorithms built entirely from scratch (without relying on core machine learning libraries like scikit-learn).
- To create a Flask-based web interface for users to upload data, apply algorithms, and view results.
- To enhance programming skills and deepen understanding of machine learning algorithms such as ID3 Decision Tree, K-means, Kohonen SOM, Naive Bayes, Association Rules, and Feature Reduction.
- To practice software project management through structured directory and code organization.

---

## 📋 Project Content
### 1. Project Structure
The project is organized as follows:
```
Machine Learning Toolkit/
├── algorithm/              # Custom-built algorithms
│   ├── __pycache__/
│   ├── association_rules.py   # Association rule mining algorithm
│   ├── decision_tree_id3.py   # ID3 decision tree algorithm
│   ├── kmeans.py             # K-means clustering algorithm
│   ├── kohonen.py            # Kohonen Self-Organizing Map algorithm
│   ├── naive_bayes.py        # Naive Bayes classifier
│   ├── preprocess.py         # Data preprocessing module
│   └── reduct.py             # Feature reduction algorithm
├── data/                   # Directory for input data
├── results/                # Directory for output results
├── routes/                 # Flask route handlers
├── static/                 # Static files (CSS, JS, images)
├── templates/              # HTML templates
├── uploads/                # Directory for uploaded files
├── utils/                  # Utility modules
│   ├── __pycache__/
│   ├── __init__.py
│   ├── decorators.py        # Custom Flask decorators
│   └── file_utils.py        # File handling utilities
└── app.py                  # Main Flask application file
```

### 2. Implemented Algorithms
- **ID3 Decision Tree**: Constructs a decision tree based on information gain and gini index.
- **K-means**: Clusters data into K groups using Euclidean distance.
- **Kohonen SOM**: Visualizes data using a self-organizing map.
- **Naive Bayes**: Performs classification based on independent feature probabilities.
- **Association Rules**: Identifies association rules in data (e.g., Apriori algorithm).
- **Feature Reduction**: Manually reduces data dimensionality.

### 3. Technologies Used
- **Programming Language**: Python 3.8+
- **Framework**: Flask
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn (support only), pydotplus, IPython
- **Tools**: Git, IDE (e.g., PyCharm/VS Code)

---

## ⚙️ Installation and Running Guide
### System Requirements
- Python 3.8 or higher.
- pip (Python package manager).

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/nthbao13/IS252.P21-Data-Mining.git
   cd `{url for this project}`
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Access the web interface at: `http://localhost:5000`.

### Usage Instructions
- Upload a dataset .CSV via the web interface.
- Select the desired algorithm and click "Run".
- View the results or download the output file.

---

## 📊 Expected Results
- A fully functional web application allowing users to apply machine learning algorithms.
- Visualized results (charts, tables) for data analysis.
- A detailed report on algorithm performance (if performance metrics are measured).

---

## 🛠️ Testing Guide
- Use sample datasets (place in the `data/` directory).
- Experiment with different parameters (e.g., number of clusters in K-means, support threshold in Association Rules).
- Record results and compare with theoretical expectations.

---

## 📧 Contact
- **Support**: [Nguyen Thai Bao] - [22520112@gm.uit.com]
- **Feedback**: Send an email or open an issue on the repository.

---

## ❤️ Acknowledgments
Thank you to teacher Mai Xuan Hung and teammates for support during the project!