# System Threat Forecaster

A machine learning-based cybersecurity threat prediction system that analyzes system configurations and characteristics to forecast potential security threats. This project implements a comprehensive data preprocessing pipeline and multiple classification algorithms to identify systems at risk.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)


## 🎯 Overview

The System Threat Forecaster is designed to predict cybersecurity threats by analyzing various system attributes including hardware specifications, operating system details, security configurations, and temporal patterns. The system employs advanced feature engineering techniques and machine learning algorithms to provide accurate threat predictions.

## ✨ Features

- **Comprehensive Data Preprocessing**: Automated handling of missing values, outliers, and feature engineering
- **Multi-Algorithm Support**: Implementation of various ML algorithms including XGBoost, Logistic Regression, and ensemble methods
- **Feature Engineering**: Advanced techniques including version parsing, date extraction, and correlation analysis
- **Scalable Pipeline**: Modular preprocessing pipeline using scikit-learn transformers
- **Automated Feature Selection**: Implementation of SelectKBest and dimensionality reduction techniques

## 📊 Dataset

The system works with cybersecurity data containing the following key attributes:

- **System Information**: Machine specifications, RAM, disk capacity, display details
- **Operating System**: Version details, build information, locale settings
- **Security Software**: Antivirus versions, real-time protection status
- **Temporal Data**: Installation dates, OS dates
- **Target Variable**: Binary threat classification (0: No Threat, 1: Threat)

## 🚀 Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required Python packages (see Dependencies section)

### Setup

1. Clone the repository:
```bash
git clone 
cd system-threat-forecaster
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook mlp-f.ipynb
```

## 💻 Usage

### Basic Usage

1. **Data Loading**: Place your training data (`train.csv`) and test data (`test.csv`) in the input directory
2. **Run the Notebook**: Execute all cells in the Jupyter notebook sequentially
3. **Get Predictions**: The system will generate a `submission.csv` file with threat predictions

### Custom Configuration

The preprocessing pipeline can be customized by modifying the following functions:

- `drop_redundant()`: Remove highly correlated features
- `handle_outliers()`: Outlier detection and treatment
- `extract_date()`: Date feature engineering
- `split_version_column()`: Version string parsing

## 🔧 Data Preprocessing Pipeline

### 1. Data Cleaning
- **Machine ID Removal**: Drops non-predictive identifier columns
- **Redundancy Elimination**: Removes highly correlated features (correlation ≥ 0.8)
- **Zero Variance Filtering**: Eliminates features with no variability

### 2. Outlier Handling
- **IQR Method**: Caps outliers using Interquartile Range
- **Target Columns**: `TotalPhysicalRAMMB`, `PrimaryDisplayDiagonalInches`, `PrimaryDiskCapacityMB`, `SystemVolumeCapacityMB`

### 3. Feature Engineering
- **Date Extraction**: Converts date strings to day, month, year components
- **Version Parsing**: Splits version strings into numerical components
- **Categorical Encoding**: One-hot encoding for categorical variables
- **Numerical Scaling**: StandardScaler for numerical features

### 4. Advanced Processing
- **Missing Value Imputation**: Mean imputation for numerical, mode for categorical
- **Feature Selection**: SelectKBest and TruncatedSVD for dimensionality reduction

## 🤖 Model Architecture

### Supported Algorithms

1. **XGBoost Classifier** (Primary Model)
   - Gradient boosting framework
   - Hyperparameter optimization
   - Regularization techniques

2. **Logistic Regression**
   - Linear classification baseline
   - Feature importance analysis

3. **Feature Engineering Pipelines**
   - Base model comparison
   - SelectKBest feature selection
   - Truncated SVD dimensionality reduction

### Pipeline Structure

```python
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_cols),
    ('categorical', categorical_pipeline, categorical_cols)
])
```


## 📦 Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and preprocessing
- **xgboost**: Gradient boosting framework

### Visualization
- **matplotlib**: Basic plotting
- **seaborn**: Statistical data visualization

### Additional Tools
- **jupyter**: Interactive notebook environment

### Complete Requirements
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

## 🔄 Workflow

1. **Data Loading**: Import training and test datasets
2. **Exploratory Analysis**: Correlation analysis and feature inspection
3. **Preprocessing**: Apply cleaning and feature engineering pipeline
4. **Model Training**: Train XGBoost classifier with optimized parameters
5. **Prediction**: Generate predictions for test dataset
6. **Export**: Save results to submission file



### Model Tuning

Modify XGBoost parameters in the model initialization:

```python
model = XGBClassifier(
    colsample_bytree=0.9,
    gamma=0,
    learning_rate=0.1,
    max_depth=4,
    n_estimators=550,
    # Add your parameters
)
```


## 📝 Notes

- The system is designed for binary classification (threat/no threat)
- All preprocessing steps are applied consistently to both training and test data
- The pipeline is modular and can be extended for additional feature engineering
- Memory usage is optimized for large datasets through efficient data types
