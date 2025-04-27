# Credit Risk Prediction

This project focuses on credit risk prediction using machine learning models. It consists of two main phases:

1. **Data Preprocessing**: Exploration, cleaning, missing value handling, outlier detection, and feature engineering
2. **Model Development**: Training multiple classification models, hyperparameter tuning, and comprehensive model evaluation

## Project Structure

```
├── german_credit_data.csv     # Original dataset
├── credit_risk_preprocessing.py  # Preprocessing script
├── data/                      # Directory for processed data
│   ├── preprocessed_data.csv  # Final preprocessed data
│   └── processed_data_with_features.csv  # Processed data with engineered features
└── plots/                     # Directory for visualization plots
    ├── missing_values_heatmap.png
    ├── categorical_distributions.png
    ├── numerical_distributions.png
    ├── numerical_boxplots.png
    ├── correlation_matrix.png
    ├── categorical_vs_credit_amount.png
    ├── pairplot.png
    └── credit_amount_by_purpose.png
```

## Preprocessing Steps

1. **Data Loading and Exploration**
   - Load the German Credit Data
   - Display basic information about the dataset
   - Check for missing values

2. **Exploratory Data Analysis**
   - Visualize distributions of categorical and numerical variables
   - Create boxplots to detect outliers
   - Generate correlation matrix
   - Analyze relationships between categorical variables and credit amount

3. **Data Preprocessing**
   - Handle missing values
     - For categorical variables: fill with the most frequent value
     - For numerical variables: fill with the median

4. **Feature Engineering**
   - Create age groups
   - Create credit amount groups
   - Create duration groups
   - One-hot encode categorical variables
   - Standardize numerical features

5. **Save Preprocessed Data**
   - Save the final preprocessed data
   - Save the processed data with engineered features

## Usage

### Data Preprocessing

To run the preprocessing script:

```bash
python credit_risk_preprocessing.py
```

This will generate the preprocessed data files in the `data` directory and visualization plots in the `plots` directory.

### Model Development

To train and evaluate the models:

```bash
python credit_risk_model.py
```

This will:
1. Train baseline models (Logistic Regression, Random Forest, XGBoost)
2. Evaluate models using various metrics (accuracy, precision, recall, F1-score)
3. Perform hyperparameter tuning on the best model
4. Generate performance visualizations in the `plots` directory
5. Save the optimized model in the `models` directory

## Next Steps

After model development, the project can proceed to:

1. Model deployment
2. Monitoring and maintenance
3. Additional feature engineering

# Credit Risk Prediction

This project implements a machine learning solution for predicting credit risk. It includes data preprocessing, model training, model interpretation using SHAP, a Streamlit web application for interactive predictions, and comprehensive reporting.

## Project Structure

- `credit_risk_preprocessing.py`: Preprocesses the German Credit Data
- `credit_risk_model.py`: Trains and evaluates machine learning models
- `model_interpretation.py`: Generates SHAP explanations for model predictions
- `credit_risk_app.py`: Streamlit web application for interactive predictions
- `generate_insights_report.py`: Creates comprehensive reports with insights and recommendations

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib
- shap
- streamlit
- pickle-mixin

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

Preprocess the German Credit Data:

```bash
python credit_risk_preprocessing.py
```

This script will:
- Load the German Credit Data
- Perform exploratory data analysis
- Handle missing values
- Create new features
- Save the preprocessed data

### 2. Model Training

Train and evaluate machine learning models:

```bash
python credit_risk_model.py
```

This script will:
- Load the preprocessed data
- Create a synthetic target variable (for demonstration)
- Train multiple models (Logistic Regression, Random Forest, XGBoost)
- Evaluate model performance
- Perform hyperparameter tuning
- Save the best model

### 3. Model Interpretation

Generate SHAP explanations for model predictions:

```bash
python model_interpretation.py
```

This script will:
- Load the best model and data
- Generate SHAP values
- Create various SHAP plots (summary, bar, dependence, waterfall, force)
- Save the plots and feature importance data

### 4. Streamlit Web Application

Run the Streamlit web application for interactive predictions:

```bash
streamlit run credit_risk_app.py
```

This application allows users to:
- Input personal and financial information
- Get credit risk predictions
- View SHAP explanations for the predictions
- Receive recommendations based on the risk assessment

### 5. Generate Insights Report

Create comprehensive reports with insights and recommendations:

```bash
python generate_insights_report.py
```

This script will:
- Analyze the model and predictions
- Generate visualizations of risk factors
- Create an HTML report with insights and recommendations
- Save a text summary report

## Results

The project generates various outputs:

- **Models**: Saved in the `models` directory
- **Plots**: Visualizations saved in the `plots` directory
  - Model performance plots in the root of `plots`
  - SHAP plots in `plots/shap`
  - Insight visualizations in `plots/insights`
- **Results**: Metrics and data saved in the `results` directory
- **Reports**: Comprehensive reports saved in the `reports` directory