# Credit Risk Prediction Project - Technical Documentation

## Project Overview

This project implements a comprehensive machine learning solution for predicting credit risk. It follows a structured approach to data analysis, preprocessing, model development, interpretation, and deployment through an interactive web application. The system is designed to help financial institutions assess the risk associated with loan applications based on applicant characteristics and loan details.

## Project Structure

The project is organized into the following components:

```
/Credit Risk Prediction/
├── credit_risk_preprocessing.py  # Data preprocessing and feature engineering
├── credit_risk_model.py          # Model training and evaluation
├── model_interpretation.py       # Model interpretation using SHAP
├── generate_insights_report.py   # Generates insights and HTML report
├── credit_risk_app.py            # Streamlit web application
├── german_credit_data.csv        # Original dataset
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
├── data/                         # Processed and preprocessed data
├── models/                       # Trained model files
├── plots/                        # Visualization outputs
├── reports/                      # Generated reports
└── results/                      # Model results and interpretations
```

## Technologies and Libraries

### Core Libraries

- **pandas (≥1.3.0)**: Used for data manipulation and analysis. Handles all data loading, transformation, and feature engineering operations.

- **numpy (≥1.20.0)**: Provides support for numerical operations and array manipulations, particularly for creating synthetic target variables and data transformations.

- **scikit-learn (≥1.0.0)**: The primary machine learning library used for:
  - Data preprocessing (StandardScaler, OneHotEncoder, SimpleImputer)
  - Model implementation (LogisticRegression, RandomForestClassifier)
  - Model evaluation (classification_report, confusion_matrix, ROC curves)
  - Model selection (train_test_split, GridSearchCV, cross_val_score)

- **xgboost (≥1.5.0)**: Provides gradient boosting implementation for an additional high-performance model option.

### Visualization Libraries

- **matplotlib (≥3.4.0)**: Used for creating static, interactive, and animated visualizations, serving as the foundation for most plots.

- **seaborn (≥0.11.0)**: Built on matplotlib, provides a high-level interface for drawing attractive statistical graphics, used extensively for EDA and result visualization.

### Model Interpretation

- **shap (≥0.41.0)**: Implements SHAP (SHapley Additive exPlanations) values for model interpretation, providing insights into feature importance and impact on predictions.

### Web Application

- **streamlit (≥1.22.0)**: Powers the interactive web application, allowing users to input applicant information and receive risk predictions with explanations.

### Utilities

- **joblib (≥1.1.0)**: Used for model serialization (saving and loading models).
- **pickle-mixin (≥1.0.2)**: Provides additional serialization capabilities.

## Implementation Phases

### Phase 1: Data Preprocessing and Feature Engineering

**File: credit_risk_preprocessing.py**

This phase handles the initial data processing, exploratory data analysis, and feature engineering:

1. **Data Loading and Exploration**:
   - Loads the German Credit Data dataset
   - Analyzes basic statistics, data types, and missing values
   - Generates visualizations for understanding data distributions

2. **Missing Value Handling**:
   - Categorical variables: Imputed with most frequent value
   - Numerical variables: Imputed with median value

3. **Feature Engineering**:
   - Creates age groups (`<25`, `25-35`, `35-45`, `45-55`, `55-65`, `>65`)
   - Creates credit amount groups (`Very Low`, `Low`, `Medium`, `High`, `Very High`)
   - Creates loan duration groups (`<1yr`, `1-2yrs`, `2-3yrs`, `3-4yrs`, `>4yrs`)

4. **Feature Transformation**:
   - One-hot encodes categorical variables (including engineered categorical features)
   - Standardizes numerical features using StandardScaler

5. **Data Storage**:
   - Saves preprocessed data to `data/preprocessed_data.csv`
   - Saves processed data with engineered features to `data/processed_data_with_features.csv`

### Phase 2: Model Development and Evaluation

**File: credit_risk_model.py**

This phase focuses on building, training, and evaluating machine learning models:

1. **Data Preparation**:
   - Loads preprocessed data
   - Creates a synthetic target variable based on credit amount and duration (for demonstration)
   - Splits data into training and testing sets (80/20 split with stratification)

2. **Model Training**:
   - Implements multiple models:
     - Logistic Regression
     - Random Forest
     - XGBoost (if available)
   - Evaluates each model using accuracy, precision, recall, and F1-score
   - Generates confusion matrices and ROC curves

3. **Model Tuning**:
   - Performs hyperparameter tuning using GridSearchCV
   - Conducts cross-validation to ensure model robustness
   - Selects the best model based on performance metrics

4. **Model Comparison**:
   - Compares models based on multiple metrics
   - Visualizes performance differences
   - Saves the best model for future use

5. **Model Storage**:
   - Saves trained models to the `models/` directory
   - Stores evaluation results in the `results/` directory

### Phase 3: Model Interpretation

**File: model_interpretation.py**

This phase focuses on explaining the model's predictions using SHAP values:

1. **SHAP Analysis**:
   - Generates SHAP values to explain individual predictions
   - Creates summary plots showing overall feature importance
   - Produces dependence plots for top features

2. **Visualization**:
   - Creates SHAP summary plots
   - Generates feature importance bar plots
   - Produces waterfall plots for sample instances
   - Creates force plots for detailed instance explanations

3. **Interpretation Storage**:
   - Saves SHAP visualizations to `plots/shap/`
   - Stores feature importance data in `results/interpretation/`

### Phase 4: Insights Generation

**File: generate_insights_report.py**

This phase generates comprehensive insights and reports based on the model:

1. **Risk Analysis**:
   - Analyzes risk distribution across different demographic factors
   - Examines risk patterns by age groups, job categories, loan purposes, etc.
   - Identifies high-risk segments

2. **Visualization**:
   - Creates visualizations for risk distribution
   - Generates plots for risk by various factors
   - Produces a confusion matrix for model performance

3. **Report Generation**:
   - Creates an HTML report with all insights and visualizations
   - Includes executive summary, key findings, and recommendations
   - Saves the report to `reports/credit_risk_insights_report.html`

### Phase 5: Interactive Web Application

**File: credit_risk_app.py**

This phase implements an interactive Streamlit web application:

1. **User Interface**:
   - Provides input forms for applicant information
   - Includes sliders and dropdowns for all required fields
   - Displays prediction results with risk probability

2. **Prediction Pipeline**:
   - Processes user inputs to match the model's expected format
   - Applies the same preprocessing steps used during training
   - Generates risk predictions using the trained model

3. **Explanation**:
   - Uses SHAP values to explain predictions
   - Visualizes feature contributions to the risk score
   - Highlights the most influential factors

4. **Insights**:
   - Provides risk assessment based on prediction
   - Shows similar profiles from the dataset
   - Displays overall model performance metrics

## Technical Implementation Details

### Data Preprocessing Techniques

1. **Categorical Encoding**:
   - One-hot encoding is used for categorical variables
   - The first category is dropped to avoid multicollinearity

2. **Numerical Scaling**:
   - StandardScaler is applied to normalize numerical features
   - This ensures all features contribute equally to the model

3. **Feature Engineering**:
   - Binning continuous variables into meaningful categories
   - Creating interaction features for better model performance

### Model Selection and Evaluation

1. **Model Comparison Criteria**:
   - Primary metrics: Accuracy, Precision, Recall, F1-score
   - Secondary considerations: Training time, model complexity

2. **Cross-Validation**:
   - Stratified k-fold cross-validation to handle class imbalance
   - Ensures model robustness across different data subsets

3. **Hyperparameter Tuning**:
   - Grid search for optimal hyperparameters
   - Focus on balancing precision and recall for credit risk

### Model Interpretation Approach

1. **SHAP Values**:
   - Local explanations: Explain individual predictions
   - Global explanations: Understand overall feature importance

2. **Interpretation Methods**:
   - TreeExplainer for tree-based models (Random Forest, XGBoost)
   - LinearExplainer for linear models (Logistic Regression)
   - KernelExplainer as a fallback for other model types

### Web Application Architecture

1. **Caching Strategy**:
   - Model and data are cached using `@st.cache_resource`
   - Improves application performance for multiple users

2. **Prediction Pipeline**:
   - Real-time preprocessing of user inputs
   - Standardization using statistics from training data
   - One-hot encoding with the same categories as training data

3. **Visualization Components**:
   - Interactive SHAP force plots
   - Risk probability gauge
   - Feature importance charts

## Conclusion

This credit risk prediction project demonstrates a comprehensive approach to developing a machine learning solution for financial risk assessment. By combining robust preprocessing, model development, interpretation, and deployment into an interactive application, it provides a complete pipeline from raw data to actionable insights.

The modular structure allows for easy maintenance and extension, while the focus on model interpretation ensures that predictions are transparent and explainable - a critical requirement in the financial domain where decisions must be justified and understood.

The project showcases best practices in machine learning development, including:

- Thorough exploratory data analysis
- Careful feature engineering
- Rigorous model evaluation and selection
- Comprehensive model interpretation
- User-friendly application deployment

These elements combine to create a solution that not only predicts credit risk accurately but also provides valuable insights into the factors driving that risk, enabling better decision-making in loan approval processes.