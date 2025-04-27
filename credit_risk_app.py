import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os
import pickle

# Set page configuration
st.set_page_config(page_title="Credit Risk Prediction App", page_icon="💰", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E88E5;
    text-align: center;
}
.sub-header {
    font-size: 1.5rem;
    color: #424242;
}
.risk-high {
    color: #D32F2F;
    font-weight: bold;
    font-size: 1.2rem;
}
.risk-low {
    color: #388E3C;
    font-weight: bold;
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Credit Risk Prediction</p>', unsafe_allow_html=True)
st.markdown('This application predicts the credit risk of loan applicants based on their personal and financial information.')

# Load the model and necessary data
@st.cache_resource
def load_model_and_data():
    try:
        # Check which model files exist
        model_files = os.listdir('models/')
        tuned_models = [f for f in model_files if f.startswith('tuned_')]
        
        if tuned_models:
            # Load the tuned model (assuming it's the best one)
            model_path = os.path.join('models', tuned_models[0])
            model = joblib.load(model_path)
            model_name = tuned_models[0].replace('tuned_', '').replace('_model.pkl', '').replace('_', ' ').title()
        else:
            # If no tuned model exists, try to load a random forest model
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=42)
            model_name = "Random Forest"
            st.warning("No tuned model found. Using a default Random Forest model.")
        
        # Load the preprocessed data for reference
        X = pd.read_csv('data/preprocessed_data.csv', index_col=0)
        original_data = pd.read_csv('data/processed_data_with_features.csv', index_col=0)
        
        # Load feature importance if available
        try:
            feature_importance = pd.read_csv('results/interpretation/feature_importance_readable.csv')
        except FileNotFoundError:
            feature_importance = None
        
        return model, model_name, X, original_data, feature_importance
    
    except Exception as e:
        st.error(f"Error loading model and data: {e}")
        return None, None, None, None, None

# Load model and data
model, model_name, X, original_data, feature_importance = load_model_and_data()

if model is None:
    st.error("Failed to load the model. Please make sure you've run the model training script.")
    st.stop()

# Create SHAP explainer
@st.cache_resource
def get_explainer(_model, X):
    if 'XGBoost' in model_name:
        # Clean feature names for XGBoost
        clean_feature_names = []
        for col in X.columns:
            clean_name = col.replace('[', '_').replace(']', '_').replace('<', '_lt_').replace('>', '_gt_')
            clean_feature_names.append(clean_name)
        
        X_clean = X.copy()
        X_clean.columns = clean_feature_names
        return shap.TreeExplainer(model), X_clean
    
    elif 'Random Forest' in model_name:
        return shap.TreeExplainer(model), X
    
    else:  # Logistic Regression or other
        return shap.LinearExplainer(model, X.iloc[:100]), X

# Get explainer
explainer, X_for_shap = get_explainer(model, X)

# Sidebar for user inputs
st.sidebar.markdown('<p class="sub-header">Applicant Information</p>', unsafe_allow_html=True)

# Get unique values for categorical features from original data
unique_sex = original_data['Sex'].unique()
unique_job = original_data['Job'].unique()
unique_housing = original_data['Housing'].unique()
unique_saving = original_data['Saving accounts'].unique()
unique_checking = original_data['Checking account'].unique()
unique_purpose = original_data['Purpose'].unique()

# User inputs
with st.sidebar.form("user_inputs"):
    # Personal information
    st.markdown("**Personal Information**")
    age = st.slider("Age", min_value=18, max_value=100, value=35)
    sex = st.selectbox("Gender", options=unique_sex)
    job = st.selectbox("Job", options=unique_job)
    
    # Financial information
    st.markdown("**Financial Information**")
    housing = st.selectbox("Housing", options=unique_housing)
    saving_account = st.selectbox("Saving Account", options=unique_saving)
    checking_account = st.selectbox("Checking Account", options=unique_checking)
    
    # Loan information
    st.markdown("**Loan Information**")
    credit_amount = st.slider("Credit Amount (€)", min_value=100, max_value=20000, value=5000, step=100)
    loan_duration = st.slider("Loan Duration (months)", min_value=6, max_value=72, value=24, step=6)
    loan_purpose = st.selectbox("Loan Purpose", options=unique_purpose)
    
    # Submit button
    submitted = st.form_submit_button("Predict Risk")

# Main content area
if submitted:
    # Create a dataframe with user inputs
    user_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Job': [job],
        'Housing': [housing],
        'Saving accounts': [saving_account],
        'Checking account': [checking_account],
        'Credit amount': [credit_amount],
        'Duration': [loan_duration],
        'Purpose': [loan_purpose]
    })
    
    # Display the user inputs
    st.markdown('<p class="sub-header">Applicant Details</p>', unsafe_allow_html=True)
    st.table(user_data.T.rename(columns={0: 'Value'}))
    
    # Create age group, credit amount group, and duration group
    user_data['Age_Group'] = pd.cut(user_data['Age'], 
                                   bins=[0, 25, 35, 45, 55, 65, 100],
                                   labels=['<25', '25-35', '35-45', '45-55', '55-65', '>65'])
    
    user_data['Credit_Amount_Group'] = pd.cut(user_data['Credit amount'],
                                             bins=[0, 1000, 2000, 5000, 10000, 20000],
                                             labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    user_data['Duration_Group'] = pd.cut(user_data['Duration'],
                                        bins=[0, 12, 24, 36, 48, 72],
                                        labels=['<1yr', '1-2yrs', '2-3yrs', '3-4yrs', '>4yrs'])
    
    # One-hot encode the user data
    # Get the column names from the training data
    categorical_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 
                       'Age_Group', 'Credit_Amount_Group', 'Duration_Group']
    
    # Create a DataFrame with all possible categories from the training data
    user_encoded = pd.DataFrame(index=[0])
    
    # For each categorical column in the original data
    for col in categorical_cols:
        # Get all unique values in the training data for this column
        unique_values = original_data[col].unique()
        
        # For each unique value, create a binary column
        for val in unique_values:
            col_name = f"{col}_{val}"
            # Check if this value is present in the user data
            if col in user_data.columns and user_data[col].iloc[0] == val:
                user_encoded[col_name] = 1
            else:
                user_encoded[col_name] = 0
    
    # Add the numerical columns (standardized)
    numerical_cols = ['Age', 'Credit amount', 'Duration']
    for col in numerical_cols:
        # Get mean and std from original data for standardization
        mean_val = original_data[col].mean()
        std_val = original_data[col].std()
        
        # Standardize the user input
        user_encoded[col] = (user_data[col].iloc[0] - mean_val) / std_val
    
    # Ensure all columns from the training data are present
    for col in X.columns:
        if col not in user_encoded.columns:
            user_encoded[col] = 0
    
    # Reorder columns to match the training data
    user_encoded = user_encoded[X.columns]
    
    # Make prediction
    if 'XGBoost' in model_name:
        # Clean feature names for XGBoost
        clean_feature_names = []
        for col in X.columns:
            clean_name = col.replace('[', '_').replace(']', '_').replace('<', '_lt_').replace('>', '_gt_')
            clean_feature_names.append(clean_name)
        
        user_encoded_xgb = user_encoded.copy()
        user_encoded_xgb.columns = clean_feature_names
        prediction = model.predict(user_encoded_xgb)[0]
        prediction_proba = model.predict_proba(user_encoded_xgb)[0]
    else:
        prediction = model.predict(user_encoded)[0]
        prediction_proba = model.predict_proba(user_encoded)[0]
    
    # Display prediction
    st.markdown('<p class="sub-header">Risk Prediction</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.markdown('<p class="risk-high">High Risk</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="risk-low">Low Risk</p>', unsafe_allow_html=True)
    
    with col2:
        st.write(f"Confidence: {prediction_proba[prediction]:.2%}")
    
    # Create a progress bar for the risk probability
    st.progress(prediction_proba[1])
    st.write(f"Risk Probability: {prediction_proba[1]:.2%}")
    
    # SHAP explanation
    st.markdown('<p class="sub-header">Explanation</p>', unsafe_allow_html=True)
    
    # Get SHAP values for the user input
    if 'XGBoost' in model_name:
        shap_values = explainer.shap_values(user_encoded_xgb)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, take the positive class
    else:
        shap_values = explainer.shap_values(user_encoded)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]  # For binary classification, take the positive class
    
    # Create a force plot
    st.write("This chart shows how each factor contributes to the prediction:")
    
    # Convert to matplotlib figure for Streamlit
    fig, ax = plt.subplots(figsize=(10, 3))
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]  # For binary classification
    
    shap.force_plot(expected_value, 
                   shap_values[0], 
                   user_encoded if 'XGBoost' not in model_name else user_encoded_xgb,
                   matplotlib=True,
                   show=False,
                   figsize=(10, 3))
    st.pyplot(fig)
    
    # Display top factors
    st.write("Top factors influencing this prediction:")
    
    # Get the absolute SHAP values
    abs_shap_values = np.abs(shap_values[0])
    
    # Get the feature names and their SHAP values
    feature_names = X.columns if 'XGBoost' not in model_name else clean_feature_names
    feature_shap = list(zip(feature_names, abs_shap_values))
    
    # Sort by absolute SHAP value
    feature_shap.sort(key=lambda x: x[1], reverse=True)
    
    # Display top 5 features
    top_features = feature_shap[:5]
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(top_features))
    feature_names = [f[0] for f in top_features]
    feature_values = [f[1] for f in top_features]
    
    ax.barh(y_pos, feature_values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('SHAP Value (impact on prediction)')
    ax.set_title('Top 5 Factors')
    
    st.pyplot(fig)
    
    # Recommendations based on the prediction
    st.markdown('<p class="sub-header">Recommendations</p>', unsafe_allow_html=True)
    
    if prediction == 1:  # High risk
        st.write("Based on the prediction, here are some recommendations:")
        st.write("1. Consider reducing the loan amount or duration to lower the risk.")
        st.write("2. Provide additional collateral or guarantees.")
        st.write("3. Improve your financial profile by increasing savings or checking account balances.")
    else:  # Low risk
        st.write("Based on the prediction, you have a good credit profile. Here are some tips:")
        st.write("1. You may qualify for better interest rates or terms.")
        st.write("2. Consider exploring other financial products that may benefit you.")
        st.write("3. Maintain your current financial habits to keep your good credit standing.")

else:
    # Display model information and feature importance
    st.markdown('<p class="sub-header">Model Information</p>', unsafe_allow_html=True)
    st.write(f"Model: {model_name}")
    
    # Display feature importance if available
    if feature_importance is not None:
        st.markdown('<p class="sub-header">Key Factors Influencing Credit Risk</p>', unsafe_allow_html=True)
        
        # Create a bar chart of feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = feature_importance.head(10)
        ax.barh(top_features['Feature_Readable'], top_features['SHAP_Importance'])
        ax.set_xlabel('Importance')
        ax.invert_yaxis()  # To display the most important feature at the top
        ax.set_title('Top 10 Factors Influencing Credit Risk')
        st.pyplot(fig)
        
        st.write("These factors are most influential in determining credit risk. Consider them carefully when applying for a loan.")
    
    # Instructions
    st.markdown('<p class="sub-header">How to Use This App</p>', unsafe_allow_html=True)
    st.write("1. Enter your personal and financial information in the sidebar.")
    st.write("2. Click 'Predict Risk' to get your credit risk assessment.")
    st.write("3. Review the explanation to understand what factors influence your risk score.")
    st.write("4. Follow the recommendations to improve your credit profile if needed.")

# Footer
st.markdown("---")
st.markdown("Credit Risk Prediction App | Developed with Streamlit and SHAP")