import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import shap
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

# Set the style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

# Create output directories if they don't exist
os.makedirs('reports', exist_ok=True)
os.makedirs('plots/insights', exist_ok=True)

print("Generating Credit Risk Insights Report...")

# Load the data
try:
    # Load the preprocessed data
    X = pd.read_csv('data/preprocessed_data.csv', index_col=0)
    y = pd.read_csv('data/synthetic_risk_target.csv', index_col=0).squeeze()
    
    # Load the original processed data for reference
    original_data = pd.read_csv('data/processed_data_with_features.csv', index_col=0)
    
    print("Data loaded successfully.")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
except FileNotFoundError:
    print("Error: Data files not found. Please run the preprocessing and modeling scripts first.")
    exit(1)

# Load the model
try:
    # Check which model files exist
    model_files = os.listdir('models/')
    tuned_models = [f for f in model_files if f.startswith('tuned_')]
    
    if tuned_models:
        # Load the tuned model (assuming it's the best one)
        model_path = os.path.join('models', tuned_models[0])
        print(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        model_name = tuned_models[0].replace('tuned_', '').replace('_model.pkl', '').replace('_', ' ').title()
    else:
        # If no tuned model exists, try to load a random forest model
        print("No tuned model found. Loading Random Forest model...")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        model_name = "Random Forest"
    
    print(f"Model '{model_name}' loaded successfully.")
    
except (FileNotFoundError, IndexError) as e:
    print(f"Error loading model: {e}")
    print("Training a new Random Forest model...")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    model_name = "Random Forest"

# Load feature importance if available
try:
    feature_importance = pd.read_csv('results/interpretation/feature_importance_readable.csv')
    print("Feature importance data loaded successfully.")
except FileNotFoundError:
    print("Feature importance data not found. Will generate from model.")
    feature_importance = None

# Split the data for evaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# If XGBoost model, prepare the data
if 'XGBoost' in model_name:
    # Clean feature names for XGBoost
    clean_feature_names = []
    for col in X.columns:
        clean_name = col.replace('[', '_').replace(']', '_').replace('<', '_lt_').replace('>', '_gt_')
        clean_feature_names.append(clean_name)
    
    X_test_clean = X_test.copy()
    X_test_clean.columns = clean_feature_names
    
    # Make predictions
    y_pred = model.predict(X_test_clean)
    y_prob = model.predict_proba(X_test_clean)[:, 1]
else:
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

# Generate insights
print("\nGenerating insights...")

# 1. Overall model performance
print("Analyzing model performance...")
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

# Convert to DataFrame for easier handling
class_report_df = pd.DataFrame(class_report).transpose()

# 2. Risk distribution by demographic factors
print("Analyzing risk distribution by demographics...")

# Combine test data with predictions
test_indices = X_test.index
original_test = original_data.loc[test_indices].copy()
original_test['Predicted_Risk'] = y_pred
original_test['Actual_Risk'] = y_test
original_test['Risk_Probability'] = y_prob

# Create risk groups
original_test['Risk_Group'] = pd.cut(original_test['Risk_Probability'], 
                                    bins=[0, 0.25, 0.5, 0.75, 1.0],
                                    labels=['Very Low', 'Low', 'Medium', 'High'])

# 3. Risk by age groups
age_risk = original_test.groupby('Age_Group')['Risk_Probability'].mean().reset_index()
age_risk = age_risk.sort_values('Risk_Probability', ascending=False)

# 4. Risk by job
job_risk = original_test.groupby('Job')['Risk_Probability'].mean().reset_index()
job_risk = job_risk.sort_values('Risk_Probability', ascending=False)

# 5. Risk by purpose
purpose_risk = original_test.groupby('Purpose')['Risk_Probability'].mean().reset_index()
purpose_risk = purpose_risk.sort_values('Risk_Probability', ascending=False)

# 6. Risk by credit amount groups
credit_risk = original_test.groupby('Credit_Amount_Group')['Risk_Probability'].mean().reset_index()

# 7. Risk by duration groups
duration_risk = original_test.groupby('Duration_Group')['Risk_Probability'].mean().reset_index()

# 8. Risk by housing type
housing_risk = original_test.groupby('Housing')['Risk_Probability'].mean().reset_index()
housing_risk = housing_risk.sort_values('Risk_Probability', ascending=False)

# 9. Risk by saving accounts
saving_risk = original_test.groupby('Saving accounts')['Risk_Probability'].mean().reset_index()
saving_risk = saving_risk.sort_values('Risk_Probability', ascending=False)

# 10. Risk by checking account
checking_risk = original_test.groupby('Checking account')['Risk_Probability'].mean().reset_index()
checking_risk = checking_risk.sort_values('Risk_Probability', ascending=False)

# Generate visualizations
print("\nGenerating visualizations...")

# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Low Risk', 'High Risk'], 
           yticklabels=['Low Risk', 'High Risk'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('plots/insights/confusion_matrix.png')
plt.close()

# 2. Risk by Age Group
plt.figure(figsize=(10, 6))
sns.barplot(x='Age_Group', y='Risk_Probability', data=age_risk)
plt.title('Credit Risk by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Risk Probability')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('plots/insights/risk_by_age.png')
plt.close()

# 3. Risk by Job
plt.figure(figsize=(12, 6))
sns.barplot(x='Job', y='Risk_Probability', data=job_risk)
plt.title('Credit Risk by Job')
plt.xlabel('Job')
plt.ylabel('Average Risk Probability')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/insights/risk_by_job.png')
plt.close()

# 4. Risk by Purpose
plt.figure(figsize=(14, 6))
sns.barplot(x='Purpose', y='Risk_Probability', data=purpose_risk)
plt.title('Credit Risk by Loan Purpose')
plt.xlabel('Purpose')
plt.ylabel('Average Risk Probability')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/insights/risk_by_purpose.png')
plt.close()

# 5. Risk by Credit Amount Group
plt.figure(figsize=(10, 6))
sns.barplot(x='Credit_Amount_Group', y='Risk_Probability', data=credit_risk)
plt.title('Credit Risk by Credit Amount')
plt.xlabel('Credit Amount Group')
plt.ylabel('Average Risk Probability')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('plots/insights/risk_by_credit_amount.png')
plt.close()

# 6. Risk by Duration Group
plt.figure(figsize=(10, 6))
sns.barplot(x='Duration_Group', y='Risk_Probability', data=duration_risk)
plt.title('Credit Risk by Loan Duration')
plt.xlabel('Duration Group')
plt.ylabel('Average Risk Probability')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('plots/insights/risk_by_duration.png')
plt.close()

# 7. Risk by Housing Type
plt.figure(figsize=(10, 6))
sns.barplot(x='Housing', y='Risk_Probability', data=housing_risk)
plt.title('Credit Risk by Housing Type')
plt.xlabel('Housing Type')
plt.ylabel('Average Risk Probability')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('plots/insights/risk_by_housing.png')
plt.close()

# 8. Risk by Saving Account
plt.figure(figsize=(10, 6))
sns.barplot(x='Saving accounts', y='Risk_Probability', data=saving_risk)
plt.title('Credit Risk by Saving Account')
plt.xlabel('Saving Account')
plt.ylabel('Average Risk Probability')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('plots/insights/risk_by_saving.png')
plt.close()

# 9. Risk by Checking Account
plt.figure(figsize=(10, 6))
sns.barplot(x='Checking account', y='Risk_Probability', data=checking_risk)
plt.title('Credit Risk by Checking Account')
plt.xlabel('Checking Account')
plt.ylabel('Average Risk Probability')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('plots/insights/risk_by_checking.png')
plt.close()

# 10. Risk Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Risk_Group', data=original_test)
plt.title('Distribution of Risk Groups')
plt.xlabel('Risk Group')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plots/insights/risk_distribution.png')
plt.close()

# 11. Feature Importance
if feature_importance is not None:
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(x='SHAP_Importance', y='Feature_Readable', data=top_features)
    plt.title('Top 15 Features Influencing Credit Risk')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('plots/insights/feature_importance.png')
    plt.close()

# Generate HTML report
print("\nGenerating HTML report...")

# Create report content
report_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Credit Risk Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; text-align: center; }}
        h2 {{ color: #3498db; margin-top: 30px; }}
        h3 {{ color: #2980b9; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .insights {{ margin-bottom: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .recommendation {{ background-color: #e8f4f8; padding: 15px; border-left: 5px solid #3498db; margin-bottom: 10px; }}
        .image-container {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin: 20px 0; }}
        .image-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        .image-box {{ width: 45%; margin-bottom: 20px; }}
        .image-box img {{ width: 100%; }}
        .image-box p {{ text-align: center; font-weight: bold; margin-top: 5px; }}
        .footer {{ text-align: center; margin-top: 50px; font-size: 0.8em; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Credit Risk Analysis Report</h1>
        <p><strong>Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>
        <p><strong>Model:</strong> {model_name}</p>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <p>This report presents a comprehensive analysis of credit risk factors based on the {model_name} model. 
            The model achieves an accuracy of {class_report_df.loc['accuracy']['f1-score']:.2%} in predicting credit risk.</p>
            
            <p><strong>Key Findings:</strong></p>
            <ul>
                <li>The model correctly identifies {conf_matrix[1, 1]} high-risk cases out of {conf_matrix[1, 0] + conf_matrix[1, 1]} total high-risk cases.</li>
                <li>The precision for high-risk prediction is {class_report_df.loc['1', 'precision']:.2%}, indicating that {class_report_df.loc['1', 'precision']:.2%} of predicted high-risk cases are actually high risk.</li>
                <li>The recall for high-risk prediction is {class_report_df.loc['1', 'recall']:.2%}, indicating that {class_report_df.loc['1', 'recall']:.2%} of actual high-risk cases are correctly identified.</li>
            </ul>
        </div>
        
        <div class="insights">
            <h2>Key Insights</h2>
            
            <h3>1. Model Performance</h3>
            <div class="image-box" style="width: 60%; margin: 0 auto;">
                <img src="../plots/insights/confusion_matrix.png" alt="Confusion Matrix">
                <p>Confusion Matrix</p>
            </div>
            
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Low Risk (0)</th>
                    <th>High Risk (1)</th>
                </tr>
                <tr>
                    <td>Precision</td>
                    <td>{class_report_df.loc['0', 'precision']:.2%}</td>
                    <td>{class_report_df.loc['1', 'precision']:.2%}</td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td>{class_report_df.loc['0', 'recall']:.2%}</td>
                    <td>{class_report_df.loc['1', 'recall']:.2%}</td>
                </tr>
                <tr>
                    <td>F1-Score</td>
                    <td>{class_report_df.loc['0', 'f1-score']:.2%}</td>
                    <td>{class_report_df.loc['1', 'f1-score']:.2%}</td>
                </tr>
            </table>
            
            <h3>2. Key Risk Factors</h3>
            <div class="image-box" style="width: 80%; margin: 0 auto;">
                <img src="../plots/insights/feature_importance.png" alt="Feature Importance">
                <p>Top Factors Influencing Credit Risk</p>
            </div>
            
            <h3>3. Risk Distribution by Demographics</h3>
            
            <div class="image-container">
                <div class="image-box">
                    <img src="../plots/insights/risk_by_age.png" alt="Risk by Age">
                    <p>Credit Risk by Age Group</p>
                </div>
                <div class="image-box">
                    <img src="../plots/insights/risk_by_job.png" alt="Risk by Job">
                    <p>Credit Risk by Job</p>
                </div>
            </div>
            
            <h3>4. Risk Distribution by Loan Characteristics</h3>
            
            <div class="image-container">
                <div class="image-box">
                    <img src="../plots/insights/risk_by_purpose.png" alt="Risk by Purpose">
                    <p>Credit Risk by Loan Purpose</p>
                </div>
                <div class="image-box">
                    <img src="../plots/insights/risk_by_credit_amount.png" alt="Risk by Credit Amount">
                    <p>Credit Risk by Credit Amount</p>
                </div>
            </div>
            
            <div class="image-container">
                <div class="image-box">
                    <img src="../plots/insights/risk_by_duration.png" alt="Risk by Duration">
                    <p>Credit Risk by Loan Duration</p>
                </div>
                <div class="image-box">
                    <img src="../plots/insights/risk_distribution.png" alt="Risk Distribution">
                    <p>Distribution of Risk Groups</p>
                </div>
            </div>
            
            <h3>5. Risk Distribution by Financial Status</h3>
            
            <div class="image-container">
                <div class="image-box">
                    <img src="../plots/insights/risk_by_housing.png" alt="Risk by Housing">
                    <p>Credit Risk by Housing Type</p>
                </div>
                <div class="image-box">
                    <img src="../plots/insights/risk_by_saving.png" alt="Risk by Saving Account">
                    <p>Credit Risk by Saving Account</p>
                </div>
            </div>
            
            <div class="image-container">
                <div class="image-box" style="width: 60%; margin: 0 auto;">
                    <img src="../plots/insights/risk_by_checking.png" alt="Risk by Checking Account">
                    <p>Credit Risk by Checking Account</p>
                </div>
            </div>
        </div>
        
        <div class="insights">
            <h2>Key Findings</h2>
            
            <h3>1. Demographic Factors</h3>
            <p>Age groups {', '.join(age_risk.head(2)['Age_Group'].astype(str).values)} show the highest risk profiles, with average risk probabilities of {age_risk.head(2)['Risk_Probability'].values[0]:.2%} and {age_risk.head(2)['Risk_Probability'].values[1]:.2%} respectively.</p>
            
            <p>Job categories {', '.join(job_risk.head(2)['Job'].astype(str).values)} are associated with higher credit risk, with average risk probabilities of {job_risk.head(2)['Risk_Probability'].values[0]:.2%} and {job_risk.head(2)['Risk_Probability'].values[1]:.2%} respectively.</p>
            
            <h3>2. Loan Characteristics</h3>
            <p>Loan purposes {', '.join(purpose_risk.head(2)['Purpose'].values)} show the highest risk, with average risk probabilities of {purpose_risk.head(2)['Risk_Probability'].values[0]:.2%} and {purpose_risk.head(2)['Risk_Probability'].values[1]:.2%} respectively.</p>
            
            <p>Credit amounts in the {credit_risk['Risk_Probability'].idxmax()} category show the highest risk at {credit_risk['Risk_Probability'].max():.2%}.</p>
            
            <p>Loan durations in the {duration_risk['Risk_Probability'].idxmax()} category show the highest risk at {duration_risk['Risk_Probability'].max():.2%}.</p>
            
            <h3>3. Financial Status</h3>
            <p>Housing type {housing_risk.iloc[0]['Housing']} is associated with the highest risk at {housing_risk.iloc[0]['Risk_Probability']:.2%}.</p>
            
            <p>Saving account status {saving_risk.iloc[0]['Saving accounts']} shows the highest risk at {saving_risk.iloc[0]['Risk_Probability']:.2%}.</p>
            
            <p>Checking account status {checking_risk.iloc[0]['Checking account']} shows the highest risk at {checking_risk.iloc[0]['Risk_Probability']:.2%}.</p>
        </div>
        
        <div class="insights">
            <h2>Recommendations</h2>
            
            <div class="recommendation">
                <h3>1. Risk Assessment Strategies</h3>
                <p>Implement stricter verification processes for applicants with the following high-risk profiles:</p>
                <ul>
                    <li>Age groups: {', '.join(age_risk.head(2)['Age_Group'].values)}</li>
                    <li>Job categories: {', '.join(job_risk.head(2)['Job'].astype(str).values)}</li>
                    <li>Loan purposes: {', '.join(purpose_risk.head(2)['Purpose'].values)}</li>
                    <li>Checking account status: {checking_risk.iloc[0]['Checking account']}</li>
                </ul>
            </div>
            
            <div class="recommendation">
                <h3>2. Loan Product Optimization</h3>
                <p>Consider adjusting loan terms based on risk factors:</p>
                <ul>
                    <li>Offer shorter loan durations or lower amounts for high-risk profiles</li>
                    <li>Develop specialized loan products for different purposes with appropriate risk mitigation</li>
                    <li>Implement tiered interest rates based on risk profiles</li>
                </ul>
            </div>
            
            <div class="recommendation">
                <h3>3. Customer Relationship Management</h3>
                <p>Develop targeted strategies for different customer segments:</p>
                <ul>
                    <li>Provide financial education resources for high-risk groups</li>
                    <li>Implement early warning systems for potential defaults</li>
                    <li>Create incentives for maintaining good financial habits</li>
                </ul>
            </div>
            
            <div class="recommendation">
                <h3>4. Model Enhancement</h3>
                <p>Continuously improve the credit risk model:</p>
                <ul>
                    <li>Collect additional data on customer behavior and payment history</li>
                    <li>Regularly retrain the model with new data</li>
                    <li>Consider ensemble methods combining multiple models for better prediction</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
            <p>Credit Risk Analysis Report</p>
        </div>
    </div>
</body>
</html>
"""

# Save the HTML report
with open('reports/credit_risk_insights_report.html', 'w') as f:
    f.write(report_content)

# Generate a text summary report
print("Generating text summary report...")

text_report = f"""CREDIT RISK ANALYSIS SUMMARY REPORT
{'-' * 40}
Date: {datetime.now().strftime('%B %d, %Y')}
Model: {model_name}

EXECUTIVE SUMMARY
{'-' * 20}
This report presents a comprehensive analysis of credit risk factors based on the {model_name} model.
The model achieves an accuracy of {class_report_df.loc['accuracy']['f1-score']:.2%} in predicting credit risk.

KEY FINDINGS
{'-' * 20}
1. DEMOGRAPHIC FACTORS
   - Age groups {', '.join(age_risk.head(2)['Age_Group'].values)} show the highest risk profiles
   - Job categories {', '.join(job_risk.head(2)['Job'].astype(str).values)} are associated with higher credit risk

2. LOAN CHARACTERISTICS
   - Loan purposes {', '.join(purpose_risk.head(2)['Purpose'].values)} show the highest risk
   - Credit amounts in the {credit_risk['Risk_Probability'].idxmax()} category show the highest risk
   - Loan durations in the {duration_risk['Risk_Probability'].idxmax()} category show the highest risk

3. FINANCIAL STATUS
   - Housing type {housing_risk.iloc[0]['Housing']} is associated with the highest risk
   - Saving account status {saving_risk.iloc[0]['Saving accounts']} shows the highest risk
   - Checking account status {checking_risk.iloc[0]['Checking account']} shows the highest risk

RECOMMENDATIONS
{'-' * 20}
1. RISK ASSESSMENT STRATEGIES
   - Implement stricter verification processes for high-risk profiles
   - Focus on key risk factors identified in the analysis

2. LOAN PRODUCT OPTIMIZATION
   - Adjust loan terms based on risk factors
   - Develop specialized loan products for different purposes
   - Implement tiered interest rates based on risk profiles

3. CUSTOMER RELATIONSHIP MANAGEMENT
   - Provide financial education resources for high-risk groups
   - Implement early warning systems for potential defaults
   - Create incentives for maintaining good financial habits

4. MODEL ENHANCEMENT
   - Collect additional data on customer behavior and payment history
   - Regularly retrain the model with new data
   - Consider ensemble methods for better prediction

This report is accompanied by detailed visualizations in the 'plots/insights' directory
and a comprehensive HTML report in the 'reports' directory.
"""

# Save the text report
with open('reports/credit_risk_insights_summary.txt', 'w') as f:
    f.write(text_report)

print("\nInsights and reports generation complete!")
print("HTML report saved to 'reports/credit_risk_insights_report.html'")
print("Summary report saved to 'reports/credit_risk_insights_summary.txt'")
print("Visualizations saved to 'plots/insights' directory")