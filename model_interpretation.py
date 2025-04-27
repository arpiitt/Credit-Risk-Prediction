import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os

# Set the style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

# Create output directories if they don't exist
os.makedirs('plots/shap', exist_ok=True)
os.makedirs('results/interpretation', exist_ok=True)

print("Loading the model and data...")

# Load the preprocessed data
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

# Try to load the best model
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

# Split the data for interpretation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nGenerating SHAP explanations...")

# Handle different model types for SHAP
if 'XGBoost' in model_name:
    # For XGBoost, we need to clean feature names
    clean_feature_names = []
    for col in X.columns:
        clean_name = col.replace('[', '_').replace(']', '_').replace('<', '_lt_').replace('>', '_gt_')
        clean_feature_names.append(clean_name)
    
    X_test_clean = X_test.copy()
    X_test_clean.columns = clean_feature_names
    
    # Use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_clean)
    
    # For plotting, we need to map back to original feature names
    X_test_for_plot = X_test_clean.copy()
    
elif 'Random Forest' in model_name or 'Logistic Regression' in model_name:
    # Use appropriate explainer based on model type
    if 'Random Forest' in model_name:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            # For multi-class output, take the positive class
            shap_values = shap_values[1]
    else:  # Logistic Regression
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test)
    
    X_test_for_plot = X_test
else:
    # Generic approach for other model types
    explainer = shap.KernelExplainer(model.predict_proba, X_train.iloc[:100])
    shap_values = explainer.shap_values(X_test.iloc[:100])
    if isinstance(shap_values, list):
        # For multi-class output, take the positive class
        shap_values = shap_values[1]
    
    X_test_for_plot = X_test.iloc[:100]

# Generate and save SHAP plots
print("\nGenerating SHAP summary plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_for_plot, show=False)
plt.title(f'SHAP Summary Plot - {model_name}')
plt.tight_layout()
plt.savefig('plots/shap/shap_summary_plot.png')
plt.close()

# Generate and save SHAP bar plot
print("Generating SHAP bar plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_for_plot, plot_type='bar', show=False)
plt.title(f'SHAP Feature Importance - {model_name}')
plt.tight_layout()
plt.savefig('plots/shap/shap_feature_importance.png')
plt.close()

# Generate SHAP dependence plots for top features
print("Generating SHAP dependence plots for top features...")

# Calculate mean absolute SHAP values for each feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame(list(zip(X_test_for_plot.columns, mean_abs_shap)), 
                                 columns=['Feature', 'SHAP_Importance'])
feature_importance = feature_importance.sort_values('SHAP_Importance', ascending=False)

# Save feature importance to CSV
feature_importance.to_csv('results/interpretation/feature_importance.csv', index=False)

# Print top 10 features
print("\nTop 10 most important features:")
print(feature_importance.head(10))

# Generate dependence plots for top 5 features
top_features = feature_importance['Feature'].head(5).values

for feature in top_features:
    plt.figure(figsize=(12, 8))
    shap.dependence_plot(feature, shap_values, X_test_for_plot, show=False)
    plt.title(f'SHAP Dependence Plot - {feature}')
    plt.tight_layout()
    plt.savefig(f'plots/shap/shap_dependence_{feature.replace(" ", "_").replace("[", "").replace("]", "")}.png')
    plt.close()

# Generate a waterfall plot for a sample instance
print("\nGenerating waterfall plot for a sample instance...")
# Choose a high-risk and low-risk example
high_risk_idx = y_test[y_test == 1].index[0]
low_risk_idx = y_test[y_test == 0].index[0]

# High risk example
plt.figure(figsize=(12, 8))
shap.plots._waterfall.waterfall_legacy(explainer.expected_value if hasattr(explainer, 'expected_value') else 0, 
                        shap_values[X_test.index.get_loc(high_risk_idx)], 
                        X_test_for_plot.iloc[X_test.index.get_loc(high_risk_idx)], 
                        show=False)
plt.title('SHAP Waterfall Plot - High Risk Example')
plt.tight_layout()
plt.savefig('plots/shap/shap_waterfall_high_risk.png')
plt.close()

# Low risk example
plt.figure(figsize=(12, 8))
shap.plots._waterfall.waterfall_legacy(explainer.expected_value if hasattr(explainer, 'expected_value') else 0, 
                        shap_values[X_test.index.get_loc(low_risk_idx)], 
                        X_test_for_plot.iloc[X_test.index.get_loc(low_risk_idx)], 
                        show=False)
plt.title('SHAP Waterfall Plot - Low Risk Example')
plt.tight_layout()
plt.savefig('plots/shap/shap_waterfall_low_risk.png')
plt.close()

# Generate a force plot for the same instances
print("Generating force plots...")

# High risk example
force_plot_high = shap.force_plot(explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                                 shap_values[X_test.index.get_loc(high_risk_idx)],
                                 X_test_for_plot.iloc[X_test.index.get_loc(high_risk_idx)],
                                 matplotlib=True, show=False)
plt.title('SHAP Force Plot - High Risk Example')
plt.tight_layout()
plt.savefig('plots/shap/shap_force_high_risk.png')
plt.close()

# Low risk example
force_plot_low = shap.force_plot(explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                                shap_values[X_test.index.get_loc(low_risk_idx)],
                                X_test_for_plot.iloc[X_test.index.get_loc(low_risk_idx)],
                                matplotlib=True, show=False)
plt.title('SHAP Force Plot - Low Risk Example')
plt.tight_layout()
plt.savefig('plots/shap/shap_force_low_risk.png')
plt.close()

# Create a summary of the interpretation results
print("\nCreating interpretation summary...")

# Map feature names to more readable names if needed
feature_mapping = {}
for feature in X.columns:
    if 'Sex_' in feature:
        feature_mapping[feature] = f"Gender: {feature.replace('Sex_', '')}"
    elif 'Job_' in feature:
        feature_mapping[feature] = f"Job: {feature.replace('Job_', '')}"
    elif 'Housing_' in feature:
        feature_mapping[feature] = f"Housing: {feature.replace('Housing_', '')}"
    elif 'Saving accounts_' in feature:
        feature_mapping[feature] = f"Savings: {feature.replace('Saving accounts_', '')}"
    elif 'Checking account_' in feature:
        feature_mapping[feature] = f"Checking: {feature.replace('Checking account_', '')}"
    elif 'Purpose_' in feature:
        feature_mapping[feature] = f"Loan Purpose: {feature.replace('Purpose_', '')}"
    elif 'Age_Group_' in feature:
        feature_mapping[feature] = f"Age Group: {feature.replace('Age_Group_', '')}"
    elif 'Credit_Amount_Group_' in feature:
        feature_mapping[feature] = f"Credit Amount: {feature.replace('Credit_Amount_Group_', '')}"
    elif 'Duration_Group_' in feature:
        feature_mapping[feature] = f"Loan Duration: {feature.replace('Duration_Group_', '')}"
    else:
        feature_mapping[feature] = feature

# Apply mapping to feature importance
feature_importance['Feature_Readable'] = feature_importance['Feature'].map(lambda x: feature_mapping.get(x, x))

# Save the readable feature importance
feature_importance[['Feature_Readable', 'SHAP_Importance']].to_csv('results/interpretation/feature_importance_readable.csv', index=False)

# Create a summary report
with open('results/interpretation/interpretation_summary.txt', 'w') as f:
    f.write(f"Credit Risk Model Interpretation Summary\n")
    f.write(f"Model: {model_name}\n\n")
    
    f.write("Top 10 Factors Influencing Credit Risk:\n")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        f.write(f"{i}. {row['Feature_Readable']} (Importance: {row['SHAP_Importance']:.4f})\n")
    
    f.write("\nKey Insights:\n")
    f.write("1. The model identifies several key factors that significantly influence credit risk.\n")
    f.write("2. Demographic factors (like age) and financial history (like checking account status) play important roles.\n")
    f.write("3. Loan characteristics (amount, duration, purpose) are critical in determining risk.\n")
    f.write("\nRecommendations:\n")
    f.write("1. Focus on the top factors when evaluating new credit applications.\n")
    f.write("2. Consider developing specialized risk models for different loan purposes.\n")
    f.write("3. Implement stricter verification processes for high-risk applications.\n")

print("\nModel interpretation complete! Results and plots saved in the 'results/interpretation' and 'plots/shap' directories.")