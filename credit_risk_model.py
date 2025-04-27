import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, but continue if not available
try:
    import xgboost as xgb
    xgboost_available = True
except ImportError:
    print("XGBoost not installed. Skipping XGBoost model.")
    xgboost_available = False

# Set the style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

# Create output directories if they don't exist
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load the preprocessed data
print("Loading preprocessed data...")
try:
    # Try to load the preprocessed data with features
    data = pd.read_csv('data/preprocessed_data.csv', index_col=0)
    print("Loaded preprocessed data successfully.")
    
    # Load the original processed data
    original_data = pd.read_csv('data/processed_data_with_features.csv', index_col=0)
    
    # Since there's no target variable in the data, we'll create a synthetic one for demonstration
    print("Note: No target variable found in the data. Creating a synthetic target for demonstration.")
    
    # Create a synthetic target based on Credit amount and Duration
    # This is just for demonstration - in a real scenario, you would use the actual target
    np.random.seed(42)  # For reproducibility
    
    # Create a rule-based target: higher credit amount and longer duration = higher risk
    credit_normalized = (original_data['Credit amount'] - original_data['Credit amount'].min()) / \
                       (original_data['Credit amount'].max() - original_data['Credit amount'].min())
    duration_normalized = (original_data['Duration'] - original_data['Duration'].min()) / \
                         (original_data['Duration'].max() - original_data['Duration'].min())
    
    # Combine factors with some randomness
    risk_score = 0.7 * credit_normalized + 0.3 * duration_normalized + 0.1 * np.random.randn(len(original_data))
    
    # Convert to binary target (1 = high risk, 0 = low risk)
    y = (risk_score > risk_score.median()).astype(int)
    y.name = 'Risk'
    
    # Save the target for future use
    risk_df = pd.DataFrame(y)
    risk_df.to_csv('data/synthetic_risk_target.csv')
    
    print(f"Synthetic target variable distribution:\n{y.value_counts()}")
    
    # Prepare the feature matrix
    X = data
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
except FileNotFoundError:
    print("Error: Preprocessed data files not found. Please run the preprocessing script first.")
    exit(1)

# Split the data into training and testing sets
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Function to evaluate and print model performance
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Start timing
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # End timing
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate probabilities for ROC curve (if the model supports predict_proba)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_available = True
    except (AttributeError, IndexError):
        roc_available = False
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'plots/confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    
    # Plot ROC curve if available
    if roc_available:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f'plots/roc_curve_{model_name.replace(" ", "_").lower()}.png')
    
    # Return the model and metrics
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time
    }

# Train and evaluate baseline models
print("\nTraining baseline models...")

# Dictionary to store model results
model_results = {}

# 1. Logistic Regression
print("\nTraining Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
model_results['Logistic Regression'] = evaluate_model(log_reg, X_train, X_test, y_train, y_test, "Logistic Regression")

# 2. Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(random_state=42)
model_results['Random Forest'] = evaluate_model(rf, X_train, X_test, y_train, y_test, "Random Forest")

# 3. XGBoost (if available)
if xgboost_available:
    print("\nTraining XGBoost...")
    # XGBoost doesn't allow special characters in feature names
    # Create a copy of the data with cleaned feature names
    X_train_xgb = X_train.copy()
    X_test_xgb = X_test.copy()
    
    # Clean feature names for XGBoost
    clean_feature_names = []
    for col in X_train.columns:
        # Replace special characters with underscores
        clean_name = col.replace('[', '_').replace(']', '_').replace('<', '_lt_').replace('>', '_gt_')
        clean_feature_names.append(clean_name)
    
    # Set the cleaned feature names
    X_train_xgb.columns = clean_feature_names
    X_test_xgb.columns = clean_feature_names
    
    xgb_model = xgb.XGBClassifier(random_state=42)
    model_results['XGBoost'] = evaluate_model(xgb_model, X_train_xgb, X_test_xgb, y_train, y_test, "XGBoost")

# Compare baseline models
print("\nComparing baseline models:")
model_comparison = pd.DataFrame({
    'Model': list(model_results.keys()),
    'Accuracy': [results['accuracy'] for results in model_results.values()],
    'Precision': [results['precision'] for results in model_results.values()],
    'Recall': [results['recall'] for results in model_results.values()],
    'F1 Score': [results['f1'] for results in model_results.values()],
    'Training Time (s)': [results['training_time'] for results in model_results.values()]
})

print(model_comparison)

# Plot model comparison
plt.figure(figsize=(12, 8))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
model_comparison_plot = model_comparison.set_index('Model')[metrics]
model_comparison_plot.plot(kind='bar', figsize=(12, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/model_comparison.png')

# Identify the best performing model based on F1 score
best_model_name = model_comparison.loc[model_comparison['F1 Score'].idxmax(), 'Model']
print(f"\nBest performing model based on F1 score: {best_model_name}")

# Hyperparameter tuning for the best model
print("\nPerforming hyperparameter tuning for the best model...")

if best_model_name == 'Logistic Regression':
    print("Tuning Logistic Regression hyperparameters...")
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }
    best_model = LogisticRegression(random_state=42, max_iter=1000)

elif best_model_name == 'Random Forest':
    print("Tuning Random Forest hyperparameters...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    best_model = RandomForestClassifier(random_state=42)

elif best_model_name == 'XGBoost' and xgboost_available:
    print("Tuning XGBoost hyperparameters...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    best_model = xgb.XGBClassifier(random_state=42)

else:
    print(f"No hyperparameter tuning defined for {best_model_name}. Skipping.")
    param_grid = {}

# Perform grid search if param_grid is not empty
if param_grid:
    print("Starting GridSearchCV...")
    grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("\nBest hyperparameters:")
    print(grid_search.best_params_)
    
    # Evaluate the tuned model
    tuned_model = grid_search.best_estimator_
    tuned_results = evaluate_model(tuned_model, X_train, X_test, y_train, y_test, f"Tuned {best_model_name}")
    
    # Compare with the baseline model
    print("\nComparison between baseline and tuned model:")
    baseline_f1 = model_results[best_model_name]['f1']
    tuned_f1 = tuned_results['f1']
    improvement = ((tuned_f1 - baseline_f1) / baseline_f1) * 100
    print(f"Baseline {best_model_name} F1 Score: {baseline_f1:.4f}")
    print(f"Tuned {best_model_name} F1 Score: {tuned_f1:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    
    # Save the tuned model
    import joblib
    joblib.dump(tuned_model, f'models/tuned_{best_model_name.replace(" ", "_").lower()}_model.pkl')
    print(f"\nTuned model saved as 'models/tuned_{best_model_name.replace(" ", "_").lower()}_model.pkl'")

# Feature importance for the best model (if applicable)
if best_model_name in ['Random Forest', 'XGBoost']:
    print("\nCalculating feature importance...")
    
    if best_model_name == 'Random Forest':
        if param_grid:  # If we performed hyperparameter tuning
            feature_importances = tuned_model.feature_importances_
        else:
            feature_importances = model_results[best_model_name]['model'].feature_importances_
        
        # Use original feature names
        feature_names = X.columns
        
    elif best_model_name == 'XGBoost' and xgboost_available:
        if param_grid:  # If we performed hyperparameter tuning
            feature_importances = tuned_model.feature_importances_
        else:
            feature_importances = model_results[best_model_name]['model'].feature_importances_
        
        # Use cleaned feature names for XGBoost
        feature_names = clean_feature_names
    
    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance_df.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(15)  # Show top 15 features
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top 15 Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')

# Cross-validation of the best model
print("\nPerforming cross-validation on the best model...")
if param_grid:  # If we performed hyperparameter tuning
    cv_model = tuned_model
else:
    cv_model = model_results[best_model_name]['model']

cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring='f1_weighted')
print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean F1 score: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

# Save cross-validation results
cv_results = pd.DataFrame({
    'Fold': range(1, 6),
    'F1 Score': cv_scores
})
cv_results.to_csv('results/cross_validation_results.csv', index=False)

# Plot cross-validation results
plt.figure(figsize=(10, 6))
sns.barplot(x='Fold', y='F1 Score', data=cv_results)
plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean F1: {cv_scores.mean():.4f}')
plt.ylim(0, 1)
plt.title(f'Cross-Validation F1 Scores - {best_model_name}')
plt.legend()
plt.tight_layout()
plt.savefig('plots/cross_validation_scores.png')

print("\nModel development complete! Results and plots saved in the 'results' and 'plots' directories.")
print(f"Best model saved in the 'models' directory as 'tuned_{best_model_name.replace(' ', '_').lower()}_model.pkl'")