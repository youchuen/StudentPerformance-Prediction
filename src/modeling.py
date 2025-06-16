import os
import pandas as pd
import numpy as np
import argparse
import joblib
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

from src.config import (
    SEED,
    TEST_SIZE,
    GRID_K_FOLD,
    LR_MAXITER,
    RF_N_EST,
    RF_GRID,
    XGB_EVAL_MT,
    XGB_GRID,
    DEFAULT_TARGET,
    SELECTED_FEATURES,
    BIN_COLS
)

# Selected Features
selected_features = SELECTED_FEATURES

# Custom transformer for threshold binning
class ThresholdBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, cols, threshold=1):
        self.cols = cols
        self.threshold = threshold
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xb = X.copy()
        for c in self.cols:
            Xb[c] = (Xb[c] >= self.threshold).astype(int)
        return Xb

# Load and filter data
def load_data(path):
    df = pd.read_csv(path)
    df[DEFAULT_TARGET] = LabelEncoder().fit_transform(df[DEFAULT_TARGET])
    # Validate selected features against available columns
    available = [f for f in selected_features if f in df.columns]
    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        print(f"Warning: the following selected features were not found and will be skipped: {missing}")
    return df[available + ['Target']]

# Build preprocessors: base and with threshold binning
def build_preprocessors(X):
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    bin_cols = BIN_COLS

    num_pipeline = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    preproc_base = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ], remainder='drop')

    # Binned preprocessor as a single ColumnTransformer
    numeric_no_bin = [c for c in numeric_cols if c not in bin_cols]
    preproc_binned = ColumnTransformer([
        ('binarize', ThresholdBinarizer(bin_cols), bin_cols),
        ('num', num_pipeline, numeric_no_bin),
        ('cat', cat_pipeline, categorical_cols)
    ], remainder='drop')
    return preproc_base, preproc_binned

# Find best model by ROC AUC
def find_best_model(reports_dir):
    """Find the model with highest ROC AUC score from the reports."""
    best_model_name = None
    best_roc_auc = 0
    
    print("\nEvaluating all trained models...")
    print("-" * 50)
    
    for file_name in os.listdir(reports_dir):
        if file_name.endswith('_report.csv'):
            model_name = file_name.replace('_report.csv', '')
            report_path = os.path.join(reports_dir, file_name)
            
            try:
                report_df = pd.read_csv(report_path, index_col=0)
                roc_auc = float(report_df.loc['roc_auc', 'roc_auc'])
                
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    best_model_name = model_name
                    
                print(f"{model_name}: ROC AUC = {roc_auc:.4f}")
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
    
    print("-" * 50)
    print(f"BEST MODEL FOUND: {best_model_name}")
    print(f"BEST ROC AUC SCORE: {best_roc_auc:.4f}")
    print("-" * 50)
    return best_model_name, best_roc_auc

# Save additional components for Streamlit app
def save_streamlit_components(output_dir, X_train, label_encoder, best_model_name, best_model):
    """Save all components needed by the Streamlit app."""
    
    print(f"\n[BEST MODEL IDENTIFIED]: {best_model_name}")
    print("=" * 60)
    print("Saving best model components for Streamlit application...")
    
    # Save label encoder
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
    print("Saved label encoder")
    
    # Save feature columns
    feature_columns = X_train.columns.tolist()
    joblib.dump(feature_columns, os.path.join(output_dir, 'feature_columns.pkl'))
    print("Saved feature columns")
    
    # Save numerical and categorical columns
    numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
    joblib.dump(numeric_cols, os.path.join(output_dir, 'numerical_columns.pkl'))
    joblib.dump(categorical_cols, os.path.join(output_dir, 'categorical_columns.pkl'))
    print("Saved column type information")
    
    # Save class names
    class_names = ['Dropout', 'Enrolled', 'Graduate']
    joblib.dump(class_names, os.path.join(output_dir, 'class_names.pkl'))
    print("Saved class names")
    
    # Save best model name to text file for easy reference
    with open(os.path.join(output_dir, 'best_model_name.txt'), 'w') as f:
        f.write(best_model_name)
    print(f"Saved best model name: {best_model_name}")
    
    # Copy best model for easy access
    best_model_path = os.path.join(output_dir, f"{best_model_name}_model.pkl")
    joblib.dump(best_model, os.path.join(output_dir, 'best_model.pkl'))
    print(f"Saved best model ({best_model_name}) as best_model.pkl")
    
    # Create feature importance if possible
    try:
        if hasattr(best_model, 'feature_importances_'):
            # For Random Forest models
            feature_names = best_model[:-1].get_feature_names_out()
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model[-1].feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(best_model[-1], 'feature_importances_'):
            # For XGBoost models in pipeline
            if hasattr(best_model[:-1], 'get_feature_names_out'):
                feature_names = best_model[:-1].get_feature_names_out()
            else:
                feature_names = [f'feature_{i}' for i in range(len(best_model[-1].feature_importances_))]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model[-1].feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            # For models without feature importance
            importance_df = pd.DataFrame({
                'feature': ['Not available'],
                'importance': [0]
            })
        
        importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        print("Saved feature importance")
    except Exception as e:
        print(f"Could not save feature importance: {e}")

# Train, evaluate, and save models & plots
def train_models(input_path, output_dir, run_all = False):

    df = load_data(input_path)
    X = df.drop(columns=[DEFAULT_TARGET])
    y = df[DEFAULT_TARGET]
    
    # Create label encoder for saving
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, stratify=y_encoded, random_state=SEED
    )

    preproc_base, preproc_binned = build_preprocessors(X_train)
    # Model setups with comprehensive configurations
    # BASE: Standard preprocessing (scaling + one-hot encoding)
    # BINN: Binned preprocessing (threshold binarization + standard preprocessing)
    # BALANCE: Uses class_weight='balanced' for handling imbalanced classes
    # SMOTE: Uses SMOTE oversampling technique for handling imbalanced classes
    # GRID: Uses GridSearchCV for hyperparameter tuning
    models = {
        # === LOGISTIC REGRESSION MODELS ===
        'BASE_LR'              : {'pre': preproc_base, 'clf': LogisticRegression(max_iter=LR_MAXITER, random_state=SEED), 'grid': None, 'smote': False},  # Basic Logistic Regression with standard preprocessing
        'BASE_LR_BALANCE'      : {'pre': preproc_base, 'clf': LogisticRegression(max_iter=LR_MAXITER, class_weight='balanced', random_state=SEED), 'grid': None, 'smote': False},  # Logistic Regression with balanced class weights
        'BASE_LR_SMOTE'        : {'pre': preproc_base, 'clf': LogisticRegression(max_iter=LR_MAXITER, random_state=SEED), 'grid': None, 'smote': True},  # Logistic Regression with SMOTE oversampling

        # === RANDOM FOREST MODELS ===
        'BASE_RF'              : {'pre': preproc_base, 'clf': RandomForestClassifier(n_estimators=RF_N_EST, random_state=SEED), 'grid': None, 'smote': False},  # Basic Random Forest with standard preprocessing
        'BASE_RF_BALANCE'      : {'pre': preproc_base, 'clf': RandomForestClassifier(n_estimators=RF_N_EST, class_weight='balanced', random_state=SEED), 'grid': None, 'smote': False},  # Random Forest with balanced class weights
        'BASE_RF_SMOTE'        : {'pre': preproc_base, 'clf': RandomForestClassifier(n_estimators=RF_N_EST, random_state=SEED), 'grid': None, 'smote': True},  # Random Forest with SMOTE oversampling
        'BASE_RF_GRID'         : {'pre': preproc_base, 'clf': RandomForestClassifier(random_state=SEED), 'grid': RF_GRID, 'smote': False},  # Random Forest with hyperparameter tuning
        'BASE_RF_GRID_SMOTE'   : {'pre': preproc_base, 'clf': RandomForestClassifier(random_state=SEED), 'grid': RF_GRID, 'smote': True},  # Random Forest with both grid search and SMOTE

        # === XGBOOST MODELS ===
        'BASE_XGB'             : {'pre': preproc_base, 'clf': XGBClassifier(eval_metric=XGB_EVAL_MT, random_state=SEED), 'grid': None, 'smote': False},  # Basic XGBoost with standard preprocessing
        'BASE_XGB_SMOTE'       : {'pre': preproc_base, 'clf': XGBClassifier(eval_metric=XGB_EVAL_MT, random_state=SEED), 'grid': None, 'smote': True },  # XGBoost with SMOTE oversampling
        'BASE_XGB_GRID'        : {'pre': preproc_base, 'clf': XGBClassifier(eval_metric=XGB_EVAL_MT, random_state=SEED), 'grid': XGB_GRID, 'smote': False},  # XGBoost with hyperparameter tuning
        'BASE_XGB_GRID_SMOTE'  : {'pre': preproc_base, 'clf': XGBClassifier(eval_metric=XGB_EVAL_MT, random_state=SEED), 'grid': XGB_GRID, 'smote': True },  # XGBoost with both grid search and SMOTE

        # === BINNED FEATURE MODELS ===
        # Using threshold binarization for specific occupation and evaluation features
        
        # --- Binned Logistic Regression ---
        'BINN_LR'              : {'pre': preproc_binned, 'clf': LogisticRegression(max_iter=LR_MAXITER, random_state=SEED), 'grid': None, 'smote': False},  # Logistic Regression with feature binarization
        'BINN_LR_BALANCE'      : {'pre': preproc_binned, 'clf': LogisticRegression(max_iter=LR_MAXITER, class_weight='balanced', random_state=SEED), 'grid': None, 'smote': False},  # Binned LR with balanced class weights
        'BINN_LR_SMOTE'        : {'pre': preproc_binned, 'clf': LogisticRegression(max_iter=LR_MAXITER, random_state=SEED), 'grid': None, 'smote': True},  # Binned LR with SMOTE oversampling

        # --- Binned Random Forest ---
        'BINN_RF'              : {'pre': preproc_binned, 'clf': RandomForestClassifier(n_estimators=RF_N_EST, random_state=SEED), 'grid': None, 'smote': False},  # Random Forest with feature binarization
        'BINN_RF_BALANCE'      : {'pre': preproc_binned, 'clf': RandomForestClassifier(n_estimators=RF_N_EST, class_weight='balanced', random_state=SEED), 'grid': None, 'smote': False},  # Binned RF with balanced class weights
        'BINN_RF_SMOTE'        : {'pre': preproc_binned, 'clf': RandomForestClassifier(n_estimators=RF_N_EST, random_state=SEED), 'grid': None, 'smote': True},  # Binned RF with SMOTE oversampling
        'BINN_RF_GRID'         : {'pre': preproc_binned, 'clf': RandomForestClassifier(random_state=SEED), 'grid': RF_GRID, 'smote': False},  # Binned RF with hyperparameter tuning
        'BINN_RF_GRID_SMOTE'   : {'pre': preproc_binned, 'clf': RandomForestClassifier(random_state=SEED), 'grid': RF_GRID, 'smote': True},  # Binned RF with both grid search and SMOTE

        # --- Binned XGBoost ---
        'BINN_XGB'             : {'pre': preproc_binned, 'clf': XGBClassifier(eval_metric=XGB_EVAL_MT, random_state=SEED), 'grid': None, 'smote': False},  # XGBoost with feature binarization
        'BINN_XGB_SMOTE'       : {'pre': preproc_binned, 'clf': XGBClassifier(eval_metric=XGB_EVAL_MT, random_state=SEED), 'grid': None, 'smote': True },  # Binned XGB with SMOTE oversampling
        'BINN_XGB_GRID'        : {'pre': preproc_binned, 'clf': XGBClassifier(eval_metric=XGB_EVAL_MT, random_state=SEED), 'grid': XGB_GRID, 'smote': False},  # Binned XGB with hyperparameter tuning
        'BINN_XGB_GRID_SMOTE'  : {'pre': preproc_binned, 'clf': XGBClassifier(eval_metric=XGB_EVAL_MT, random_state=SEED), 'grid': XGB_GRID, 'smote': True }  # Binned XGB with both grid search and SMOTE - often the best performing model

    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Store trained models for finding best one
    trained_models = {}
    
    for name, cfg in models.items():
        model_path = os.path.join(output_dir, f"{name}_model.pkl")
        if not run_all and os.path.exists(model_path):
            print(f"Skipping {name}: model file exists.")
            continue
        
        steps = []
        steps.append(('pre', cfg['pre']))
        if cfg['smote']:
            steps.append(('smote', SMOTE(random_state=SEED)))
        steps.append(('clf', cfg['clf']))

        if cfg['smote']:
            pipeline = ImbPipeline(steps)
        else:
            pipeline = Pipeline(steps)
        # Grid search if specified
        if cfg['grid']:
            gs = GridSearchCV(pipeline, cfg['grid'], cv=GRID_K_FOLD, scoring='roc_auc_ovr_weighted', n_jobs=-1)
            model = gs.fit(X_train, y_train).best_estimator_
        else:
            model = pipeline.fit(X_train, y_train)

        # Store trained model
        trained_models[name] = model

        # Prepare output subfolders
        reports_dir = os.path.join(output_dir, 'reports')
        figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        # Save model
        path = os.path.join(output_dir, f"{name}_model.pkl")
        joblib.dump(model, path)
        print(f"Saved {name} -> {path}")
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Report
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        df_report['roc_auc'] = ''
        df_report.loc['roc_auc', 'roc_auc'] = roc_auc
        df_report.to_csv(os.path.join(reports_dir, f"{name}_report.csv"))
        print(f"{name} ROC AUC: {roc_auc:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
        fig, ax = plt.subplots(figsize=(6,6))
        disp.plot(ax=ax, cmap='Blues')
        plt.title(f"Confusion Matrix: {name}")
        fig.savefig(os.path.join(figures_dir, f"{name}_confusion.png"))
        plt.close(fig)
    
    print('All models trained and plots saved.')
    
    # Find and save best model components
    reports_dir = os.path.join(output_dir, 'reports')
    best_model_name, best_roc_auc = find_best_model(reports_dir)
    
    if best_model_name and best_model_name in trained_models:
        best_model = trained_models[best_model_name]
        save_streamlit_components(output_dir, X_train, label_encoder, best_model_name, best_model)
        
        # Create summary results CSV
        results_data = []
        for name in trained_models.keys():
            report_path = os.path.join(reports_dir, f"{name}_report.csv")
            try:
                report_df = pd.read_csv(report_path, index_col=0)
                roc_auc = float(report_df.loc['roc_auc', 'roc_auc'])
                accuracy = float(report_df.loc['accuracy', 'f1-score'])
                precision = float(report_df.loc['weighted avg', 'precision'])
                recall = float(report_df.loc['weighted avg', 'recall'])
                f1 = float(report_df.loc['weighted avg', 'f1-score'])
                
                results_data.append({
                    'model': name,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc
                })
            except Exception as e:
                print(f"Error processing {name}: {e}")
        
        results_df = pd.DataFrame(results_data).set_index('model')
        results_df.to_csv(os.path.join(output_dir, 'model_performance_results.csv'))
        print("Saved model performance summary")
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS - BEST MODEL SELECTION")
        print("=" * 60)
        print(f"WINNING MODEL: {best_model_name}")
        print(f"BEST ROC AUC SCORE: {best_roc_auc:.4f}")
        print("=" * 60)
        print(f"Model saved as: best_model.pkl")
        print(f"Model name saved to: best_model_name.txt")
        print("Training completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run-all', action='store_true', help='Retrain all models, even if output exists')
    args = parser.parse_args()
    train_models(args.input, args.output, run_all=args.run_all)