#!/usr/bin/env python3
"""
Generate all required components for Streamlit app from existing models.
This script creates the missing components needed by the Streamlit app.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  
CLEANED_PATH = ROOT / "data" / "processed" / "cleaned.csv"
MODEL_PATH   = ROOT / "outputs" / "models"
REPORTS_PATH = MODEL_PATH / "reports"

def find_best_model_from_reports():
    """Find the best model name from existing reports."""
    reports_dir = REPORTS_PATH
    best_model_name = None
    best_roc_auc = 0
    
    print("Analyzing model performance reports...")
    
    for file_name in os.listdir(reports_dir):
        if file_name.endswith('_report.csv'):
            model_name = file_name.replace('_report.csv', '')
            report_path = os.path.join(reports_dir, file_name)
            
            try:
                report_df = pd.read_csv(report_path, index_col=0)
                roc_auc = float(report_df.loc['roc_auc', 'roc_auc'])
                print(f"  {model_name}: ROC AUC = {roc_auc:.4f}")
                
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    best_model_name = model_name
            except Exception as e:
                print(f"  Error reading {file_name}: {e}")
    
    print(f"\nBest model: {best_model_name} with ROC AUC = {best_roc_auc:.4f}")
    return best_model_name, best_roc_auc

def generate_components():
    """Generate all required Streamlit components."""
    output_dir = MODEL_PATH
    
    # Find best model
    best_model_name, best_roc_auc = find_best_model_from_reports()
    
    # Load the data to get feature information
    data_path = CLEANED_PATH
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return False
    
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Create label encoder
    print("Creating label encoder...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(data['Target'])
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
    print("Saved label_encoder.pkl")
    
    # Get feature columns (excluding Target)
    feature_columns = [col for col in data.columns if col != 'Target']
    joblib.dump(feature_columns, os.path.join(output_dir, 'feature_columns.pkl'))
    print("Saved feature_columns.pkl")
    
    # Get numerical and categorical columns
    X = data[feature_columns]
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    
    joblib.dump(numeric_cols, os.path.join(output_dir, 'numerical_columns.pkl'))
    joblib.dump(categorical_cols, os.path.join(output_dir, 'categorical_columns.pkl'))
    print("Saved numerical_columns.pkl and categorical_columns.pkl")
    
    # Create class names
    class_names = ['Dropout', 'Enrolled', 'Graduate']
    joblib.dump(class_names, os.path.join(output_dir, 'class_names.pkl'))
    print("Saved class_names.pkl")
    
    # Copy best model if not already done
    best_model_path = os.path.join(output_dir, f"{best_model_name}_model.pkl")
    best_model_copy_path = os.path.join(output_dir, 'best_model.pkl')
    
    if os.path.exists(best_model_path):
        if not os.path.exists(best_model_copy_path):
            import shutil
            shutil.copy2(best_model_path, best_model_copy_path)
            print(f"Copied {best_model_name} to best_model.pkl")
        else:
            print("best_model.pkl already exists")
    
    # Generate feature importance
    print("Generating feature importance...")
    try:
        model = joblib.load(best_model_copy_path)
        
        if hasattr(model, 'feature_importances_'):
            feature_names = feature_columns
            importance_values = model.feature_importances_
        elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_importances_'):
            if hasattr(model[:-1], 'get_feature_names_out'):
                try:
                    feature_names = model[:-1].get_feature_names_out()
                except:
                    feature_names = feature_columns
            else:
                feature_names = feature_columns
            importance_values = model.steps[-1][1].feature_importances_
        else:
            feature_names = feature_columns
            importance_values = np.ones(len(feature_columns)) / len(feature_columns)
        
        if len(importance_values) != len(feature_names):
            print(f"Feature count mismatch. Using top {len(importance_values)} features")
            feature_names = feature_names[:len(importance_values)]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        print("Saved feature_importance.csv")
        
    except Exception as e:
        print(f"Could not generate feature importance: {e}")
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': np.ones(len(feature_columns)) / len(feature_columns)
        })
        importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        print("Saved dummy feature_importance.csv")
    
    # Create performance summary
    print("Creating performance summary...")
    results_data = []
    reports_dir = os.path.join(output_dir, 'reports')
    
    for file_name in os.listdir(reports_dir):
        if file_name.endswith('_report.csv'):
            model_name = file_name.replace('_report.csv', '')
            report_path = os.path.join(reports_dir, file_name)
            
            try:
                report_df = pd.read_csv(report_path, index_col=0)
                
                results_data.append({
                    'model': model_name,
                    'accuracy': float(report_df.loc['accuracy', 'f1-score']),
                    'precision': float(report_df.loc['weighted avg', 'precision']),
                    'recall': float(report_df.loc['weighted avg', 'recall']),
                    'f1': float(report_df.loc['weighted avg', 'f1-score']),
                    'roc_auc': float(report_df.loc['roc_auc', 'roc_auc'])
                })
            except Exception as e:
                print(f"Error processing {model_name}: {e}")
    
    if results_data:
        results_df = pd.DataFrame(results_data).set_index('model')
        results_df.to_csv(os.path.join(output_dir, 'model_performance_results.csv'))
        print("Saved model_performance_results.csv")
    
    return True

def main():
    print("Student Performance Prediction - Component Generator")
    print("=" * 60)
    print("Generating Streamlit components from existing models...")
    
    if generate_components():
        print("\n" + "=" * 60)
        print("All components generated successfully!")
        print("\nGenerated files in outputs/models/:")
        print("  • best_model.pkl")
        print("  • label_encoder.pkl")
        print("  • feature_columns.pkl")
        print("  • numerical_columns.pkl")
        print("  • categorical_columns.pkl") 
        print("  • class_names.pkl")
        print("  • feature_importance.csv")
        print("  • model_performance_results.csv")
        
        print("\nYou can now run the Streamlit app:")
        print("  streamlit run streamlit_app/app_2.py")
    else:
        print("\nComponent generation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
