import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import shutil

def load_reports(models_dir):
    # Load each *_report.csv into a DataFrame
    records = []
    for fname in os.listdir(models_dir):
        if fname.endswith('_report.csv'):
            model_name = fname.replace('_report.csv','')
            df = pd.read_csv(os.path.join(models_dir, fname), index_col=0)

            accuracy = df.loc['accuracy','precision'] if 'precision' in df.columns and 'accuracy' in df.index else None

            if 'weighted avg' in df.index:
                row = df.loc['weighted avg']
            else:
                row = df.loc['macro avg']
            precision = row['precision']
            recall    = row['recall']
            f1        = row['f1-score'] if 'f1-score' in row else row['f1']

            roc_auc = df.at['roc_auc','roc_auc'] if 'roc_auc' in df.index and 'roc_auc' in df.columns else None
            records.append({
                'model': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            })
    return pd.DataFrame(records)


def plot_roc_aucs(df_metrics, output_dir):
    plt.figure(figsize=(10,6))
    df_metrics = df_metrics.set_index('model')
    df_metrics['roc_auc'].plot(kind='bar')
    plt.title('ROC AUC by Model')
    plt.ylabel('ROC AUC')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_auc_comparison.png'))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', required=True, help='Directory with model .pkl and reports')
    parser.add_argument('--output', required=True, help='Directory to save summary and plots')
    args = parser.parse_args()

    summary_dir = os.path.join(args.output, 'summary')
    reports_dir = os.path.join(args.models_dir, 'reports')
    os.makedirs(summary_dir, exist_ok=True)

    metrics_df = load_reports(reports_dir)

    metrics_df.to_csv(os.path.join(summary_dir, 'model_summary.csv'), index=False)
    print('Summary saved to model_summary.csv')

    # Plot ROC AUCs
    if 'roc_auc' in metrics_df.columns:
        plot_roc_aucs(metrics_df, summary_dir)
        print('ROC AUC comparison plot saved.')

        best_idx = metrics_df['roc_auc'].idxmax()
        best_row = metrics_df.loc[best_idx]
        print(f"=== 🏆 Best model: {best_row['model']}  (ROC AUC = {best_row['roc_auc']:.3f}) ===")

        src = os.path.join(args.models_dir, f"{best_row['model']}_model.pkl")
        dst = os.path.join(args.models_dir, 'best_model.pkl')
        shutil.copy(src, dst)
        print(f"Copied best model to {dst}")