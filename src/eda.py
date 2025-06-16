import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from src.config import DEFAULT_TARGET

def sanitize_filename(name):
    return name.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('	', '_')

def summary_stats(df, output_dir):
    out = os.path.join(output_dir, 'univariate_plots')
    os.makedirs(out, exist_ok=True)
    for col in df.select_dtypes(include=['int64','float64']).columns:
        safe_col = sanitize_filename(col)
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(out, f'{safe_col}_hist.png'))
        plt.close()

def bivariate_analysis(df, target_col, output_dir):
    out = os.path.join(output_dir, 'bivariate_plots')
    os.makedirs(out, exist_ok=True)
    safe_target = sanitize_filename(target_col)
    for col in df.select_dtypes(include=['int64','float64']).columns:
        if col == target_col:
            continue
        safe_col = sanitize_filename(col)
        plt.figure()
        sns.boxplot(x=df[target_col], y=df[col])
        plt.title(f'{col} vs {target_col}')
        plt.savefig(os.path.join(out, f'{safe_col}_vs_{safe_target}.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target', default=DEFAULT_TARGET)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    summary_stats(df, args.output)
    bivariate_analysis(df, args.target, args.output)
    print('EDA plots saved.')

if __name__ == '__main__':
    main()
