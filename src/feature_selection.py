import os
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import (
    SEED,
    RF_N_EST,
    SELECTED_FEATURES,
    DEFAULT_TARGET
)

# After Feature Selection
selected_features = SELECTED_FEATURES

def compute_rf_importance(X, y):
    rf = RandomForestClassifier(n_estimators=RF_N_EST, random_state=SEED)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)


def compute_mutual_info(X, y):
    mi = mutual_info_classif(X, y, random_state=SEED)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def plot_importance(series, title, path):
    plt.figure()
    series.plot(kind='bar')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_selected_features(df, output_dir, target_col=DEFAULT_TARGET):

    if df[target_col].dtype != 'category':
        df[target_col] = df[target_col].astype('category')
    
    # Create 3x3 grid for first 9 selected features
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    os.makedirs(output_dir, exist_ok=True)
    for idx, col in enumerate(selected_features[:9]):
        ax = axes[idx]
        if df[col].nunique() <= 10:
            sns.countplot(data=df, x=col, hue=target_col, ax=ax)
        else:
            sns.boxplot(data=df, x=target_col, y=col, ax=ax)
        ax.set_title(f"{col} vs {target_col}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'selected_features_grid.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-features", required=True)
    parser.add_argument("--figures", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    y = df[DEFAULT_TARGET]
    X = df.drop(columns=[DEFAULT_TARGET])

    # Compute importances for reference
    rf_imp = compute_rf_importance(X, y)
    mi_imp = compute_mutual_info(X, y)

    os.makedirs(args.figures, exist_ok=True)
    plot_importance(rf_imp, "RF Feature Importance", os.path.join(args.figures, 'rf_importance.png'))
    plot_importance(mi_imp, "Mutual Information", os.path.join(args.figures, 'mi_importance.png'))

    feats = pd.DataFrame({'rf': rf_imp, 'mi': mi_imp})
    os.makedirs(os.path.dirname(args.output_features), exist_ok=True)
    feats.to_csv(args.output_features, index=True)
    print(f"Features saved to {args.output_features}")

    # Only plot selected features if they actually exist in df
    available = [f for f in selected_features if f in df.columns]
    if available:
        plot_selected_features(df[available + [DEFAULT_TARGET]], 
                               os.path.join(args.figures, 'selected_features'))
        print("Selected feature distributions plotted.")
    else:
        print("No selected features found in the data; skipping distribution plots.")

if __name__ == "__main__":
    main()