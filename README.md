# Student Performance Prediction Pipeline

This repository contains a fully automated end-to-end pipeline to preprocess student performance data, perform exploratory data analysis, select features, train a suite of classification models, and evaluate them—all reproducibly in an isolated Python environment.

---

## 📂 Repository Structure
StudentPerformance-Prediction/
├── config.yaml # All your tunable settings: seeds, test split, grids, selected features, target name, bin_cols
├── environment.txt # Pinned package dependencies
├── scripts/
│ └── run_all.sh # Bootstrap venv & run the full pipeline
├── data/
│ ├── raw/
│ │ └── data.csv # Raw input data
│ └── processed/
│ ├── cleaned.csv # Cleaned output from data_preprocessing
│ └── features.csv # Selected features saved by feature_selection
├── outputs/
│ ├── figures/ # All plots (EDA, selected‐feature distributions, confusion matrices, ROC curves, etc.)
│ └── models/ # Fitted model .pkl files & per‐model CSV reports
├── src/
│ ├── config.py # Loads constants from config.yaml
│ ├── data_preprocessing.py # Cleans & normalizes raw data → cleaned.csv
│ ├── eda.py # Generates histograms, boxplots under outputs/figures
│ ├── feature_selection.py # Computes feature importances & saves selected_features → features.csv
│ ├── modeling.py # Trains LR, RF, XGB variants, saves models + reports + figures
│ └── evaluation.py # Aggregates all reports into a summary CSV & plots ROC AUC comparison
├── streamlit_app/
│ ├──app.py # Run streamlit app
│ ├──generate_streamlit_componenets.py # Generate all required components for Streamlit app from existing models
└── README.md # (this file)

## 🚀 Quick Start

1. **Clone the repo**  
   ```bash
   git clone https://github.com/youchuen/StudentPerformance-Prediction.git
   cd StudentPerformance-Prediction
2. **Run the full pipeline**
    ```bash  
    bash scripts/run_all.sh

    This will:
        a. Create (or reuse) a Python 3 virtual environment in ./.venv/

        b. Install everything in requirements.txt

        c. Execute each step in order:

        d. Data preprocessing (src.data_preprocessing)

        e. Exploratory Data Analysis (src.eda)

        f. Feature selection (src.feature_selection)

        g. Model training (src.modeling --run-all)

        h. Evaluation & summary (src.evaluation)

        i. Run pre-requisite for streamlit app and lauch streamlit app on lcoal host
3. **Inspect your results**
    ```bash  
    a. Cleaned & feature‐engineered data:
        data/processed/cleaned.csv, data/processed/features.csv

    b. Plots:
        outputs/figures/

    c. Models & per‐model reports:
        outputs/models/

    d. Combined summary & best‐model selection in outputs (outputs/summary/, outputs/figures/roc_auc_comparison.png)

