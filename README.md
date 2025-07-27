# Student Performance Prediction

A fully automated pipeline for predicting student performance using machine learning. This project covers data cleaning, exploratory analysis, feature selection, model training, evaluation, and a Streamlit web app for interactive results.

---

## Features
- **Automated Data Pipeline:** Clean, preprocess, and engineer features from raw student data.
- **Exploratory Data Analysis:** Visualize distributions, relationships, and key insights.
- **Feature Selection:** Identify and select the most important features for modeling.
- **Model Training:** Train and compare multiple classifiers (Logistic Regression, Random Forest, XGBoost) with various strategies (balancing, SMOTE, grid search).
- **Evaluation:** Aggregate model results, compare ROC AUC, and select the best model.
- **Streamlit App:** Launch an interactive dashboard to explore predictions and model insights.

---

## Project Structure
```
StudentPerformance-Prediction/
├── config.yaml              # Pipeline configuration
├── environment.txt          # Python dependencies
├── scripts/
│   └── run_all.sh           # Run the full pipeline
├── data/
│   ├── raw/                 # Raw input data
│   └── processed/           # Cleaned and feature-engineered data
├── outputs/
│   ├── figures/             # All generated plots
│   └── models/              # Trained models and reports
├── src/
│   ├── config.py            # Loads config.yaml
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_selection.py
│   ├── modeling.py
│   └── evaluation.py
├── streamlit_app/
│   ├── app.py               # Streamlit dashboard
│   └── generate_streamlit_components.py
└── README.md
```

---

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/youchuen/StudentPerformance-Prediction.git
   cd StudentPerformance-Prediction
   ```

2. **Run the pipeline**
   ```bash
   bash scripts/run_all.sh
   ```
   - Sets up a Python virtual environment
   - Installs dependencies from `environment.txt`
   - Runs all pipeline steps: preprocessing, EDA, feature selection, modeling, evaluation
   - Prepares results for the Streamlit app

3. **Launch the Streamlit app**
   ```bash
   cd streamlit_app
   streamlit run app.py
   ```
   - Explore model predictions and visualizations interactively

---

## Outputs
- **Processed Data:** `data/processed/cleaned.csv`, `data/processed/features.csv`
- **Figures:** `outputs/figures/`
- **Models & Reports:** `outputs/models/`
- **Summary & Best Model:** `outputs/models/model_performance_results.csv`, `outputs/models/best_model.pkl`

---

## Requirements
- Python 3.8+
- See `environment.txt` for all required packages

---

