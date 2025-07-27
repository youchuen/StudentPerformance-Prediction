set -e
# ─── Find a Python interpreter ───────────────────────────────────────────────
if   command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python  >/dev/null 2>&1; then
  PY=python
elif command -v py      >/dev/null 2>&1; then
  PY="py -3"
else
  echo "❌ No Python interpreter found; please install Python."
  exit 1
fi

echo "Using interpreter: $PY"

# ─── Create & activate virtual environment ──────────────────────────────────
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR"
  $PY -m venv "$VENV_DIR"
fi

# detect activation script
if [ -f "$VENV_DIR/bin/activate" ]; then
  # Unix / WSL / macOS
  source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
  # Windows (Git Bash)
  source "$VENV_DIR/Scripts/activate"
else
  echo "❌ Could not find venv activation script."
  exit 1
fi

# ─── Install dependencies inside the venv ──────────────────────────────────
echo "[Step 1/8] Installing requirements…"
# Use the venv’s pip via python -m pip to be 100% sure
$PY -m pip install --upgrade pip
$PY -m pip install -r environment.txt

echo "[Step 2/8] Running data processing..."
$PY -m src.data_preprocessing --input data/raw/data.csv --output data/processed/cleaned.csv

echo "[Step 3/8] Running EDA..."
$PY -m src.eda --input data/processed/cleaned.csv --output outputs/figures/

echo "[Step 4/8] Running feature selection..."
$PY -m src.feature_selection --input data/processed/cleaned.csv --output-features data/processed/features.csv --figures outputs/figures/

echo "[Step 5/8] Training models..."
$PY -m src.modeling --input data/processed/cleaned.csv --output outputs/models/ --run-all

echo "[Step 6/8] Evaluating models..."
$PY -m src.evaluation --models outputs/models/ --output outputs/figures/

echo "[Step 7/8] Generating Streamlit Componenets..."
$PY -m streamlit_app.generate_streamlit_components 

echo "[Step 8/8] Starting Streamlit App..."
$PY -m streamlit run streamlit_app/app.py

echo "All done."
