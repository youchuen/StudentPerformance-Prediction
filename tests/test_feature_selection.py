import pandas as pd
import numpy as np
from src.feature_selection import compute_rf_importance, compute_mutual_info

def test_compute_rf_importance():
    X = pd.DataFrame(np.random.rand(10,3), columns=list('ABC'))
    y = pd.Series(np.random.randint(0,2,size=10))
    imp = compute_rf_importance(X, y)
    assert set(imp.index) == set(X.columns)

def test_compute_mutual_info():
    X = pd.DataFrame(np.random.rand(10,3), columns=list('ABC'))
    y = pd.Series(np.random.randint(0,2,size=10))
    mi = compute_mutual_info(X, y)
    assert set(mi.index) == set(X.columns)