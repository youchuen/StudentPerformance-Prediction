import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.evaluation import evaluate_model

def test_evaluate_model():
    X = pd.DataFrame(np.random.rand(10,3), columns=list('ABC'))
    y = np.random.randint(0,2,size=10)
    model = RandomForestClassifier().fit(X, y)
    report, fpr, tpr, roc_auc = evaluate_model(model, X, y)
    assert 'accuracy' in report
    assert len(fpr) == len(tpr)
    assert isinstance(roc_auc, float)