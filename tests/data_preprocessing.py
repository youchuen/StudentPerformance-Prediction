import pandas as pd
import os
import pytest
from src.data_processing import clean_data, save_processed

def test_clean_data():
    df = pd.DataFrame({'A':[1,None,3], 'Target':[1,0,1]})
    cleaned = clean_data(df)
    assert cleaned['A'].isnull().sum() == 0

def test_save_processed(tmp_path):
    df = pd.DataFrame({'A':[1,2]})
    out = tmp_path / "out.csv"
    save_processed(df, str(out))
    assert os.path.exists(str(out))