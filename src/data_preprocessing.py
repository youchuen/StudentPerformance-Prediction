import os
import pandas as pd
import argparse
import csv

def ensure_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

def load_data(path):
    ensure_file_exists(path)
    df = pd.read_csv(
        path , 
        delimiter=";",
        encoding='utf-8-sig',
        engine='python',              
        quoting=csv.QUOTE_NONE
        ) 

    df.columns = df.columns.str.strip().str.strip('"')

    str_cols = df.select_dtypes(include="object").columns
    for c in str_cols:
        df[c] = df[c].str.strip().str.strip('"')
    return df

def clean_data(df):
    df = df.dropna()
    df = df.fillna(df.median(numeric_only=True))
    return df

def save_processed(df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = load_data(args.input)
    cleaned = clean_data(df)
    save_processed(cleaned, args.output)
    print(f"Cleaned data saved to {args.output}")

if __name__ == "__main__":
    main()