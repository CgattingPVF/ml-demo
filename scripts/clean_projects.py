#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "projects_export_gdpr_safe.xlsx"
OUT_CSV = ROOT / "data" / "projects_export_gdpr_safe_cleaned.csv"
OUT_XLSX = ROOT / "data" / "projects_export_gdpr_safe_cleaned.xlsx"
LOG = ROOT / "data" / "projects_cleaning_log.txt"

df = pd.read_excel(IN, sheet_name='GDPR Safe Data', dtype=object)

def snake(s):
    s = str(s).strip()
    s = s.lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"__+", "_", s)
    return s.strip("_")

df.columns = [snake(c) for c in df.columns]

empty_cols = [c for c in df.columns if df[c].isna().all()]
df = df.drop(columns=empty_cols)

for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({'': np.nan, 'nan': np.nan})

date_cols = []
for c in df.columns:
    parsed = pd.to_datetime(df[c], errors='coerce')
    if parsed.notna().mean() > 0.5:
        df[c] = parsed
        date_cols.append(c)

num_cols = []
for c in df.columns:
    coerced = pd.to_numeric(df[c].astype(str).str.replace(r"[^0-9.+\-eE]", "", regex=True), errors='coerce')
    if coerced.notna().sum() > 0 and (coerced.notna().mean() > 0.9 or df[c].dropna().apply(lambda x: isinstance(x,(int,float))).all()):
        df[c] = coerced
        num_cols.append(c)

if 'c_homeownerpostcode' in df.columns:
    df['c_homeownerpostcode'] = df['c_homeownerpostcode'].astype(str).str.replace(r"\s+", "", regex=True).str.upper()

if 'c_edgeprotection' in df.columns:
    def normalize_edge(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s in ('yes','y','true','t','1'):
            return 'Yes'
        if s in ('no','n','false','f','0'):
            return 'No'
        return x.strip()
    df['c_edgeprotection_normalized'] = df['c_edgeprotection'].apply(normalize_edge)

for c in df.columns:
    vals = df[c].dropna().astype(str).str.strip()
    if not vals.empty and vals.map(lambda x: x in ("0","1","0.0","1.0")).all():
        df[c] = df[c].astype(float).map({0.0: False, 1.0: True})

empty_cols_after = [c for c in df.columns if df[c].isna().all()]
df = df.drop(columns=empty_cols_after)

before = len(df)
df = df.drop_duplicates()
after = len(df)

df.to_csv(OUT_CSV, index=False)
df.to_excel(OUT_XLSX, index=False)

with open(LOG, "w") as f:
    f.write(f"Input: {IN}\n")
    f.write(f"Dropped empty columns: {empty_cols + empty_cols_after}\n")
    f.write(f"Converted date columns: {date_cols}\n")
    f.write(f"Converted numeric columns: {num_cols}\n")
    f.write(f"Rows before dedupe: {before}, after: {after}\n")
    f.write("Edge protection normalization: created column c_edgeprotection_normalized\n")

print("Cleaned files written:", OUT_CSV, OUT_XLSX)
print("Log written:", LOG)
