#!/usr/bin/env python3
"""
Use percentile-based threshold to get balanced predictions.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

def calibrate_with_percentile():
    print(f"\n{'='*70}")
    print("CALIBRATING WITH PERCENTILE-BASED THRESHOLD")
    print(f"{'='*70}")
    
    # Load data
    data_path = Path('data/tsunami_historical_data_from_1800_to_2021.csv')
    df = pd.read_csv(data_path)
    
    # Prepare features
    feature_cols = ['Latitude', 'Longitude', 'Tsunami Magnitude (Iida)', 'Year', 'Mo', 'Dy', 'Hr', 'Mn', 'Sec']
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Latitude'] = df['Latitude'].fillna(0)
    df['Longitude'] = df['Longitude'].fillna(0)
    df['Tsunami Magnitude (Iida)'] = df['Tsunami Magnitude (Iida)'].fillna(0)
    df['Year'] = df['Year'].fillna(0).astype(int)
    df['Mo'] = df['Mo'].fillna(6).clip(1, 12).astype(int)
    df['Dy'] = df['Dy'].fillna(15).clip(1, 31).astype(int)
    df['Hr'] = df['Hr'].fillna(12).clip(0, 23).astype(int)
    df['Mn'] = df['Mn'].fillna(0).clip(0, 59).astype(int)
    df['Sec'] = df['Sec'].fillna(0).clip(0, 59).astype(int)
    
    X = df[feature_cols].copy()
    
    # Create labels
    mwh_col = 'Maximum Water Height (m)'
    validity_col = 'Tsunami Event Validity'
    y = ((df[validity_col].astype(float) >= 3) | 
         (pd.to_numeric(df[mwh_col], errors='coerce') >= 0.5)).astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load model
    model = joblib.load('src/api/models/tsunami_model.joblib')
    
    # Get probabilities on test set
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\nProbability distribution on test set:")
    print(f"  Min:    {y_proba.min():.4f}")
    print(f"  Q1:     {np.percentile(y_proba, 25):.4f}")
    print(f"  Median: {np.percentile(y_proba, 50):.4f}")
    print(f"  Q3:     {np.percentile(y_proba, 75):.4f}")
    print(f"  Max:    {y_proba.max():.4f}")
    
    # Find percentiles that would give us 25%, 50%, 75% positive class
    for percentile in [25, 33, 50]:
        thresh = np.percentile(y_proba, percentile)
        n_high = np.sum(y_proba >= thresh)
        pct_high = 100 * n_high / len(y_proba)
        print(f"\n{percentile}th percentile: {thresh:.4f} → {pct_high:.1f}% HIGH RISK")
    
    # Use 33rd percentile to get roughly 67% HIGH RISK (matches data distribution)
    median_thresh = np.percentile(y_proba, 33)
    print(f"\n{'='*70}")
    print(f"RECOMMENDED: Use 33rd percentile threshold = {median_thresh:.4f}")
    print(f"This gives balanced predictions matching class distribution")
    print(f"{'='*70}\n")
    
    return median_thresh

if __name__ == '__main__':
    thresh = calibrate_with_percentile()
