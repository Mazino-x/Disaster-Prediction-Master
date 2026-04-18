#!/usr/bin/env python3
"""
Wrapper to test optimal decision threshold for tsunami model.
Uses ROC/precision-recall analysis to find best threshold.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split

def find_optimal_threshold():
    print(f"\n{'='*70}")
    print("FINDING OPTIMAL DECISION THRESHOLD")
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
    
    # Calculate metrics for different thresholds
    print("\nThreshold Analysis (on test set):")
    print("Threshold | Accuracy | Precision | Recall | F1-Score | Support")
    print("-" * 70)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.3, 0.8, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        acc = (tp + tn) / len(y_test)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        print(f"  {threshold:.2f}    |  {acc:.4f}   |  {prec:.4f}    | {rec:.4f} | {f1:.4f}   | Balanced")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print("\n" + "="*70)
    print(f"Recommended threshold: {best_threshold:.2f} (F1-score: {best_f1:.4f})")
    print("="*70)
    
    # Show results with optimal threshold
    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    print(f"\nClassification Report (threshold = {best_threshold:.2f}):")
    print(classification_report(y_test, y_pred_optimal, target_names=['No Tsunami', 'Tsunami']))
    
    return best_threshold

if __name__ == '__main__':
    optimal = find_optimal_threshold()
    print(f"\nOptimal threshold: {optimal:.2f}")
