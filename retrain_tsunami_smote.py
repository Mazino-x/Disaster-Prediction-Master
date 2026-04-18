#!/usr/bin/env python3
"""
Tsunami model retraining with SMOTE resampling for better class balance.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
try:
    from imblearn.over_sampling import SMOTE
    has_smote = True
except ImportError:
    has_smote = False
    print("Warning: imbalanced-learn not installed, skipping SMOTE")
import warnings
warnings.filterwarnings('ignore')

def main():
    print(f"\n{'='*70}")
    print("TSUNAMI MODEL - SMOTE REBALANCING")
    print(f"{'='*70}")
    
    data_path = Path('data/tsunami_historical_data_from_1800_to_2021.csv')
    print(f"\nLoading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} records")
    
    # Prepare features
    print("\nPreparing features...")
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
    
    # More conservative labeling: only high-confidence tsunamis
    mwh_col = 'Maximum Water Height (m)'
    validity_col = 'Tsunami Event Validity'
    
    # Class 1: Only definite/probable tsunamis (validity >= 3 OR high water height)
    y = ((df[validity_col].astype(float) >= 3) | 
         (pd.to_numeric(df[mwh_col], errors='coerce') >= 0.5)).astype(int)
    
    print(f"Original label distribution: {y.value_counts().to_dict()}")
    print(f"  Class 0: {(y == 0).sum()}")
    print(f"  Class 1: {(y == 1).sum()}")
    
    # Split BEFORE resampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain/test split:")
    print(f"  Train: {len(X_train)} samples (Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()})")
    print(f"  Test: {len(X_test)} samples (Class 0: {(y_test==0).sum()}, Class 1: {(y_test==1).sum()})")
    
    # Apply SMOTE only to training data
    if has_smote:
        print("\nApplying SMOTE resampling to training data...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {len(X_train_resampled)} samples (Class 0: {(y_train_resampled==0).sum()}, Class 1: {(y_train_resampled==1).sum()})")
        X_train = X_train_resampled
        y_train = y_train_resampled
    
    # Train without class weights (data is now balanced)
    print("\nTraining RandomForest (300 trees, balanced data)...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
            verbose=0
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate on TEST set (original imbalanced data)
    y_test_pred = pipeline.predict(X_test)
    
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, 
                              target_names=['No Tsunami', 'Tsunami'],
                              digits=4))
    
    print(f"\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Save
    output_path = Path('src/api/models/tsunami_model.joblib')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"\n✓ Model saved to {output_path}")
    
    print(f"\n{'='*70}")
    print("RETRAINING COMPLETE")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
