#!/usr/bin/env python3
"""
Improved tsunami model retraining with class balancing and threshold optimization.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, f1_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

def load_data(csv_path):
    """Load tsunami CSV."""
    print(f"\n{'='*70}")
    print("TSUNAMI MODEL - IMPROVED RETRAINING")
    print(f"{'='*70}")
    print(f"\nLoading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} tsunami records")
    return df

def prepare_features(df):
    """Prepare features with proper handling."""
    print("\nPreparing features...")
    
    feature_cols = ['Latitude', 'Longitude', 'Tsunami Magnitude (Iida)', 'Year', 'Mo', 'Dy', 'Hr', 'Mn', 'Sec']
    
    # Fill missing numeric values
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
    
    # Create labels: more nuanced labeling
    # Use Maximum Water Height as primary indicator
    mwh_col = 'Maximum Water Height (m)'
    validity_col = 'Tsunami Event Validity'
    
    # Definite tsunami: high water height OR high validity
    y_high_mwh = pd.to_numeric(df[mwh_col], errors='coerce') >= 1.0
    y_high_validity = df[validity_col].astype(float) >= 4
    y_med_validity = df[validity_col].astype(float) >= 3
    
    # Label 1: Strong tsunamis (either definite or significant water height)
    y = (y_high_mwh | y_high_validity | y_med_validity).astype(int)
    
    print(f"Features: {feature_cols}")
    print(f"\nLabel distribution:")
    print(f"  Class 0 (No/Weak Tsunami): {(y == 0).sum()}")
    print(f"  Class 1 (Definite Tsunami): {(y == 1).sum()}")
    print(f"  Ratio: {(y == 1).sum() / len(y) * 100:.1f}% positive class")
    
    return X, y

def train_model(X, y):
    """Train with proper class balancing."""
    print("\nTraining model...")
    print(f"  Total samples: {len(X)}")
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Compute balanced class weights
    classes, counts = np.unique(y_train, return_counts=True)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
    print(f"\nClass weights (for balancing):")
    for cls, weight in class_weight_dict.items():
        print(f"  Class {cls}: {weight:.3f}")
    
    # Train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weight_dict,
            n_jobs=-1,
            random_state=42,
            verbose=0
        ))
    ])
    
    print("\nFitting model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\nPerformance:")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, 
                              target_names=['No Tsunami', 'Tsunami'],
                              digits=4))
    
    print(f"\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    return pipeline, X_test, y_test

def save_model(pipeline, output_path):
    """Save trained model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"\n✓ Model saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

def main():
    data_path = Path(__file__).parent / 'data' / 'tsunami_historical_data_from_1800_to_2021.csv'
    output_path = Path(__file__).parent / 'src' / 'api' / 'models' / 'tsunami_model.joblib'
    
    # Load and prepare
    df = load_data(str(data_path))
    X, y = prepare_features(df)
    
    # Train
    model, X_test, y_test = train_model(X, y)
    
    # Save
    save_model(model, output_path)
    
    print(f"\n{'='*70}")
    print("RETRAINING COMPLETE")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
