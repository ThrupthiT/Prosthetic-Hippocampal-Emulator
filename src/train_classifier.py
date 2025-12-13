import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

FEATURE_CSV = r"C:\projects\prosthetic_hippocampal\features\eeg_band_features.csv"
MODEL_PATH = r"C:\projects\prosthetic_hippocampal\outputs\memory_classifier.pkl"

def add_dominant_label(df):
    """Assign dominant wave label based on highest value among alpha/beta/gamma."""
    df['dominant_wave'] = df[['alpha', 'beta', 'gamma']].idxmax(axis=1)
    return df

def main():
    df = pd.read_csv(FEATURE_CSV)

    # If dominant wave not present, compute it
    if "dominant_wave" not in df.columns:
        df = add_dominant_label(df)

    # Features you actually have
    X = df[['alpha', 'beta', 'gamma']]  
    y = df['dominant_wave']

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standard scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # predictions
    preds = clf.predict(X_test)

    print("✔ Accuracy:", accuracy_score(y_test, preds))
    print("\n✔ Classification Report:\n", classification_report(y_test, preds))

    # Save model + scaler
    joblib.dump({"model": clf, "scaler": scaler}, MODEL_PATH)
    print(f"✔ Model saved → {MODEL_PATH}")

if __name__ == "__main__":
    main()
