import pickle
import csv
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Manage gesture training data collection and storage
class GestureDataManager:

    def __init__(self, csv_path: str = "data/gestures.csv"):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize_dataset(self):
        if self.csv_path.exists():
            return

        headers = ['gesture']
        for hand_num in range(1, 3):
            for landmark_idx in range(21):
                for axis in ['x', 'y', 'z', 'v']:
                    headers.append(f'hand{hand_num}_landmark{landmark_idx}_{axis}')

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def add_gesture_sample(self, gesture_label: str, landmarks: list):
        if not self.csv_path.exists():
            self.initialize_dataset()

        row = [gesture_label] + landmarks
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def get_dataset_info(self) -> Dict[str, Any]:
        if not self.csv_path.exists():
            return None

        df = pd.read_csv(self.csv_path)
        return {
            'total_samples': len(df),
            'num_gestures': df['gesture'].nunique(),
            'gesture_distribution': df['gesture'].value_counts().to_dict()
        }


class GestureModelTrainer:

    # Train machine learning model for gesture classification
    def __init__(self, dataset_path: str = "data/gestures.csv"):
        self.dataset_path = dataset_path
        self.model = None
        self.feature_columns = None
        self.class_labels = None
        self.metrics = {}

    def train(self, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        print("Loading dataset...")
        df = pd.read_csv(self.dataset_path)

        X = df.drop('gesture', axis=1)
        y = df['gesture']

        self.feature_columns = X.columns.tolist()
        self.class_labels = y.unique().tolist()

        print(f"Dataset loaded: {len(X)} samples, {len(self.class_labels)} classes")
        print(f"Classes: {self.class_labels}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        print("Building and training pipeline...")

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            ))
        ])

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'test_size': test_size,
            'num_samples': len(X),
            'num_features': len(self.feature_columns),
            'num_classes': len(self.class_labels),
            'classes': self.class_labels
        }

        print("\nTraining Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        return self.metrics

    def save_model(self, model_path: str = "models/gesture_model.pkl"):
        if self.model is None:
            raise ValueError("Model not trained. Train model first.")

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'class_labels': self.class_labels,
            'metrics': self.metrics
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {model_path}")


class GestureClassifier:

    # Load and use trained model for real-time gesture prediction
    def __init__(self, model_path: str = "models/gesture_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_columns = None
        self.class_labels = None
        self.metrics = None

        if self.model_path.exists():
            self.load_model()

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.class_labels = model_data['class_labels']
        self.metrics = model_data.get('metrics', {})

        print(f"Model loaded from: {self.model_path}")

    def predict(self, landmarks: list) -> Tuple[str, float]:
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")

        if len(landmarks) != len(self.feature_columns):
            print(f"Warning: Expected {len(self.feature_columns)} features, got {len(landmarks)}")
            landmarks = landmarks[:len(self.feature_columns)]

        prediction = self.model.predict([landmarks])[0]
        confidence = max(self.model.predict_proba([landmarks])[0])

        return prediction, confidence

    def is_ready(self) -> bool:
        return self.model is not None

