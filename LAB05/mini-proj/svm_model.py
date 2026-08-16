from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocess import build_feature_pipeline


def train_svm(X_train, y_train, pca_components=50):
    # Feature extraction (one-hot + TF-IDF) + scaling + PCA in one pipeline
    # so the test split always gets the exact same fitted transform.
    # PCA also keeps the RBF kernel fast once Brand/Country one-hot columns
    # and the 150 TF-IDF columns are combined.
    scaler = Pipeline([
        ("features", build_feature_pipeline()),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_components, whiten=True, random_state=42)),
    ])

    # Fit and transform training data
    X_train_scaled = scaler.fit_transform(X_train, y_train)

    # Create SVM model
    model = SVC(
        kernel="rbf", C=10, gamma="scale", cache_size=1000, class_weight="balanced"
    )

    # Train model
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict_svm(model, scaler, X_test):

    # Apply the same feature extraction + scaling used for training data
    X_test_scaled = scaler.transform(X_test)
    # Predict
    predictions = model.predict(X_test_scaled)

    return predictions
