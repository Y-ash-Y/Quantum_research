from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_models():
    """Classical baselines. Features are on different scales, so the
    scale-sensitive models (LogReg, SVM) are wrapped with standardisation."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42),
        "SVM": make_pipeline(
            StandardScaler(), SVC(probability=True, random_state=42)),
    }
    return models