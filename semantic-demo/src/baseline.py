from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _load_split(random_state: int):
    data = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=random_state, stratify=data.target
    )
    return X_train, X_test, y_train, y_test


def run_pca_logistic_baseline(n_components: int = 16, random_state: int = 0):
    X_train, X_test, y_train, y_test = _load_split(random_state)
    projector = PCA(n_components=n_components, random_state=random_state)
    X_train_reduced = projector.fit_transform(X_train)
    X_test_reduced = projector.transform(X_test)
    clf = LogisticRegression(max_iter=2000, random_state=random_state)
    clf.fit(X_train_reduced, y_train)
    accuracy = clf.score(X_test_reduced, y_test)
    return {
        "method": "pca_logistic",
        "n_components": n_components,
        "accuracy": float(accuracy),
        "input_dim": int(X_train.shape[1]),
    }


def run_supervised_projection_baseline(n_components: int = 16, random_state: int = 0):
    X_train, X_test, y_train, y_test = _load_split(random_state)
    projector = LinearDiscriminantAnalysis(n_components=min(n_components, len(set(y_train)) - 1))
    X_train_reduced = projector.fit_transform(X_train, y_train)
    X_test_reduced = projector.transform(X_test)
    clf = LogisticRegression(max_iter=2000, random_state=random_state)
    clf.fit(X_train_reduced, y_train)
    accuracy = clf.score(X_test_reduced, y_test)
    return {
        "method": "supervised_projection",
        "n_components": int(X_train_reduced.shape[1]),
        "accuracy": float(accuracy),
        "input_dim": int(X_train.shape[1]),
    }