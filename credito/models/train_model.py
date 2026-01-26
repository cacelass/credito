import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from credito.utils.paths import MODELS_DIR

def train_models(X_train, y_train):
    """
    Entrena múltiples modelos y los guarda en disco.
    """
    print("--> Entrenando modelos...")
    models = {}

    # Decision Tree
    print("    Entrenando Decision Tree...")
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    models['DecisionTree'] = dt

    # Guardar modelos
    for name, model in models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    
    return models