import joblib
from credito.utils.paths import MODELS_DIR
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train):
    """
    Entrena múltiples modelos y los guarda en disco.
    """
    print("--> Entrenando modelos...")
    models = {}

    print("    Entrenando el random forest...")
    dt =  RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',  
        max_depth=10,
        random_state=2
    )
    dt.fit(X_train, y_train)
    models['RandomForest'] = dt

    # Guardar modelos
    for name, model in models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    
    return models