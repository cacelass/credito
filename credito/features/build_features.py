# credito/features/build_features.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from credito.utils.paths import ARTIFACTS_DIR

def preprocess_data(df, target_col='y', save_artifacts=True):
    """
    Procesa los datos siguiendo la lógica del proyecto:
    1. Eliminación de duplicados.
    2. Split train/test ANTES de ajustar los transformadores.
    3. Label Encoding y escalado Min-Max ajustados SOLO con el train set.
    """
    print("--> Preprocesando datos...")
    
    # 1. Limpieza: Eliminación de duplicados
    df = df.drop_duplicates()
    
    # Separar X e y
    if target_col in df.columns:
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col]
    else:
        X = df.copy()
        y = None

    # 2. Split 80/20 con estratificación ANTES de ajustar los transformadores.
    #    Ajustar scaler/encoders con todo el dataset filtra información del test set.
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, None, None

    # Guardar el orden de las columnas original
    if save_artifacts:
        joblib.dump(X.columns.tolist(), ARTIFACTS_DIR / "columns.joblib")

    # 3. Variables Categóricas: LabelEncoder ajustado SOLO con el train set
    cat_cols = X_train.select_dtypes(include=['object']).columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        if X_test is not None:
            X_test[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le
    
    if save_artifacts:
        joblib.dump(encoders, ARTIFACTS_DIR / "encoders.joblib")

    # 4. Escalado: MinMaxScaler ajustado SOLO con el train set
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns
        )
    
    if save_artifacts:
        joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")

    if y is not None:
        return X_train_scaled, X_test_scaled, y_train, y_test
    else:
        return X_train_scaled

def process_input(user_data):
    """
    Transforma la entrada del usuario para predicción usando los artefactos entrenados.
    """
    try:
        columns = joblib.load(ARTIFACTS_DIR / "columns.joblib")
        encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
        scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    except FileNotFoundError:
        raise Exception("Artefactos no encontrados. Ejecute el entrenamiento primero.")

    # Crear DataFrame y asegurar orden de columnas
    df = pd.DataFrame([user_data])
    df = df[columns]

    # Aplicar Encoders guardados
    for col, le in encoders.items():
        try:
            # Intentamos transformar el valor del usuario
            df[col] = le.transform(df[col].astype(str))
        except ValueError:
            print(f"Advertencia: Valor desconocido en '{col}'. Usando clase por defecto.")
            # Si el valor es nuevo, asignamos la clase 0 o la más común
            df[col] = 0 

    # Aplicar Escalado
    df_scaled = scaler.transform(df)
    
    return df_scaled