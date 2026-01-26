# credito/features/build_features.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from credito.utils.paths import ARTIFACTS_DIR

def preprocess_data(df, target_col='y', save_artifacts=True):
    """
    Procesa los datos para entrenamiento y guarda los codificadores.
    """
    print("--> Preprocesando datos de entrenamiento...")
    
    # 1. Limpieza
    df = df.drop_duplicates()
    
    # Separar X e y
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        # Caso para predicción sin target
        X = df
        y = None

    # Guardamos el orden de las columnas para pedirselas al usuario luego
    if save_artifacts:
        joblib.dump(X.columns.tolist(), ARTIFACTS_DIR / "columns.joblib")

    # 2. Categóricas
    cat_cols = X.select_dtypes(include=['object']).columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        # Convertir a string para evitar errores
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    if save_artifacts:
        joblib.dump(encoders, ARTIFACTS_DIR / "encoders.joblib")

    # 3. Escalado
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    if save_artifacts:
        joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")

    # Retorno
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        return X_scaled

def process_input(user_data):
    """
    Toma un diccionario con los datos del usuario y los transforma
    usando los artefactos guardados.
    """
    # 1. Cargar artefactos
    try:
        columns = joblib.load(ARTIFACTS_DIR / "columns.joblib")
        encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
        scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    except FileNotFoundError:
        raise Exception("No se encontraron los archivos de entrenamiento. Entrena el modelo primero.")

    # 2. Crear DataFrame con las columnas correctas
    df = pd.DataFrame([user_data])
    
    # Asegurar que el orden de columnas es el mismo que en el entrenamiento
    df = df[columns]

    # 3. Aplicar Encoders (Categorías)
    for col, le in encoders.items():
        # Manejo básico de errores si el usuario pone algo desconocido
        try:
            df[col] = le.transform(df[col].astype(str))
        except ValueError:
            # Si el valor no existe (ej: Trabajo='Youtuber'), asignamos un valor por defecto o fallamos
            print(f"Advertencia: El valor '{df[col].iloc[0]}' en '{col}' no se vio en el entrenamiento.")
            df[col] = 0 # Asignamos 0 por defecto (o podrías lanzar error)

    # 4. Aplicar Scaler (Numéricos)
    df_scaled = scaler.transform(df)
    
    return df_scaled