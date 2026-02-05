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
    2. Label Encoding para variables categóricas.
    3. Escalado Min-Max para todas las variables.
    """
    print("--> Preprocesando datos...")
    
    # 1. Limpieza: Eliminación de duplicados (identificados 12 en el train set del notebook)
    df = df.drop_duplicates()
    
    # Separar X e y
    if target_col in df.columns:
        X = df.drop(columns=[target_col]).copy()
        # En el notebook, 'y' ya viene como numérica (0, 1), 
        # pero nos aseguramos por si acaso
        y = df[target_col]
    else:
        X = df.copy()
        y = None

    # Guardar el orden de las columnas original
    if save_artifacts:
        joblib.dump(X.columns.tolist(), ARTIFACTS_DIR / "columns.joblib")

    # 2. Variables Categóricas: LabelEncoder (como se usa en el notebook)
    # Identificamos columnas tipo objeto: job, marital, education, default, housing, 
    # loan, contact, month, day_of_week, poutcome
    cat_cols = X.select_dtypes(include=['object']).columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        # El notebook asume que no hay nulos (confirmado en el EDA)
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    if save_artifacts:
        joblib.dump(encoders, ARTIFACTS_DIR / "encoders.joblib")

    # 3. Escalado: MinMaxScaler (utilizado en las celdas de preprocesado del proyecto)
    scaler = MinMaxScaler()
    # Aplicamos el escalado a todo el conjunto X transformado
    X_scaled_array = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)
    
    if save_artifacts:
        joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")

    # Retorno con split 80/20 como es estándar (el notebook usa train/test pre-separados,
    # pero para el pipeline build_features es mejor devolver el split del train set)
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        return X_scaled

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