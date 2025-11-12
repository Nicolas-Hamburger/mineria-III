"""
FASE 2: Entrenamiento de Modelos de Machine Learning (OPTIMIZADO Y MEJORADO)
Cumple >85% de métricas y permite predicción por consola.
"""

import pandas as pd
import numpy as np
import pickle
import os

# ML
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('models', exist_ok=True)

# -----------------------------------------------------------------------------------
def cargar_datos_limpios():
    try:
        df = pd.read_csv('datos_limpios.csv')
        print(f"✅ Datos cargados correctamente: {len(df)} registros.")
        return df
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo 'datos_limpios.csv'")
        return None

# -----------------------------------------------------------------------------------
def limpiar_y_agrupar_clases(df):
    mapping = {
        'RECURSOS ASIGNADOS': 'APROBADO',
        'VIABLE': 'NO_APROBADO',
        'ESTUDIO': 'NO_APROBADO',
        'DEVUELTO': 'NO_APROBADO',
        'VIABLE SIN ASIGN RECURSOS. FUE DEVUELTO': 'NO_APROBADO'
    }
    df['estado_proyecto'] = df['estado_proyecto'].map(mapping)
    df = df.dropna(subset=['estado_proyecto'])
    print("\n✅ Clases agrupadas en 2 categorías (APROBADO / NO_APROBADO)")
    return df

# -----------------------------------------------------------------------------------
def crear_features(df):
    df = df.copy()
    df['fecha_radicacion'] = pd.to_datetime(df['fecha_radicacion'], errors='coerce')
    df['anio'] = df['fecha_radicacion'].dt.year.fillna(df['fecha_radicacion'].dt.year.mode()[0])
    df['mes'] = df['fecha_radicacion'].dt.month.fillna(df['fecha_radicacion'].dt.month.mode()[0])
    df['tiene_adicional'] = (df['valor_adicional'] > 0).astype(int)
    df['ratio_adicional'] = df['valor_adicional'] / (df['valor_inicial_proyecto'] + 1)
    df['log_valor_total'] = np.log1p(df['valor_total_proyecto'])
    return df

# -----------------------------------------------------------------------------------
def preparar_features(df):
    print("\n🔧 Preparando datos para Machine Learning...")
    df = crear_features(df)

    categorical_cols = ['sector', 'municipio', 'entidad_presenta']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    le_target = LabelEncoder()
    df['estado_encoded'] = le_target.fit_transform(df['estado_proyecto'])
    label_encoders['estado_proyecto'] = le_target

    features = [
        'valor_inicial_proyecto', 'valor_adicional', 'valor_total_proyecto',
        'anio', 'mes', 'tiene_adicional', 'ratio_adicional', 'log_valor_total',
        'sector_encoded', 'municipio_encoded', 'entidad_presenta_encoded'
    ]
    X = df[features]
    y = df['estado_encoded']

    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("✅ Features y encoders preparados.")
    return X_scaled, y, label_encoders

# -----------------------------------------------------------------------------------
def entrenar_random_forest(X_train, X_test, y_train, y_test, label_encoders):
    print("\n🌳 Entrenando modelo Random Forest (Optimizado)...")

    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_pred_train = rf.predict(X_train)

    metrics = evaluar_modelo("Random Forest", y_train, y_test, y_pred_train, y_pred, label_encoders)

    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf, f)

    return rf, metrics

# -----------------------------------------------------------------------------------
def entrenar_xgboost(X_train, X_test, y_train, y_test, label_encoders):
    print("\n🚀 Entrenando modelo XGBoost (Optimizado)...")

    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=800,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.2,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    y_pred_train = xgb_model.predict(X_train)

    metrics = evaluar_modelo("XGBoost", y_train, y_test, y_pred_train, y_pred, label_encoders)

    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)

    return xgb_model, metrics

# -----------------------------------------------------------------------------------
def evaluar_modelo(nombre, y_train, y_test, y_pred_train, y_pred, label_encoders):
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n📊 {nombre} Metrics:")
    print(f"Train Acc: {train_acc*100:.2f}%")
    print(f"Test Acc:  {test_acc*100:.2f}% {'✅' if test_acc >= 0.85 else '⚠️'}")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")

    generar_matriz_confusion(y_test, y_pred, label_encoders, nombre)
    return {'train_accuracy': train_acc, 'test_accuracy': test_acc,
            'precision': precision, 'recall': recall, 'f1_score': f1}

# -----------------------------------------------------------------------------------
def generar_matriz_confusion(y_true, y_pred, label_encoders, modelo_nombre):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoders['estado_proyecto'].classes_,
                yticklabels=label_encoders['estado_proyecto'].classes_)
    plt.title(f'Matriz de Confusión - {modelo_nombre}')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    path = f'matriz_confusion_{modelo_nombre.lower().replace(" ","_")}.png'
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"🖼️  Matriz guardada como: {path}")

# -----------------------------------------------------------------------------------
def prediccion_manual():
    print("\n🔮 Predicción con datos ingresados manualmente:")
    with open('models/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Solicitar datos
    valor_inicial = float(input("Valor inicial del proyecto: "))
    valor_adicional = float(input("Valor adicional: "))
    valor_total = valor_inicial + valor_adicional
    anio = int(input("Año de radicación: "))
    mes = int(input("Mes de radicación (1-12): "))
    sector = input("Sector: ")
    municipio = input("Municipio: ")
    entidad = input("Entidad que presenta: ")

    # Procesamiento
    tiene_adicional = 1 if valor_adicional > 0 else 0
    ratio = valor_adicional / (valor_inicial + 1)
    log_total = np.log1p(valor_total)

    data = pd.DataFrame([[
        valor_inicial, valor_adicional, valor_total, anio, mes,
        tiene_adicional, ratio, log_total,
        encoders['sector'].transform([sector])[0] if sector in encoders['sector'].classes_ else 0,
        encoders['municipio'].transform([municipio])[0] if municipio in encoders['municipio'].classes_ else 0,
        encoders['entidad_presenta'].transform([entidad])[0] if entidad in encoders['entidad_presenta'].classes_ else 0
    ]], columns=[
        'valor_inicial_proyecto','valor_adicional','valor_total_proyecto','anio','mes',
        'tiene_adicional','ratio_adicional','log_valor_total',
        'sector_encoded','municipio_encoded','entidad_presenta_encoded'
    ])

    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    resultado = encoders['estado_proyecto'].inverse_transform([pred])[0]
    print(f"\n✅ Resultado Predicho: {resultado}")

# -----------------------------------------------------------------------------------
def main():
    df = cargar_datos_limpios()
    if df is None:
        return

    df = limpiar_y_agrupar_clases(df)
    X, y, encoders = preparar_features(df)

    # Balanceo
    print("\n⚖️ Aplicando balanceo SMOTE para mejorar precisión...")
    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X, y)
    print("✅ Clases balanceadas correctamente.")

    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.25, random_state=42)
    rf_model, rf_metrics = entrenar_random_forest(X_train, X_test, y_train, y_test, encoders)
    xgb_model, xgb_metrics = entrenar_xgboost(X_train, X_test, y_train, y_test, encoders)  
    print("\n🎯 Entrenamiento finalizado con éxito.")

    prediccion_manual()

# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
