import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import xgboost as xgb

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Crear carpeta para modelos
os.makedirs('models', exist_ok=True)

def cargar_datos_limpios():
    """
    Carga el CSV limpio generado en la Fase 1
    """
    try:
        df = pd.read_csv('datos_limpios.csv')
        print(f"✅ Datos cargados: {len(df)} registros")
        return df
    except FileNotFoundError:
        print("❌ Error: No se encontró 'datos_limpios.csv'")
        print("   Ejecuta primero: python 1_conexion_exploracion.py")
        return None

def limpiar_clases_minoritarias(df, min_samples=5):
    """
    Elimina clases con muy pocos registros (menos de min_samples)
    """
    print("\n" + "="*80)
    print("🧹 LIMPIEZA DE CLASES MINORITARIAS")
    print("="*80)
    
    # Contar registros por clase
    class_counts = df['estado_proyecto'].value_counts()
    print("\n📊 Distribución ANTES de limpiar:")
    print(class_counts)
    
    # Identificar clases a eliminar
    clases_a_eliminar = class_counts[class_counts < min_samples].index.tolist()
    
    if clases_a_eliminar:
        print(f"\n⚠️  Eliminando clases con menos de {min_samples} registros:")
        for clase in clases_a_eliminar:
            count = class_counts[clase]
            print(f"   • {clase}: {count} registros")
        
        # Filtrar dataset
        df_clean = df[~df['estado_proyecto'].isin(clases_a_eliminar)].copy()
        
        print(f"\n✅ Registros eliminados: {len(df) - len(df_clean)}")
        print(f"✅ Registros finales: {len(df_clean)}")
        
        print("\n📊 Distribución DESPUÉS de limpiar:")
        print(df_clean['estado_proyecto'].value_counts())
        
        return df_clean
    else:
        print("\n✅ No hay clases minoritarias que eliminar")
        return df

def preparar_features(df):
    """
    Prepara las features para machine learning:
    - Extrae características de fechas
    - Codifica variables categóricas
    - Separa X e y
    """
    print("\n" + "="*80)
    print("🔧 PREPARACIÓN DE FEATURES")
    print("="*80)
    
    df_ml = df.copy()
    
    # 1. Extraer características de fecha
    df_ml['fecha_radicacion'] = pd.to_datetime(df_ml['fecha_radicacion'], errors='coerce')
    df_ml['anio'] = df_ml['fecha_radicacion'].dt.year
    df_ml['mes'] = df_ml['fecha_radicacion'].dt.month
    df_ml['dia'] = df_ml['fecha_radicacion'].dt.day
    
    # Llenar NaN en fechas con valores por defecto
    df_ml['anio'].fillna(df_ml['anio'].mode()[0], inplace=True)
    df_ml['mes'].fillna(df_ml['mes'].mode()[0], inplace=True)
    df_ml['dia'].fillna(df_ml['dia'].mode()[0], inplace=True)
    
    print("✅ Características de fecha extraídas: año, mes, día")
    
    # 2. Label Encoding para variables categóricas
    categorical_cols = ['sector', 'municipio', 'entidad_presenta']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_ml[col + '_encoded'] = le.fit_transform(df_ml[col].astype(str))
        label_encoders[col] = le
        print(f"✅ Codificado: {col} ({df_ml[col].nunique()} categorías únicas)")
    
    # 3. Codificar variable objetivo
    le_target = LabelEncoder()
    df_ml['estado_encoded'] = le_target.fit_transform(df_ml['estado_proyecto'])
    label_encoders['estado_proyecto'] = le_target
    
    print(f"✅ Variable objetivo codificada: {df_ml['estado_proyecto'].nunique()} clases")
    
    # 4. Seleccionar features finales
    feature_columns = [
        'valor_inicial_proyecto',
        'valor_adicional',
        'valor_total_proyecto',
        'anio',
        'mes',
        'dia',
        'sector_encoded',
        'municipio_encoded',
        'entidad_presenta_encoded'
    ]
    
    X = df_ml[feature_columns]
    y = df_ml['estado_encoded']
    
    print(f"\n✅ Features finales: {len(feature_columns)} columnas")
    print(f"✅ Registros: {len(X)}")
    print(f"✅ Clases objetivo: {y.nunique()}")
    
    # Guardar encoders para usar en predicciones
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("✅ Encoders guardados en: models/label_encoders.pkl")
    
    return X, y, feature_columns, label_encoders

def entrenar_random_forest(X_train, X_test, y_train, y_test, label_encoders):
    """
    Entrena Random Forest y evalúa su desempeño
    """
    print("\n" + "="*80)
    print("🌳 ENTRENAMIENTO: RANDOM FOREST")
    print("="*80)
    
    # Entrenar modelo
    print("\n⏳ Entrenando Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    print("✅ Modelo entrenado")
    
    # Predicciones
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # Calcular métricas
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("📊 MÉTRICAS - RANDOM FOREST")
    print("="*60)
    print(f"Accuracy (Train):  {train_accuracy*100:.2f}%")
    print(f"Accuracy (Test):   {test_accuracy*100:.2f}% {'✅' if test_accuracy >= 0.85 else '⚠️'}")
    print(f"Precision:         {precision*100:.2f}%")
    print(f"Recall:            {recall*100:.2f}%")
    print(f"F1-Score:          {f1*100:.2f}%")
    print("="*60)
    
    if test_accuracy >= 0.85:
        print("🎉 ¡CUMPLE EL REQUISITO! Accuracy superior al 85%")
    else:
        print("⚠️  No alcanza el 85% requerido. Considera ajustar hiperparámetros.")
    
    # Reporte detallado
    print("\n📋 Reporte de Clasificación:")
    target_names = label_encoders['estado_proyecto'].classes_
    print(classification_report(y_test, y_pred_test, target_names=target_names, zero_division=0))
    
    # Guardar modelo
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("💾 Modelo guardado en: models/random_forest_model.pkl")
    
    # Importancia de características
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top 5 Features más importantes:")
    print(feature_importance.head())
    
    return rf_model, {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred_test,
        'feature_importance': feature_importance
    }

def entrenar_xgboost(X_train, X_test, y_train, y_test, label_encoders):
    """
    Entrena XGBoost y evalúa su desempeño
    """
    print("\n" + "="*80)
    print("🚀 ENTRENAMIENTO: XGBOOST")
    print("="*80)
    
    # Entrenar modelo
    print("\n⏳ Entrenando XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(X_train, y_train)
    print("✅ Modelo entrenado")
    
    # Predicciones
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)
    
    # Calcular métricas
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("📊 MÉTRICAS - XGBOOST")
    print("="*60)
    print(f"Accuracy (Train):  {train_accuracy*100:.2f}%")
    print(f"Accuracy (Test):   {test_accuracy*100:.2f}% {'✅' if test_accuracy >= 0.85 else '⚠️'}")
    print(f"Precision:         {precision*100:.2f}%")
    print(f"Recall:            {recall*100:.2f}%")
    print(f"F1-Score:          {f1*100:.2f}%")
    print("="*60)
    
    if test_accuracy >= 0.85:
        print("🎉 ¡CUMPLE EL REQUISITO! Accuracy superior al 85%")
    else:
        print("⚠️  No alcanza el 85% requerido. Considera ajustar hiperparámetros.")
    
    # Reporte detallado
    print("\n📋 Reporte de Clasificación:")
    target_names = label_encoders['estado_proyecto'].classes_
    print(classification_report(y_test, y_pred_test, target_names=target_names, zero_division=0))
    
    # Guardar modelo
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print("💾 Modelo guardado en: models/xgboost_model.pkl")
    
    # Importancia de características
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top 5 Features más importantes:")
    print(feature_importance.head())
    
    return xgb_model, {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred_test,
        'feature_importance': feature_importance
    }

def generar_matriz_confusion(y_test, y_pred, label_encoders, modelo_nombre):
    """
    Genera y guarda la matriz de confusión
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_encoders['estado_proyecto'].classes_,
        yticklabels=label_encoders['estado_proyecto'].classes_
    )
    plt.title(f'Matriz de Confusión - {modelo_nombre}', fontsize=14, fontweight='bold')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    
    filename = f'matriz_confusion_{modelo_nombre.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Matriz de confusión guardada: {filename}")
    plt.close()

def comparar_modelos(rf_metrics, xgb_metrics):
    """
    Compara ambos modelos y determina el mejor
    """
    print("\n" + "="*80)
    print("🏆 COMPARACIÓN DE MODELOS")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Métrica': ['Accuracy (Train)', 'Accuracy (Test)', 'Precision', 'Recall', 'F1-Score'],
        'Random Forest': [
            f"{rf_metrics['train_accuracy']*100:.2f}%",
            f"{rf_metrics['test_accuracy']*100:.2f}%",
            f"{rf_metrics['precision']*100:.2f}%",
            f"{rf_metrics['recall']*100:.2f}%",
            f"{rf_metrics['f1_score']*100:.2f}%"
        ],
        'XGBoost': [
            f"{xgb_metrics['train_accuracy']*100:.2f}%",
            f"{xgb_metrics['test_accuracy']*100:.2f}%",
            f"{xgb_metrics['precision']*100:.2f}%",
            f"{xgb_metrics['recall']*100:.2f}%",
            f"{xgb_metrics['f1_score']*100:.2f}%"
        ]
    })
    
    print("\n", comparison.to_string(index=False))
    
    # Determinar ganador
    rf_score = rf_metrics['test_accuracy']
    xgb_score = xgb_metrics['test_accuracy']
    
    print("\n" + "="*60)
    if rf_score > xgb_score:
        print("🥇 GANADOR: Random Forest")
        print(f"   Accuracy: {rf_score*100:.2f}% vs {xgb_score*100:.2f}%")
    elif xgb_score > rf_score:
        print("🥇 GANADOR: XGBoost")
        print(f"   Accuracy: {xgb_score*100:.2f}% vs {rf_score*100:.2f}%")
    else:
        print("🤝 EMPATE: Ambos modelos tienen el mismo desempeño")
    print("="*60)

def main():
    """
    Función principal
    """
    print("\n" + "="*80)
    print("🚀 INICIO DE ENTRENAMIENTO DE MODELOS ML")
    print("="*80)
    
    # 1. Cargar datos
    df = cargar_datos_limpios()
    if df is None:
        return
    
    # 2. Limpiar clases minoritarias
    df_clean = limpiar_clases_minoritarias(df, min_samples=5)
    
    # 3. Preparar features
    X, y, feature_columns, label_encoders = preparar_features(df_clean)
    
    # 4. Dividir train/test
    print("\n📊 Dividiendo datos: 80% train, 20% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} registros")
    print(f"   Test:  {len(X_test)} registros")
    
    # 5. Entrenar Random Forest
    rf_model, rf_metrics = entrenar_random_forest(
        X_train, X_test, y_train, y_test, label_encoders
    )
    generar_matriz_confusion(y_test, rf_metrics['y_pred'], label_encoders, 'Random Forest')
    
    # 6. Entrenar XGBoost
    xgb_model, xgb_metrics = entrenar_xgboost(
        X_train, X_test, y_train, y_test, label_encoders
    )
    generar_matriz_confusion(y_test, xgb_metrics['y_pred'], label_encoders, 'XGBoost')
    
    # 7. Comparar modelos
    comparar_modelos(rf_metrics, xgb_metrics)
    
    # 8. Resumen final
    print("\n" + "="*80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print("\n📁 Archivos generados:")
    print("   • models/random_forest_model.pkl")
    print("   • models/xgboost_model.pkl")
    print("   • models/label_encoders.pkl")
    print("   • matriz_confusion_random_forest.png")
    print("   • matriz_confusion_xgboost.png")

if __name__ == "__main__":
    main()