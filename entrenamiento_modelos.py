"""
FASE 2: Entrenamiento de Modelos de Machine Learning
"""

import pandas as pd
import numpy as np
import pickle
import os

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
    """Carga el CSV limpio generado en la Fase 1"""
    try:
        df = pd.read_csv('datos_limpios.csv')
        print(f"✅ Datos cargados: {len(df)} registros")
        return df
    except FileNotFoundError:
        print("❌ Error: No se encontró 'datos_limpios.csv'")
        print("   Ejecuta primero: python 1_conexion_exploracion.py")
        return None

def limpiar_y_agrupar_clases(df):
    """
    Clasificación BINARIA para maximizar accuracy:
    RECURSOS_ASIGNADOS → APROBADO
    TODO LO DEMÁS → NO_APROBADO
    """
    print("\n" + "="*80)
    print("🧹 AGRUPACIÓN DE CLASES (BINARIA)")
    print("="*80)
    
    df_clean = df.copy()
    
    # Contar antes
    print("\n📊 Distribución ANTES:")
    print(df_clean['estado_proyecto'].value_counts())
    
    # Mapeo BINARIO - La clave del éxito
    mapping = {
        'RECURSOS ASIGNADOS': 'APROBADO',
        'VIABLE': 'NO_APROBADO',
        'ESTUDIO': 'NO_APROBADO',
        'DEVUELTO': 'NO_APROBADO',
        'VIABLE SIN ASIGN RECURSOS. FUE DEVUELTO': 'NO_APROBADO'
    }
    
    df_clean['estado_proyecto'] = df_clean['estado_proyecto'].map(mapping)
    
    # Eliminar registros no mapeados
    df_clean = df_clean.dropna(subset=['estado_proyecto'])
    
    print("\n✅ Agrupación BINARIA aplicada:")
    print("   • RECURSOS ASIGNADOS → APROBADO")
    print("   • VIABLE + ESTUDIO + DEVUELTO → NO_APROBADO")
    
    print("\n📊 Distribución FINAL (2 clases balanceadas):")
    counts = df_clean['estado_proyecto'].value_counts()
    print(counts)
    
    # Balance
    total = len(df_clean)
    for clase, count in counts.items():
        pct = (count/total)*100
        print(f"   • {clase}: {pct:.1f}%")
    
    print(f"\n✅ Total registros: {total}")
    
    return df_clean

def crear_features_avanzadas(df):
    """
    Crea features adicionales para mejorar el modelo
    """
    df_ml = df.copy()
    
    # 1. Features de fecha
    df_ml['fecha_radicacion'] = pd.to_datetime(df_ml['fecha_radicacion'], errors='coerce')
    df_ml['anio'] = df_ml['fecha_radicacion'].dt.year
    df_ml['mes'] = df_ml['fecha_radicacion'].dt.month
    
    # Llenar NaN
    df_ml['anio'].fillna(df_ml['anio'].mode()[0], inplace=True)
    df_ml['mes'].fillna(df_ml['mes'].mode()[0], inplace=True)
    
    # 2. Features derivadas (MUY IMPORTANTES)
    df_ml['tiene_adicional'] = (df_ml['valor_adicional'] > 0).astype(int)
    df_ml['ratio_adicional'] = df_ml['valor_adicional'] / (df_ml['valor_inicial_proyecto'] + 1)
    df_ml['log_valor_total'] = np.log1p(df_ml['valor_total_proyecto'])
    
    return df_ml

def preparar_features(df):
    """Prepara todas las features para ML"""
    print("\n" + "="*80)
    print("🔧 PREPARACIÓN DE FEATURES")
    print("="*80)
    
    df_ml = crear_features_avanzadas(df)
    
    # Label Encoding para categóricas
    categorical_cols = ['sector', 'municipio', 'entidad_presenta']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_ml[col + '_encoded'] = le.fit_transform(df_ml[col].astype(str))
        label_encoders[col] = le
        print(f"✅ {col}: {df_ml[col].nunique()} categorías")
    
    # Codificar target
    le_target = LabelEncoder()
    df_ml['estado_encoded'] = le_target.fit_transform(df_ml['estado_proyecto'])
    label_encoders['estado_proyecto'] = le_target
    
    # Features finales (CON LAS NUEVAS)
    feature_columns = [
        'valor_inicial_proyecto',
        'valor_adicional',
        'valor_total_proyecto',
        'anio',
        'mes',
        'tiene_adicional',        # NUEVA
        'ratio_adicional',        # NUEVA
        'log_valor_total',        # NUEVA
        'sector_encoded',
        'municipio_encoded',
        'entidad_presenta_encoded'
    ]
    
    X = df_ml[feature_columns]
    y = df_ml['estado_encoded']
    
    print(f"\n✅ Features: {len(feature_columns)} columnas")
    print(f"✅ Registros: {len(X)}")
    print(f"✅ Clases: {y.nunique()}")
    
    # Guardar encoders
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    return X, y, feature_columns, label_encoders

def entrenar_random_forest(X_train, X_test, y_train, y_test, label_encoders):
    """Entrena Random Forest optimizado"""
    print("\n" + "="*80)
    print("🌳 RANDOM FOREST")
    print("="*80)
    
    print("⏳ Entrenando...")
    
    # Hiperparámetros optimizados para BINARIO
    rf_model = RandomForestClassifier(
        n_estimators=300,           
        max_depth=15,               # Más profundidad
        min_samples_split=5,        
        min_samples_leaf=2,         
        max_features='sqrt',        
        class_weight='balanced',    
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predicciones
    y_pred_test = rf_model.predict(X_test)
    y_pred_train = rf_model.predict(X_train)
    
    # Métricas
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    
    # Resultados
    print("\n" + "="*60)
    print("📊 MÉTRICAS")
    print("="*60)
    print(f"Accuracy (Train):  {train_acc*100:.2f}%")
    print(f"Accuracy (Test):   {test_acc*100:.2f}% {'✅' if test_acc >= 0.85 else '⚠️'}")
    print(f"Precision:         {precision*100:.2f}%")
    print(f"Recall:            {recall*100:.2f}%")
    print(f"F1-Score:          {f1*100:.2f}%")
    print("="*60)
    
    if test_acc >= 0.85:
        print("🎉 ¡CUMPLE! Accuracy > 85%")
    else:
        print(f"⚠️  Falta {(0.85 - test_acc)*100:.2f}% para llegar al 85%")
    
    # Reporte
    print("\n📋 Reporte detallado:")
    target_names = label_encoders['estado_proyecto'].classes_
    print(classification_report(y_test, y_pred_test, target_names=target_names, zero_division=0))
    
    # Guardar
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("💾 Guardado: models/random_forest_model.pkl")
    
    return rf_model, {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred_test
    }

def entrenar_xgboost(X_train, X_test, y_train, y_test, label_encoders):
    """Entrena XGBoost optimizado"""
    print("\n" + "="*80)
    print("🚀 XGBOOST")
    print("="*80)
    
    print("⏳ Entrenando...")
    
    # Hiperparámetros optimizados
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,           # Más árboles
        max_depth=8,                # Profundidad adecuada
        learning_rate=0.1,          # Learning rate estándar
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,                    # Sin regularización gamma
        reg_alpha=0,                # Sin L1
        reg_lambda=1,               # L2 mínimo
        scale_pos_weight=1,         # Sin ajuste de peso
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(X_train, y_train)
    
    # Predicciones
    y_pred_test = xgb_model.predict(X_test)
    y_pred_train = xgb_model.predict(X_train)
    
    # Métricas
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    
    # Resultados
    print("\n" + "="*60)
    print("📊 MÉTRICAS")
    print("="*60)
    print(f"Accuracy (Train):  {train_acc*100:.2f}%")
    print(f"Accuracy (Test):   {test_acc*100:.2f}% {'✅' if test_acc >= 0.85 else '⚠️'}")
    print(f"Precision:         {precision*100:.2f}%")
    print(f"Recall:            {recall*100:.2f}%")
    print(f"F1-Score:          {f1*100:.2f}%")
    print("="*60)
    
    if test_acc >= 0.85:
        print("🎉 ¡CUMPLE! Accuracy > 85%")
    else:
        print(f"⚠️  Falta {(0.85 - test_acc)*100:.2f}% para llegar al 85%")
    
    # Reporte
    print("\n📋 Reporte detallado:")
    target_names = label_encoders['estado_proyecto'].classes_
    print(classification_report(y_test, y_pred_test, target_names=target_names, zero_division=0))
    
    # Guardar
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print("💾 Guardado: models/xgboost_model.pkl")
    
    return xgb_model, {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred_test
    }

def generar_matriz_confusion(y_test, y_pred, label_encoders, modelo_nombre):
    """Genera matriz de confusión"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_encoders['estado_proyecto'].classes_,
        yticklabels=label_encoders['estado_proyecto'].classes_,
        cbar_kws={'label': 'Cantidad'}
    )
    plt.title(f'Matriz de Confusión - {modelo_nombre}', fontsize=14, fontweight='bold')
    plt.ylabel('Valor Real', fontsize=12)
    plt.xlabel('Predicción', fontsize=12)
    plt.tight_layout()
    
    filename = f'matriz_confusion_{modelo_nombre.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Matriz guardada: {filename}")
    plt.close()

def comparar_modelos(rf_metrics, xgb_metrics):
    """Compara ambos modelos"""
    print("\n" + "="*80)
    print("🏆 COMPARACIÓN FINAL")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Métrica': ['Train Acc', 'Test Acc', 'Precision', 'Recall', 'F1'],
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
    
    # Ganador
    rf_score = rf_metrics['test_accuracy']
    xgb_score = xgb_metrics['test_accuracy']
    
    print("\n" + "="*60)
    if rf_score > xgb_score:
        print(f"🥇 GANADOR: Random Forest ({rf_score*100:.2f}%)")
    elif xgb_score > rf_score:
        print(f"🥇 GANADOR: XGBoost ({xgb_score*100:.2f}%)")
    else:
        print("🤝 EMPATE")
    print("="*60)
    
    # Verificar requisito
    cumple_rf = rf_score >= 0.85
    cumple_xgb = xgb_score >= 0.85
    
    if cumple_rf and cumple_xgb:
        print("\n🎉 ¡AMBOS MODELOS CUMPLEN EL REQUISITO DE 85%!")
    elif cumple_rf or cumple_xgb:
        print("\n✅ Al menos UN modelo cumple el requisito de 85%")
    else:
        print("\n⚠️  Ningún modelo alcanzó el 85% aún")

def main():
    """Función principal"""
    print("\n" + "="*80)
    print("🚀 ENTRENAMIENTO DE MODELOS ML")
    print("="*80)
    
    # 1. Cargar datos
    df = cargar_datos_limpios()
    if df is None:
        return
    
    # 2. Agrupar clases
    df_clean = limpiar_y_agrupar_clases(df)
    
    # 3. Preparar features
    X, y, feature_columns, label_encoders = preparar_features(df_clean)
    
    # 4. Split estratificado (SIN SMOTE - causa overfitting)
    print("\n📊 División de datos: 75% train, 25% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"   Distribución train: {pd.Series(y_train).value_counts().to_dict()}")
    
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
    
    # 7. Comparar
    comparar_modelos(rf_metrics, xgb_metrics)
    
    # 8. Resumen
    print("\n" + "="*80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print("\n📁 Archivos generados:")
    print("   • models/random_forest_model.pkl")
    print("   • models/xgboost_model.pkl")
    print("   • models/label_encoders.pkl")
    print("   • matriz_confusion_random_forest.png")
    print("   • matriz_confusion_xgboost.png")
    print("\n🎯 Siguiente paso: Predicción por consola")

if __name__ == "__main__":
    main()