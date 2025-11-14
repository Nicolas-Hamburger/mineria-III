"""
FASE 3: Aplicativo Web con Streamlit
Dashboard de Machine Learning - Predicción de Proyectos Públicos
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# ML
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Configuración de página
st.set_page_config(
    page_title="🎯 Predictor de Proyectos Públicos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
    <style>
    .main { padding: 0rem 0rem; }
    .metric-box { 
        background-color: #f0f2f6; 
        padding: 20px; 
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box { background-color: #d4edda; padding: 15px; border-radius: 5px; }
    .warning-box { background-color: #fff3cd; padding: 15px; border-radius: 5px; }
    .danger-box { background-color: #f8d7da; padding: 15px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# ==================== FUNCIONES ====================

@st.cache_resource
def cargar_modelos():
    """Carga los modelos entrenados"""
    try:
        with open('models/random_forest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('models/xgboost_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open('models/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return rf_model, xgb_model, encoders, scaler
    except FileNotFoundError:
        st.error("❌ Los modelos no se han entrenado. Ejecuta primero el script de entrenamiento.")
        return None, None, None, None

@st.cache_data
def cargar_datos_csv():
    """Carga los datos limpios"""
    try:
        df = pd.read_csv('datos_limpios.csv')
        return df
    except FileNotFoundError:
        return None

def crear_features(valor_inicial, valor_adicional, anio, mes):
    """Crea features para predicción"""
    valor_total = valor_inicial + valor_adicional
    tiene_adicional = 1 if valor_adicional > 0 else 0
    ratio = valor_adicional / (valor_inicial + 1)
    log_total = np.log1p(valor_total)
    
    return valor_inicial, valor_adicional, valor_total, anio, mes, tiene_adicional, ratio, log_total

def hacer_prediccion(modelo, scaler, encoders, datos_input):
    """Realiza predicción con el modelo"""
    data = pd.DataFrame([datos_input], columns=[
        'valor_inicial_proyecto', 'valor_adicional', 'valor_total_proyecto',
        'anio', 'mes', 'tiene_adicional', 'ratio_adicional', 'log_valor_total',
        'sector_encoded', 'municipio_encoded', 'entidad_presenta_encoded'
    ])
    
    data_scaled = scaler.transform(data)
    prediccion = modelo.predict(data_scaled)[0]
    probabilidad = modelo.predict_proba(data_scaled)[0]
    
    resultado = encoders['estado_proyecto'].inverse_transform([prediccion])[0]
    confianza = max(probabilidad) * 100
    
    return resultado, confianza, probabilidad

def mostrar_metricas(rf_model, xgb_model, encoders, scaler, df):
    """Muestra métricas de los modelos"""
    
    # Preparar datos
    df_clean = df.copy()
    df_clean['fecha_radicacion'] = pd.to_datetime(df_clean['fecha_radicacion'], errors='coerce')
    df_clean['anio'] = df_clean['fecha_radicacion'].dt.year.fillna(df_clean['fecha_radicacion'].dt.year.mode()[0])
    df_clean['mes'] = df_clean['fecha_radicacion'].dt.month.fillna(df_clean['fecha_radicacion'].dt.month.mode()[0])
    df_clean['tiene_adicional'] = (df_clean['valor_adicional'] > 0).astype(int)
    df_clean['ratio_adicional'] = df_clean['valor_adicional'] / (df_clean['valor_inicial_proyecto'] + 1)
    df_clean['log_valor_total'] = np.log1p(df_clean['valor_total_proyecto'])
    
    # Features
    features = [
        'valor_inicial_proyecto', 'valor_adicional', 'valor_total_proyecto',
        'anio', 'mes', 'tiene_adicional', 'ratio_adicional', 'log_valor_total',
        'sector_encoded', 'municipio_encoded', 'entidad_presenta_encoded'
    ]
    
    X = df_clean[features]
    y = df_clean['estado_encoded']
    
    # SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_bal, y_bal = smote.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.25, random_state=42, stratify=y_bal
    )
    
    # Predicciones RF
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf, average='macro', zero_division=0)
    rec_rf = recall_score(y_test, y_pred_rf, average='macro', zero_division=0)
    f1_rf = f1_score(y_test, y_pred_rf, average='macro', zero_division=0)
    
    # Predicciones XGB
    y_pred_xgb = xgb_model.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    prec_xgb = precision_score(y_test, y_pred_xgb, average='macro', zero_division=0)
    rec_xgb = recall_score(y_test, y_pred_xgb, average='macro', zero_division=0)
    f1_xgb = f1_score(y_test, y_pred_xgb, average='macro', zero_division=0)
    
    return {
        'rf': {
            'accuracy': acc_rf,
            'precision': prec_rf,
            'recall': rec_rf,
            'f1': f1_rf,
            'y_pred': y_pred_rf,
            'y_test': y_test
        },
        'xgb': {
            'accuracy': acc_xgb,
            'precision': prec_xgb,
            'recall': rec_xgb,
            'f1': f1_xgb,
            'y_pred': y_pred_xgb,
            'y_test': y_test
        }
    }

# ==================== INTERFAZ STREAMLIT ====================

# HEADER
st.title("🎯 Predictor de Proyectos Públicos")
st.markdown("### Dashboard de Machine Learning para Proyectos del Sector Público")

# SIDEBAR
with st.sidebar:
    st.markdown("## 🔧 Configuración")
    pagina = st.radio(
        "Selecciona una sección:",
        ["📊 Dashboard", "🔮 Predicción", "📈 Métricas", "📋 Información"]
    )

# Cargar modelos
rf_model, xgb_model, encoders, scaler = cargar_modelos()
df = cargar_datos_csv()

if rf_model is None or df is None:
    st.error("❌ Error al cargar datos o modelos")
    st.stop()

# ==================== PÁGINA: DASHBOARD ====================
if pagina == "📊 Dashboard":
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📁 Total de Proyectos", len(df))
    
    with col2:
        aprobados = sum(df['estado_proyecto'] == 'APROBADO')
        st.metric("✅ Proyectos Aprobados", aprobados)
    
    with col3:
        no_aprobados = len(df) - aprobados
        st.metric("❌ Proyectos No Aprobados", no_aprobados)
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Estados")
        estado_counts = df['estado_proyecto'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(estado_counts.index, estado_counts.values, color=['#2ecc71', '#e74c3c'])
        ax.set_xlabel('Cantidad')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Top 10 Sectores")
        sector_counts = df['sector'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(sector_counts.index, sector_counts.values, color='#3498db')
        ax.set_xlabel('Cantidad')
        st.pyplot(fig)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Valores Totales")
        fig, ax = plt.subplots(figsize=(8, 6))
        df['valor_total_proyecto'].hist(bins=50, ax=ax, edgecolor='black', color='#9b59b6')
        ax.set_xlabel('Valor Total')
        ax.set_ylabel('Frecuencia')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Proyectos con/sin Valor Adicional")
        adicional = df['valor_adicional'].sum()
        sin_adicional = len(df) - (df['valor_adicional'] > 0).sum()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
            [df['valor_adicional'].sum(), len(df) - (df['valor_adicional'] > 0).sum()],
            labels=['Con Adicional', 'Sin Adicional'],
            autopct='%1.1f%%',
            colors=['#f39c12', '#34495e']
        )
        st.pyplot(fig)

# ==================== PÁGINA: PREDICCIÓN ====================
elif pagina == "🔮 Predicción":
    st.markdown("---")
    st.subheader("Realiza predicciones ingresando los datos del proyecto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Valores")
        valor_inicial = st.number_input(
            "Valor inicial del proyecto",
            min_value=0.0,
            value=100000.0,
            step=10000.0
        )
        valor_adicional = st.number_input(
            "Valor adicional",
            min_value=0.0,
            value=0.0,
            step=10000.0
        )
    
    with col2:
        st.markdown("### 📅 Información")
        anio = st.number_input("Año de radicación", min_value=2010, max_value=2025, value=2024)
        mes = st.number_input("Mes de radicación", min_value=1, max_value=12, value=1)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🏢 Datos Categóricos")
        sector = st.selectbox(
            "Sector",
            sorted(df['sector'].dropna().unique())
        )
    
    with col2:
        municipio = st.selectbox(
            "Municipio",
            sorted(df['municipio'].dropna().unique())
        )
    
    with col3:
        entidad = st.selectbox(
            "Entidad que presenta",
            sorted(df['entidad_presenta'].dropna().unique())
        )
    
    st.markdown("---")
    
    # Botón de predicción
    if st.button("🔮 Realizar Predicción", use_container_width=True):
        # Preparar datos
        val_ini, val_adi, val_tot, a, m, ten_adi, ratio, log_tot = crear_features(
            valor_inicial, valor_adicional, anio, mes
        )
        
        # Codificar categóricas
        try:
            sector_enc = encoders['sector'].transform([sector])[0] if sector in encoders['sector'].classes_ else 0
            municipio_enc = encoders['municipio'].transform([municipio])[0] if municipio in encoders['municipio'].classes_ else 0
            entidad_enc = encoders['entidad_presenta'].transform([entidad])[0] if entidad in encoders['entidad_presenta'].classes_ else 0
        except:
            sector_enc = municipio_enc = entidad_enc = 0
        
        datos_input = [val_ini, val_adi, val_tot, a, m, ten_adi, ratio, log_tot, sector_enc, municipio_enc, entidad_enc]
        
        # Predicciones
        resultado_rf, conf_rf, prob_rf = hacer_prediccion(rf_model, scaler, encoders, datos_input)
        resultado_xgb, conf_xgb, prob_xgb = hacer_prediccion(xgb_model, scaler, encoders, datos_input)
        
        st.markdown("---")
        st.subheader("📊 Resultados de Predicción")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌳 Random Forest")
            if resultado_rf == 'APROBADO':
                st.success(f"✅ Predicción: {resultado_rf}")
            else:
                st.error(f"❌ Predicción: {resultado_rf}")
            st.metric("Confianza", f"{conf_rf:.2f}%")
        
        with col2:
            st.markdown("### 🚀 XGBoost")
            if resultado_xgb == 'APROBADO':
                st.success(f"✅ Predicción: {resultado_xgb}")
            else:
                st.error(f"❌ Predicción: {resultado_xgb}")
            st.metric("Confianza", f"{conf_xgb:.2f}%")

# ==================== PÁGINA: MÉTRICAS ====================
elif pagina == "📈 Métricas":
    st.markdown("---")
    st.subheader("Métricas de Rendimiento de los Modelos")
    
    with st.spinner("Calculando métricas..."):
        metricas = mostrar_metricas(rf_model, xgb_model, encoders, scaler, df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌳 Random Forest")
        st.metric("Accuracy", f"{metricas['rf']['accuracy']*100:.2f}%")
        st.metric("Precision (Macro)", f"{metricas['rf']['precision']*100:.2f}%")
        st.metric("Recall (Macro)", f"{metricas['rf']['recall']*100:.2f}%")
        st.metric("F1-Score (Macro)", f"{metricas['rf']['f1']*100:.2f}%")
        
        # Matriz de confusión RF
        st.subheader("Matriz de Confusión - Random Forest")
        cm_rf = confusion_matrix(metricas['rf']['y_test'], metricas['rf']['y_pred'])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=encoders['estado_proyecto'].classes_,
                    yticklabels=encoders['estado_proyecto'].classes_)
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicho')
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 🚀 XGBoost")
        st.metric("Accuracy", f"{metricas['xgb']['accuracy']*100:.2f}%")
        st.metric("Precision (Macro)", f"{metricas['xgb']['precision']*100:.2f}%")
        st.metric("Recall (Macro)", f"{metricas['xgb']['recall']*100:.2f}%")
        st.metric("F1-Score (Macro)", f"{metricas['xgb']['f1']*100:.2f}%")
        
        # Matriz de confusión XGB
        st.subheader("Matriz de Confusión - XGBoost")
        cm_xgb = confusion_matrix(metricas['xgb']['y_test'], metricas['xgb']['y_pred'])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=ax,
                    xticklabels=encoders['estado_proyecto'].classes_,
                    yticklabels=encoders['estado_proyecto'].classes_)
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicho')
        st.pyplot(fig)

# ==================== PÁGINA: INFORMACIÓN ====================
elif pagina == "📋 Información":
    st.markdown("---")
    st.subheader("ℹ️ Información del Proyecto")
    
    st.markdown("""
    ### 🎯 Objetivo
    Este dashboard predice si un proyecto público será **APROBADO** o **NO APROBADO** 
    utilizando dos algoritmos de Machine Learning: **Random Forest** y **XGBoost**.
    
    ### 📊 Datos Utilizados
    - **Total de proyectos**: 1,021
    - **Clases**: APROBADO, NO_APROBADO
    - **Features**: Valores, fechas, sector, municipio, entidad
    
    ### 🤖 Modelos
    - **Random Forest**: 300 árboles, profundidad 12
    - **XGBoost**: 400 árboles, profundidad 7
    
    ### ✅ Métricas Alcanzadas
    - Accuracy: > 80%
    - Precision (Macro): > 85% ✅
    - Recall (Macro): > 85% ✅
    - F1-Score (Macro): > 85% ✅
    
    ### 🛠️ Tecnologías
    - Python, Streamlit, Scikit-learn
    - XGBoost, Pandas, NumPy
    - Matplotlib, Seaborn
    
    ### 👨‍💻 Desarrollado por
    Minería de Datos III - Proyecto de ML
    """)
    
    st.markdown("---")
    
    st.subheader("📁 Archivos Generados")
    st.write("""
    - `modelo_entrenamiento.py` - Script de entrenamiento
    - `app.py` - Este dashboard Streamlit
    - `models/random_forest_model.pkl` - Modelo RF
    - `models/xgboost_model.pkl` - Modelo XGB
    - `models/label_encoders.pkl` - Encoders
    - `datos_limpios.csv` - Dataset procesado
    """)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <p>Desarrollado con ❤️ usando Streamlit | © 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)