"""
FASE 1: Conexión y Exploración de Datos
Este script se conecta a MySQL en Railway y explora los datos
"""

import pandas as pd
import mysql.connector
from mysql.connector import Error
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

def conectar_mysql():
    """
    Conecta a la base de datos MySQL en Railway
    """
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
        
        if connection.is_connected():
            print("✅ Conexión exitosa a MySQL")
            return connection
    
    except Error as e:
        print(f"❌ Error al conectar a MySQL: {e}")
        return None

def cargar_datos(connection):
    """
    Carga todos los datos de la tabla de proyectos
    Convierte columnas numéricas que vienen como strings a tipos numéricos.
    """
    query = """
    SELECT 
        codigo,
        fecha_radicacion,
        nombre_proyecto,
        valor_inicial_proyecto,
        valor_adicional,
        valor_total_proyecto,
        sector,
        municipio,
        entidad_presenta,
        estado_proyecto
    FROM railway.proyectos_publicos;
    """
    
    try:
        df = pd.read_sql(query, connection)

        # convertir fecha a datetime (si viene como string)
        if 'fecha_radicacion' in df.columns:
            df['fecha_radicacion'] = pd.to_datetime(df['fecha_radicacion'], errors='coerce')

        # columnas que deben ser numéricas; coercionar errores a NaN
        numeric_cols = ['valor_inicial_proyecto', 'valor_adicional', 'valor_total_proyecto']
        for col in numeric_cols:
            if col in df.columns:
                # eliminar símbolos no numéricos comunes (monedas, comas, espacios) y convertir
                df[col] = df[col].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"✅ Datos cargados: {len(df)} registros")
        print("   • NaNs en columnas numéricas:")
        print(df[[c for c in numeric_cols if c in df.columns]].isnull().sum())
        return df
    except Error as e:
        print(f"❌ Error al cargar datos: {e}")
        return None

def explorar_datos(df):
    """
    Realiza exploración inicial de los datos
    """
    print("\n" + "="*80)
    print("📊 EXPLORACIÓN DE DATOS")
    print("="*80)
    
    # Información básica
    print("\n1️⃣ INFORMACIÓN BÁSICA:")
    print(f"   • Total de registros: {len(df)}")
    print(f"   • Total de columnas: {len(df.columns)}")
    print(f"   • Columnas: {list(df.columns)}")
    
    # Tipos de datos
    print("\n2️⃣ TIPOS DE DATOS:")
    print(df.dtypes)
    
    # Valores nulos
    print("\n3️⃣ VALORES NULOS:")
    nulos = df.isnull().sum()
    if nulos.sum() > 0:
        print(nulos[nulos > 0])
    else:
        print("   ✅ No hay valores nulos")
    
    # Estadísticas de valores numéricos
    print("\n4️⃣ ESTADÍSTICAS DE VALORES:")
    print(df[['valor_inicial_proyecto', 'valor_adicional', 'valor_total_proyecto']].describe())
    
    # Distribución de estados (VARIABLE OBJETIVO)
    print("\n5️⃣ DISTRIBUCIÓN DE ESTADOS (Variable a predecir):")
    print(df['estado_proyecto'].value_counts())
    print(f"\n   Total de clases diferentes: {df['estado_proyecto'].nunique()}")
    
    # Distribución de sectores
    print("\n6️⃣ DISTRIBUCIÓN DE SECTORES:")
    print(df['sector'].value_counts().head(10))
    
    # Distribución de municipios
    print("\n7️⃣ TOP 10 MUNICIPIOS:")
    print(df['municipio'].value_counts().head(10))
    
    # Proyectos con valor adicional
    print("\n8️⃣ ANÁLISIS DE VALOR ADICIONAL:")
    # asegurar tipo numérico y manejar NaN
    if 'valor_adicional' in df.columns:
        adicional_series = pd.to_numeric(
            df['valor_adicional'].fillna('0').astype(str).str.replace(r'[^0-9\.\-]', '', regex=True),
            errors='coerce'
        ).fillna(0)
    else:
        adicional_series = pd.Series([0]*len(df))
    total = len(df) if len(df) > 0 else 1
    con_adicional = int((adicional_series > 0).sum())
    sin_adicional = int((adicional_series == 0).sum())
    print(f"   • Con valor adicional: {con_adicional} ({con_adicional/total*100:.2f}%)")
    print(f"   • Sin valor adicional: {sin_adicional} ({sin_adicional/total*100:.2f}%)")

def visualizar_datos(df):
    """
    Crea visualizaciones de los datos
    """
    print("\n" + "="*80)
    print("📈 GENERANDO VISUALIZACIONES")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Distribución de estados
    estado_counts = df['estado_proyecto'].value_counts()
    axes[0, 0].barh(estado_counts.index, estado_counts.values)
    axes[0, 0].set_title('Distribución de Estados del Proyecto', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Cantidad')
    
    # 2. Top 10 sectores
    sector_counts = df['sector'].value_counts().head(10)
    axes[0, 1].barh(sector_counts.index, sector_counts.values, color='green')
    axes[0, 1].set_title('Top 10 Sectores', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Cantidad')
    
    # 3. Distribución de valores totales (asegurar numérico)
    if 'valor_total_proyecto' in df.columns:
        valores_totales = pd.to_numeric(df['valor_total_proyecto'].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True), errors='coerce').dropna()
    else:
        valores_totales = pd.Series([], dtype=float)
    axes[1, 0].hist(valores_totales, bins=50, edgecolor='black')
    axes[1, 0].set_title('Distribución de Valores Totales', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Valor Total')
    axes[1, 0].set_ylabel('Frecuencia')
    
    # 4. Proyectos con/sin valor adicional (asegurar numérico)
    if 'valor_adicional' in df.columns:
        adicional_series = pd.to_numeric(
            df['valor_adicional'].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True),
            errors='coerce'
        ).fillna(0)
    else:
        adicional_series = pd.Series([0]*len(df))
    adicional_data = ['Con adicional' if x > 0 else 'Sin adicional' for x in adicional_series]
    adicional_counts = pd.Series(adicional_data).value_counts()
    axes[1, 1].pie(adicional_counts.values, labels=adicional_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Proyectos con/sin Valor Adicional', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('exploracion_datos.png', dpi=300, bbox_inches='tight')
    print("✅ Gráficos guardados en: exploracion_datos.png")
    plt.show()

def verificar_calidad_datos(df):
    """
    Verifica la calidad de los datos para machine learning
    """
    print("\n" + "="*80)
    print("🔍 VERIFICACIÓN DE CALIDAD PARA ML")
    print("="*80)
    
    # Normalizar estados (quitar espacios y convertir a mayúsculas)
    df_clean = df.copy()
    df_clean['estado_proyecto'] = df_clean['estado_proyecto'].str.strip().str.upper()
    
    print("\n✅ ESTADOS NORMALIZADOS:")
    print(df_clean['estado_proyecto'].value_counts())
    
    # Verificar balance de clases
    estado_counts = df_clean['estado_proyecto'].value_counts()
    clase_mayoritaria = estado_counts.iloc[0]
    total = len(df_clean)
    
    print(f"\n📊 BALANCE DE CLASES:")
    print(f"   • Clase mayoritaria: {estado_counts.index[0]} ({clase_mayoritaria/total*100:.2f}%)")
    print(f"   • Clase minoritaria: {estado_counts.index[-1]} ({estado_counts.iloc[-1]/total*100:.2f}%)")
    
    if clase_mayoritaria/total > 0.7:
        print("   ⚠️ ADVERTENCIA: Dataset desbalanceado. Considerar técnicas de balanceo.")
    else:
        print("   ✅ Dataset relativamente balanceado")
    
    # Guardar dataset limpio
    df_clean.to_csv('datos_limpios.csv', index=False)
    print("\n💾 Dataset limpio guardado en: datos_limpios.csv")
    
    return df_clean

def main():
    
    # Función principal
    
    print("\n" + "="*80)
    print("🚀 INICIO DE EXPLORACIÓN DE DATOS")
    print("="*80)
    
    # 1. Conectar a MySQL
    connection = conectar_mysql()
    if connection is None:
        return
    
    # 2. Cargar datos
    df = cargar_datos(connection)
    if df is None:
        return
    
    # 3. Explorar datos
    explorar_datos(df)
    
    # 4. Visualizar datos
    visualizar_datos(df)
    
    # 5. Verificar calidad
    df_clean = verificar_calidad_datos(df)
    
    # 6. Cerrar conexión
    connection.close()
    print("\n✅ Conexión cerrada")
    
    print("\n" + "="*80)
    print("✅ EXPLORACIÓN COMPLETADA")
    print("="*80)
    print("\n📁 Archivos generados:")
    print("   • exploracion_datos.png")
    print("   • datos_limpios.csv")

if __name__ == "__main__":
    main()