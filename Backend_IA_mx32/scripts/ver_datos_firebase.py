import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import csv
import pandas as pd
from datetime import datetime

# Configurar el project_id
PROJECT_ID = "mx32-76c52"

print(f"🔍 Conectando a Firebase para ver datos del proyecto: {PROJECT_ID}")

# Intentar usar credenciales de gcloud primero
try:
    # Usar credenciales por defecto de gcloud
    firebase_admin.initialize_app()
    print("✅ Firebase Admin SDK inicializado con credenciales de gcloud.")
except Exception as gcloud_error:
    print(f"⚠️  Error con credenciales de gcloud: {gcloud_error}")
    exit(1)

try:
    # Crear cliente de Firestore
    db = firestore.client()
    print("✅ Cliente de Firestore creado correctamente.")
    
    # Función para formatear timestamps
    def format_timestamp(timestamp):
        if hasattr(timestamp, 'timestamp'):
            return datetime.fromtimestamp(timestamp.timestamp()).strftime('%Y-%m-%d %H:%M:%S')
        return str(timestamp)
    
    # Función para formatear documentos
    def format_document(doc):
        data = doc.to_dict()
        formatted_data = {}
        for key, value in data.items():
            if hasattr(value, 'timestamp'):
                formatted_data[key] = format_timestamp(value)
            else:
                formatted_data[key] = value
        return formatted_data
    
    print("\n" + "="*60)
    print("📊 DATOS EN FIRESTORE")
    print("="*60)
    
    # Obtener todas las colecciones
    collections = db.collections()
    collection_count = 0
    all_data = []  # Lista para almacenar todos los datos
    collection_names = []  # Lista para almacenar nombres de colecciones
    
    for collection in collections:
        collection_count += 1
        collection_name = collection.id
        collection_names.append(collection_name)
        print(f"\n📁 Colección: {collection_name}")
        print("-" * 40)
        
        # Obtener todos los documentos de la colección
        docs = collection.stream()
        doc_count = 0
        
        for doc in docs:
            doc_count += 1
            print(f"  📄 Documento ID: {doc.id}")
            formatted_data = format_document(doc)
            
            # Agregar metadatos para el CSV
            row_data = {
                'coleccion': collection_name,
                'documento_id': doc.id
            }
            row_data.update(formatted_data)
            all_data.append(row_data)
            
            # Mostrar los datos del documento
            for key, value in formatted_data.items():
                print(f"    {key}: {value}")
            print()
        
        if doc_count == 0:
            print("  (No hay documentos en esta colección)")
        else:
            print(f"  Total documentos: {doc_count}")
    
    if collection_count == 0:
        print("❌ No se encontraron colecciones en la base de datos")
    else:
        print(f"\n✅ Total de colecciones encontradas: {collection_count}")
        print(f"📋 Nombres de colecciones: {', '.join(collection_names)}")
    
    # Exportar a CSV si hay datos
    if all_data:
        print("\n" + "="*60)
        print("💾 EXPORTANDO DATOS A CSV")
        print("="*60)
        
        # Crear DataFrame
        df = pd.DataFrame(all_data)
        
        # Generar nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"firebase_data_{timestamp}.csv"
        excel_filename = f"firebase_data_{timestamp}.xlsx"
        
        # Exportar a CSV
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"✅ Datos exportados a CSV: {csv_filename}")
        
        # Exportar a Excel
        try:
            df.to_excel(excel_filename, index=False, engine='openpyxl')
            print(f"✅ Datos exportados a Excel: {excel_filename}")
        except ImportError:
            print("⚠️  Para exportar a Excel, instala openpyxl: pip install openpyxl")
        
        # Mostrar resumen de datos
        print(f"\n📊 Resumen de exportación:")
        print(f"   - Total de registros: {len(all_data)}")
        print(f"   - Total de colecciones: {len(collection_names)}")
        print(f"   - Columnas: {list(df.columns)}")
        
        # Mostrar primeras filas
        print(f"\n🔍 Primeras 3 filas de datos:")
        print(df.head(3).to_string(index=False))
    
    print("\n" + "="*60)
    print("🎯 Exportación completada exitosamente")
    print("="*60)
    
except Exception as e:
    print(f"❌ Error al conectar con Firestore: {e}")
    exit(1)
