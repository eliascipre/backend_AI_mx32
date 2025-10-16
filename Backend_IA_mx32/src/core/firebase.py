import firebase_admin
from firebase_admin import credentials, firestore
import os

# Configurar el project_id
PROJECT_ID = "mx32-76c52"

print(f"Iniciando conexión a Firebase con proyecto: {PROJECT_ID}")

# Intentar usar credenciales de gcloud primero
try:
    # Usar credenciales por defecto de gcloud
    firebase_admin.initialize_app()
    print("✅ Firebase Admin SDK inicializado con credenciales de gcloud.")
except Exception as gcloud_error:
    print(f"⚠️  Error con credenciales de gcloud: {gcloud_error}")
    
    # Fallback al archivo serviceAccountKey.json si existe
    if os.path.exists("serviceAccountKey.json"):
        try:
            # Usar credenciales del service account
            cred = credentials.Certificate("serviceAccountKey.json")
            firebase_admin.initialize_app(cred, {'projectId': PROJECT_ID})
            print("✅ Firebase Admin SDK inicializado con service account.")
        except Exception as json_error:
            print(f"❌ Error con service account: {json_error}")
            print("💡 Soluciones:")
            print("   1. Ejecuta: gcloud auth application-default login")
            print("   2. O descarga un serviceAccountKey.json válido desde Firebase Console")
            exit(1)
    else:
        print("❌ No se encontró serviceAccountKey.json")
        print("💡 Soluciones:")
        print("   1. Ejecuta: gcloud auth application-default login")
        print("   2. O descarga un serviceAccountKey.json válido desde Firebase Console")
        exit(1)

try:
    # Crear cliente de Firestore
    db = firestore.client()
    print("✅ Cliente de Firestore creado correctamente.")
    
    # Probar la conexión escribiendo un documento de prueba
    doc_ref = db.collection('pruebas').document('test-python')
    doc_ref.set({
        'mensaje': 'Conexión desde Python exitosa',
        'timestamp': firestore.SERVER_TIMESTAMP,
        'proyecto': PROJECT_ID
    })
    print("¡Éxito! Se ha escrito un documento en Firestore en la colección 'pruebas'.")
    print("Verifica tu consola de Firebase.")
except Exception as e:
    print(f"❌ Error al conectar con Firestore: {e}")
    print("💡 Posibles soluciones:")
    print("   1. Verifica que el archivo serviceAccountKey.json sea válido")
    print("   2. Asegúrate de que la cuenta de servicio tenga permisos de Firestore")
    print("   3. Verifica que Firestore esté habilitado en tu proyecto")
    exit(1)