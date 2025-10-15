"""
Script de integración para conectar el sistema RAG con el backend existente de MX32
Modifica el backend existente para incluir capacidades RAG
"""

import os
import sys
import shutil
from pathlib import Path

def integrar_rag_con_backend():
    """
    Integra el sistema RAG con el backend existente de MX32
    """
    
    # Rutas
    backend_path = Path("/home/elias/Documentos/mx32-backend")
    rag_path = Path("/home/elias/Documentos/Backend_IA_mx32")
    
    print("🔧 Iniciando integración RAG con backend existente...")
    
    # 1. Verificar que el backend existe
    if not backend_path.exists():
        print(f"❌ Backend no encontrado en: {backend_path}")
        return False
    
    # 2. Copiar archivos RAG al backend
    archivos_rag = [
        "rag_agent.py",
        "rag_service.py", 
        "rag_endpoints.py"
    ]
    
    print("📁 Copiando archivos RAG al backend...")
    for archivo in archivos_rag:
        origen = rag_path / archivo
        destino = backend_path / "app" / "rag" / archivo
        
        if origen.exists():
            # Crear directorio si no existe
            destino.parent.mkdir(parents=True, exist_ok=True)
            
            # Copiar archivo
            shutil.copy2(origen, destino)
            print(f"✅ Copiado: {archivo}")
        else:
            print(f"⚠️ Archivo no encontrado: {archivo}")
    
    # 3. Crear __init__.py para el módulo rag
    init_file = backend_path / "app" / "rag" / "__init__.py"
    init_file.write_text('"""Módulo RAG para MX32 Backend"""\n')
    print("✅ Creado: app/rag/__init__.py")
    
    # 4. Modificar routes.py para incluir endpoints RAG
    routes_file = backend_path / "app" / "api" / "routes.py"
    if routes_file.exists():
        print("📝 Modificando routes.py para incluir endpoints RAG...")
        
        # Leer contenido actual
        with open(routes_file, 'r', encoding='utf-8') as f:
            contenido = f.read()
        
        # Agregar import y router RAG si no existe
        if "from .rag import rag_router" not in contenido:
            # Agregar import después de los imports existentes
            lines = contenido.split('\n')
            import_index = -1
            for i, line in enumerate(lines):
                if line.startswith('from .') or line.startswith('import '):
                    import_index = i
            
            if import_index >= 0:
                lines.insert(import_index + 1, "from .rag import rag_router")
                lines.insert(import_index + 2, "")
            
            # Agregar router RAG después de los routers existentes
            if "api_router.include_router(rag_router" not in contenido:
                lines.append("api_router.include_router(rag_router, tags=[\"rag\"], prefix=\"/rag\")")
            
            # Escribir archivo modificado
            with open(routes_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            print("✅ Modificado: app/api/routes.py")
        else:
            print("ℹ️ routes.py ya contiene endpoints RAG")
    
    # 5. Crear archivo de configuración RAG
    config_rag_file = backend_path / "app" / "rag" / "config.py"
    config_content = '''"""
Configuración para el sistema RAG de MX32
"""

import os
from typing import Optional

class RAGConfig:
    """Configuración del sistema RAG"""
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    
    # Firebase
    FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "mx32-76c52")
    
    # Cache
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "True").lower() == "true"
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def is_openai_configured(cls) -> bool:
        """Verifica si OpenAI está configurado"""
        return cls.OPENAI_API_KEY is not None and cls.OPENAI_API_KEY != ""
    
    @classmethod
    def get_openai_config(cls) -> dict:
        """Obtiene configuración de OpenAI"""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.OPENAI_MODEL,
            "max_tokens": cls.OPENAI_MAX_TOKENS,
            "temperature": cls.OPENAI_TEMPERATURE
        }

# Instancia global de configuración
rag_config = RAGConfig()
'''
    
    with open(config_rag_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    print("✅ Creado: app/rag/config.py")
    
    # 6. Actualizar requirements.txt
    requirements_file = backend_path / "requirements.txt"
    rag_requirements_file = rag_path / "requirements_rag.txt"
    
    if requirements_file.exists() and rag_requirements_file.exists():
        print("📦 Actualizando requirements.txt...")
        
        # Leer requirements existentes
        with open(requirements_file, 'r', encoding='utf-8') as f:
            existing_reqs = f.read()
        
        # Leer requirements RAG
        with open(rag_requirements_file, 'r', encoding='utf-8') as f:
            rag_reqs = f.read()
        
        # Agregar dependencias RAG si no existen
        new_deps = []
        for line in rag_reqs.split('\n'):
            if line.strip() and not line.startswith('#') and line not in existing_reqs:
                new_deps.append(line)
        
        if new_deps:
            with open(requirements_file, 'a', encoding='utf-8') as f:
                f.write('\n\n# Dependencias RAG\n')
                f.write('\n'.join(new_deps))
            print(f"✅ Agregadas {len(new_deps)} dependencias RAG")
        else:
            print("ℹ️ No se agregaron nuevas dependencias")
    
    # 7. Crear archivo de ejemplo de variables de entorno
    env_example_file = backend_path / ".env.rag.example"
    env_content = '''# Configuración RAG para MX32 Backend
# Agregar estas variables a tu archivo .env existente

# OpenAI API Key (requerido para funcionalidad RAG)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Configuración de OpenAI
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_TOKENS=1000
OPENAI_TEMPERATURE=0.7

# Configuración de cache
CACHE_TTL=3600
ENABLE_CACHE=True
'''
    
    with open(env_example_file, 'w', encoding='utf-8') as f:
        f.write(env_content)
    print("✅ Creado: .env.rag.example")
    
    # 8. Crear script de prueba
    test_script_file = backend_path / "test_rag_integration.py"
    test_content = '''"""
Script de prueba para la integración RAG
"""

import asyncio
import sys
import os

# Agregar el directorio del backend al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.rag.rag_service import rag_service

async def test_rag_integration():
    """Probar la integración RAG"""
    print("🧪 Probando integración RAG...")
    
    try:
        # Probar obtención de estados
        estados = await rag_service.obtener_estados_con_rag()
        print(f"✅ Estados obtenidos: {len(estados.get('estados', []))}")
        
        if estados.get('estados'):
            estado_ejemplo = estados['estados'][0]['nombre']
            print(f"🔍 Probando con estado: {estado_ejemplo}")
            
            # Probar consulta básica
            consulta = await rag_service.consulta_estado_con_rag(estado_ejemplo)
            print(f"✅ Consulta exitosa: {consulta.get('estado')}")
            
            # Probar consulta con IA
            pregunta = "¿Cuáles son las principales características de este estado?"
            respuesta_ia = await rag_service.consulta_ia_estado(estado_ejemplo, pregunta)
            print(f"✅ Respuesta IA generada: {len(respuesta_ia.get('respuesta', ''))} caracteres")
        
        print("🎉 Integración RAG funcionando correctamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_rag_integration())
    sys.exit(0 if success else 1)
'''
    
    with open(test_script_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    print("✅ Creado: test_rag_integration.py")
    
    print("\n🎉 Integración RAG completada!")
    print("\n📋 Próximos pasos:")
    print("1. Instalar dependencias: pip install -r requirements.txt")
    print("2. Configurar OPENAI_API_KEY en tu archivo .env")
    print("3. Ejecutar: python test_rag_integration.py")
    print("4. Iniciar el backend: uvicorn app.main:app --reload")
    print("5. Probar endpoints RAG en: http://localhost:8000/api/rag/")
    
    return True

if __name__ == "__main__":
    success = integrar_rag_con_backend()
    if success:
        print("\n✅ Integración completada exitosamente!")
    else:
        print("\n❌ Error en la integración")
        sys.exit(1)

