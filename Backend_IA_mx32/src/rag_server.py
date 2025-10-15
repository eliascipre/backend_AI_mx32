"""
Servidor FastAPI independiente para el sistema RAG de MX32
Proporciona endpoints RAG que se pueden integrar con el backend existente
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Importar endpoints RAG
from rag_endpoints import rag_router

# Configuración
HOST = os.getenv("RAG_HOST", "0.0.0.0")
PORT = int(os.getenv("RAG_PORT", "8001"))
DEBUG = os.getenv("RAG_DEBUG", "True").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

def create_rag_app() -> FastAPI:
    """Crear aplicación FastAPI para el sistema RAG"""
    
    app = FastAPI(
        title="MX32 RAG Agent API",
        description="Sistema de Retrieval-Augmented Generation para análisis de estados mexicanos",
        version="1.0.0",
        debug=DEBUG
    )
    
    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # En producción, especificar dominios específicos
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Endpoint raíz
    @app.get("/", tags=["root"], summary="Welcome to MX32 RAG API")
    def root():
        return {
            "message": "Bienvenido a MX32 RAG Agent API",
            "version": "1.0.0",
            "description": "Sistema de IA para análisis de estados mexicanos",
            "endpoints": {
                "consulta_estado_rag": "/api/rag/consulta-estado-rag",
                "consulta_ia": "/api/rag/consulta-ia",
                "estados_rag": "/api/rag/estados-rag",
                "health": "/api/rag/health-rag"
            }
        }
    
    # Incluir router RAG
    app.include_router(rag_router, prefix="/api/rag", tags=["rag"])
    
    # Endpoint de salud general
    @app.get("/health", tags=["health"])
    async def health_check():
        """Endpoint de salud general del servidor"""
        return {
            "status": "healthy",
            "service": "MX32 RAG Agent",
            "version": "1.0.0"
        }
    
    return app

# Crear aplicación
app = create_rag_app()

# Configurar logging
logger.remove()  # Remover handler por defecto
logger.add(
    "rag_agent.log",
    level=LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention="7 days"
)
logger.add(
    lambda msg: print(msg, end=""),  # También imprimir en consola
    level=LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

if __name__ == "__main__":
    logger.info(f"🚀 Iniciando MX32 RAG Agent en {HOST}:{PORT}")
    logger.info(f"🔧 Modo debug: {DEBUG}")
    logger.info(f"📊 Nivel de log: {LOG_LEVEL}")
    
    # Verificar configuración
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.warning("⚠️ OPENAI_API_KEY no configurada. Algunas funcionalidades no estarán disponibles.")
    else:
        logger.info("✅ OpenAI API Key configurada")
    
    # Iniciar servidor
    uvicorn.run(
        "rag_server:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )

