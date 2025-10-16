"""
Archivo principal para el servidor RAG + LangChain + Cerebras
"""

import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from dotenv import load_dotenv

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Cargar variables de entorno
load_dotenv()

# Importar configuración
from src.core.config import settings

# Importar endpoints
from src.rag_endpoints import rag_router
from src.api.deepchat_endpoints import router as deepchat_router

def create_app() -> FastAPI:
    """Crear aplicación FastAPI"""
    
    app = FastAPI(
        title="MX32 RAG + LangChain + Cerebras API",
        description="Sistema integrado de RAG, LangChain y Cerebras para análisis de estados mexicanos",
        version="2.0.0",
        debug=True
    )
    
    # Configurar CORS para el frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Endpoint raíz
    @app.get("/", tags=["root"])
    def root():
        return {
            "message": "MX32 RAG + LangChain + Cerebras API",
            "version": "2.0.0",
            "features": [
                "RAG (Retrieval-Augmented Generation)",
                "LangChain Expression Language (LCEL)",
                "Output Parsers estructurados",
                "Few-Shot Prompting",
                "Function Calling",
                "Conversation Memory",
                "Cerebras AI Integration"
            ],
            "endpoints": {
                "rag": "/api/rag/",
                "deepchat": "/api/deepchat/",
                "health": "/api/deepchat/health"
            },
            "frontend_compatible": True,
            "deepchat_ready": True
        }
    
    # Health check general
    @app.get("/health", tags=["health"])
    def health_check():
        return {
            "status": "healthy",
            "services": {
                "rag": "✅ Activo",
                "langchain": "✅ Activo", 
                "cerebras": "✅ Activo",
                "deepchat": "✅ Activo"
            },
            "timestamp": "2024-12-19T00:00:00Z"
        }
    
    # Incluir routers
    app.include_router(rag_router, prefix="/api/rag", tags=["rag"])
    app.include_router(deepchat_router, prefix="/api/deepchat", tags=["deepchat"])
    
    return app

# Crear aplicación
app = create_app()

if __name__ == "__main__":
    logger.info("🚀 Iniciando MX32 RAG + LangChain + Cerebras API")
    logger.info("🔗 Frontend compatible: mx32-frontend")
    logger.info("🤖 Motor de IA: Cerebras GPT-OSS-120B")
    logger.info("📊 RAG habilitado: Firebase + LangChain")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.app_debug,
        log_level=settings.log_level.lower()
    )
