"""
Configuración central del sistema MX32
"""

import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Configuración del sistema MX32"""
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    # --- Variables de la Aplicación ---
    app_name: str = "mx32-rag-langchain"
    app_version: str = "2.0.0"
    app_env: str = "development"
    app_debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # --- Cerebras API ---
    cerebras_api_key: str = "csk-9356538ykt4yvhkfnccdcpft69869kd5njyvw3vf2y5x48f4"
    cerebras_model: str = "gpt-oss-120b"
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    
    # --- Firebase ---
    firebase_project_id: str = "tu-proyecto-firebase"
    firebase_credentials_path: str = "./src/core/serviceAccountKey.json"
    
    # --- CORS ---
    cors_origins: str = "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001"
    
    # --- LangChain ---
    langchain_temperature: float = 0.7
    langchain_max_tokens: int = 2000
    langchain_top_p: float = 1.0
    langchain_reasoning_effort: str = "high"
    
    # --- OpenAI (no usado, solo para compatibilidad) ---
    openai_api_key: str = "dummy"
    
    # --- Memoria ---
    conversation_memory_k: int = 10
    entity_memory_enabled: bool = True
    
    # --- RAG ---
    rag_enabled: bool = True
    rag_confidence_threshold: float = 0.7
    rag_max_results: int = 5
    
    # --- Logging ---
    log_level: str = "INFO"
    log_file: str = "rag_langchain.log"
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Convierte el string de CORS en una lista"""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(",")]
        return self.cors_origins

# Instancia global de configuración
settings = Settings()
