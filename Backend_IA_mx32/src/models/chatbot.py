"""
Modelos de datos para el chatbot MX32 con integración RAG y LangChain
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    """Roles de usuario en el sistema"""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"

class MessageRole(str, Enum):
    """Roles de mensajes en la conversación"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatMessage(BaseModel):
    """Modelo para un mensaje individual en la conversación"""
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

class UserContext(BaseModel):
    """Contexto del usuario para el chatbot"""
    user_id: str
    user_type: UserRole = UserRole.VIEWER
    current_page: Optional[str] = None
    current_state: Optional[str] = None
    current_parameter: Optional[str] = None
    session_id: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    last_activity: Optional[datetime] = Field(default_factory=datetime.now)
    rag_enabled: bool = Field(default=True, description="Si RAG está habilitado")

class ChatRequest(BaseModel):
    """Solicitud de chat desde el frontend"""
    message: str = Field(..., description="Mensaje del usuario")
    user_id: str = Field(..., description="ID del usuario")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Contexto adicional")
    session_id: Optional[str] = Field(default=None, description="ID de sesión")
    use_rag: bool = Field(default=True, description="Usar RAG para la respuesta")

class DeepChatMessage(BaseModel):
    """Mensaje específico para Deep Chat"""
    role: str = Field(..., description="Rol del mensaje")
    text: str = Field(..., description="Contenido del mensaje")

class DeepChatRequest(BaseModel):
    """Solicitud específica para Deep Chat"""
    messages: List[DeepChatMessage] = Field(..., description="Lista de mensajes")
    user_id: str = Field(..., description="ID del usuario")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Contexto adicional")
    use_rag: bool = Field(default=True, description="Usar RAG para la respuesta")

class ChatResponse(BaseModel):
    """Respuesta del chatbot"""
    response: str = Field(..., description="Respuesta del asistente")
    structured_data: Optional[Dict[str, Any]] = Field(default=None, description="Datos estructurados")
    rag_data: Optional[Dict[str, Any]] = Field(default=None, description="Datos específicos de RAG")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Nivel de confianza")
    sources: Optional[List[str]] = Field(default=None, description="Fuentes de información")
    suggested_actions: Optional[List[str]] = Field(default=None, description="Acciones sugeridas")
    follow_up_questions: Optional[List[str]] = Field(default=None, description="Preguntas de seguimiento")
    model_used: Optional[str] = Field(default=None, description="Modelo LLM utilizado")
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = Field(default=None, description="ID de sesión")
    memory_used: bool = Field(default=False, description="Si se usó memoria de conversación")
    entities_extracted: Optional[List[str]] = Field(default=None, description="Entidades extraídas")
    rag_enabled: bool = Field(default=True, description="Si RAG fue utilizado")

class ConversationHistory(BaseModel):
    """Historial de conversación"""
    session_id: str
    messages: List[ChatMessage]
    created_at: datetime
    last_activity: datetime
    user_context: UserContext

class AnalysisRequest(BaseModel):
    """Solicitud de análisis de datos"""
    data_type: str = Field(..., description="Tipo de datos a analizar")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Filtros aplicados")
    analysis_type: str = Field(default="summary", description="Tipo de análisis")
    user_id: str = Field(..., description="ID del usuario")

class AnalysisResponse(BaseModel):
    """Respuesta de análisis de datos"""
    analysis: str = Field(..., description="Análisis generado")
    insights: List[str] = Field(default=[], description="Insights principales")
    recommendations: List[str] = Field(default=[], description="Recomendaciones")
    data_summary: Optional[Dict[str, Any]] = Field(default=None, description="Resumen de datos")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Nivel de confianza")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatResponseStructured(BaseModel):
    """Respuesta de chat estructurada con todos los componentes"""
    response: str = Field(description="Respuesta principal del asistente")
    structured_data: Optional[Dict[str, Any]] = Field(default=None, description="Datos estructurados del análisis")
    comparison_data: Optional[Dict[str, Any]] = Field(default=None, description="Datos de comparación si aplica")
    trend_data: Optional[Dict[str, Any]] = Field(default=None, description="Datos de tendencias si aplica")
    entity_data: Optional[Dict[str, Any]] = Field(default=None, description="Datos de entidades extraídas")
    rag_data: Optional[Dict[str, Any]] = Field(default=None, description="Datos específicos de RAG")
    function_calls: List[Dict[str, Any]] = Field(description="Llamadas a funciones realizadas", default=[])
    suggested_actions: List[str] = Field(description="Acciones sugeridas", default=[])
    follow_up_questions: List[str] = Field(description="Preguntas de seguimiento", default=[])
    confidence: float = Field(description="Nivel de confianza general", ge=0.0, le=1.0)
    sources: List[str] = Field(description="Fuentes de información utilizadas", default=[])
    model_used: str = Field(description="Modelo LLM utilizado")
    session_id: str = Field(description="ID de sesión")
    timestamp: datetime = Field(default_factory=datetime.now)
    memory_used: bool = Field(description="Si se utilizó memoria de conversación", default=False)
    rag_enabled: bool = Field(description="Si RAG fue utilizado", default=False)

