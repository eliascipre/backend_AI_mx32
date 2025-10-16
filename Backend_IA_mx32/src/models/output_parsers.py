"""
Modelos Pydantic para Output Parsers de LangChain
Integrados con el sistema RAG existente
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class AnalysisType(str, Enum):
    """Tipos de análisis disponibles"""
    SECURITY = "seguridad"
    ECONOMY = "economia"
    INFRASTRUCTURE = "infraestructura"
    DEMOGRAPHICS = "demografia"
    EDUCATION = "educacion"
    HEALTH = "salud"
    GENERAL = "general"
    RAG_ANALYSIS = "rag_analysis"

class ConfidenceLevel(str, Enum):
    """Niveles de confianza"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class StructuredAnalysisResponse(BaseModel):
    """Respuesta estructurada para análisis de datos"""
    analysis: str = Field(description="Análisis principal del estado/parámetro")
    insights: List[str] = Field(description="Insights clave identificados", min_items=1, max_items=5)
    recommendations: List[str] = Field(description="Recomendaciones específicas", min_items=1, max_items=5)
    key_metrics: Dict[str, Any] = Field(description="Métricas clave extraídas")
    confidence: ConfidenceLevel = Field(description="Nivel de confianza del análisis")
    analysis_type: AnalysisType = Field(description="Tipo de análisis realizado")
    state_context: Optional[str] = Field(description="Contexto específico del estado")
    timestamp: datetime = Field(default_factory=datetime.now)

class ComparisonResponse(BaseModel):
    """Respuesta estructurada para comparaciones entre estados"""
    states_compared: List[str] = Field(description="Estados comparados", min_items=2)
    comparison_metrics: Dict[str, Dict[str, Any]] = Field(description="Métricas de comparación")
    winner_state: str = Field(description="Estado con mejor desempeño")
    key_differences: List[str] = Field(description="Diferencias clave identificadas")
    recommendations: List[str] = Field(description="Recomendaciones basadas en la comparación")
    confidence: ConfidenceLevel = Field(description="Nivel de confianza de la comparación")

class TrendAnalysisResponse(BaseModel):
    """Respuesta estructurada para análisis de tendencias"""
    trend_direction: str = Field(description="Dirección de la tendencia (creciente, decreciente, estable)")
    trend_strength: str = Field(description="Fuerza de la tendencia (fuerte, moderada, débil)")
    key_drivers: List[str] = Field(description="Factores clave que impulsan la tendencia")
    future_predictions: List[str] = Field(description="Predicciones futuras basadas en la tendencia")
    risk_factors: List[str] = Field(description="Factores de riesgo identificados")
    opportunities: List[str] = Field(description="Oportunidades identificadas")
    confidence: ConfidenceLevel = Field(description="Nivel de confianza del análisis de tendencias")

class EntityExtractionResponse(BaseModel):
    """Respuesta estructurada para extracción de entidades"""
    states_mentioned: List[str] = Field(description="Estados mencionados en la consulta")
    parameters_mentioned: List[str] = Field(description="Parámetros mencionados en la consulta")
    metrics_mentioned: List[str] = Field(description="Métricas mencionadas en la consulta")
    time_periods: List[str] = Field(description="Períodos de tiempo mencionados")
    entities: Dict[str, List[str]] = Field(description="Entidades extraídas por categoría")
    intent: str = Field(description="Intención detectada en la consulta")

class FunctionCallResponse(BaseModel):
    """Respuesta estructurada para function calling"""
    function_name: str = Field(description="Nombre de la función llamada")
    parameters: Dict[str, Any] = Field(description="Parámetros pasados a la función")
    result: Any = Field(description="Resultado de la función")
    success: bool = Field(description="Si la función se ejecutó exitosamente")
    error_message: Optional[str] = Field(description="Mensaje de error si hubo fallo")
    execution_time: float = Field(description="Tiempo de ejecución en segundos")

class RAGAnalysisResponse(BaseModel):
    """Respuesta específica para análisis RAG"""
    estado: str = Field(description="Estado analizado")
    pregunta: str = Field(description="Pregunta original")
    respuesta: str = Field(description="Respuesta generada por RAG")
    parametros_utilizados: List[str] = Field(description="Parámetros utilizados en el análisis")
    datos_utilizados: Dict[str, Any] = Field(description="Datos específicos utilizados")
    confidence: ConfidenceLevel = Field(description="Nivel de confianza del análisis RAG")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatResponseStructured(BaseModel):
    """Respuesta de chat estructurada con todos los componentes"""
    response: str = Field(description="Respuesta principal del asistente")
    structured_data: Optional[StructuredAnalysisResponse] = Field(default=None, description="Datos estructurados del análisis")
    comparison_data: Optional[ComparisonResponse] = Field(default=None, description="Datos de comparación si aplica")
    trend_data: Optional[TrendAnalysisResponse] = Field(default=None, description="Datos de tendencias si aplica")
    entity_data: Optional[EntityExtractionResponse] = Field(default=None, description="Datos de entidades extraídas")
    rag_data: Optional[RAGAnalysisResponse] = Field(default=None, description="Datos específicos de RAG")
    function_calls: List[FunctionCallResponse] = Field(description="Llamadas a funciones realizadas", default=[])
    suggested_actions: List[str] = Field(description="Acciones sugeridas", default=[])
    follow_up_questions: List[str] = Field(description="Preguntas de seguimiento", default=[])
    confidence: float = Field(description="Nivel de confianza general", ge=0.0, le=1.0)
    sources: List[str] = Field(description="Fuentes de información utilizadas", default=[])
    model_used: str = Field(description="Modelo LLM utilizado")
    session_id: str = Field(description="ID de sesión")
    timestamp: datetime = Field(default_factory=datetime.now)
    memory_used: bool = Field(description="Si se utilizó memoria de conversación", default=False)
    rag_enabled: bool = Field(description="Si RAG fue utilizado", default=False)

