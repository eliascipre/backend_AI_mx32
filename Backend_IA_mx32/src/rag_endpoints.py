"""
Endpoints RAG para MX32 Backend
Nuevos endpoints que integran capacidades de IA con los datos existentes
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from src.rag_service import rag_service

# Router para endpoints RAG
rag_router = APIRouter()

# Modelos Pydantic para las peticiones
class ConsultaRAGRequest(BaseModel):
    estado: str
    parametros: Optional[List[str]] = None
    pregunta: Optional[str] = None

class ConsultaIARequest(BaseModel):
    estado: str
    pregunta: str

class ConsultaParametrosRAGRequest(BaseModel):
    estado: str
    parametros: List[str]
    pregunta: Optional[str] = None

# Endpoints RAG

@rag_router.post("/consulta-estado-rag")
async def consulta_estado_rag_endpoint(consulta: ConsultaRAGRequest):
    """
    Endpoint principal para consultas de estado con capacidades RAG.
    Combina los datos estructurados existentes con respuestas generadas por IA.
    
    - **estado**: Nombre del estado a consultar
    - **parametros**: Lista opcional de parámetros específicos a consultar
    - **pregunta**: Pregunta opcional para generar respuesta con IA
    """
    try:
        resultado = await rag_service.consulta_estado_con_rag(
            estado_nombre=consulta.estado,
            parametros=consulta.parametros,
            pregunta=consulta.pregunta
        )
        return resultado
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}"
        )

@rag_router.post("/consulta-ia")
async def consulta_ia_endpoint(consulta: ConsultaIARequest):
    """
    Endpoint específico para consultas de IA sobre un estado.
    Genera respuestas contextuales basadas en los datos del estado.
    
    - **estado**: Nombre del estado a consultar
    - **pregunta**: Pregunta específica sobre el estado
    """
    try:
        resultado = await rag_service.consulta_ia_estado(
            estado_nombre=consulta.estado,
            pregunta=consulta.pregunta
        )
        return resultado
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando consulta de IA: {str(e)}"
        )

@rag_router.get("/estados-rag")
async def obtener_estados_rag():
    """
    Obtiene lista de estados con información adicional del RAG.
    Incluye metadatos sobre parámetros disponibles y capacidades de IA.
    """
    try:
        resultado = await rag_service.obtener_estados_con_rag()
        return resultado
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo estados con RAG: {str(e)}"
        )

@rag_router.get("/estado/{estado_nombre}/resumen")
async def obtener_resumen_estado(estado_nombre: str):
    """
    Obtiene un resumen ejecutivo de un estado usando RAG.
    
    - **estado_nombre**: Nombre del estado a resumir
    """
    try:
        resultado = await rag_service.obtener_resumen_estado_rag(estado_nombre)
        return resultado
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo resumen: {str(e)}"
        )

@rag_router.post("/consulta-parametros-rag")
async def consulta_parametros_rag_endpoint(consulta: ConsultaParametrosRAGRequest):
    """
    Endpoint para consultas específicas de parámetros con RAG.
    Útil cuando se quiere consultar parámetros específicos con capacidades de IA.
    
    - **estado**: Nombre del estado
    - **parametros**: Lista de parámetros específicos a consultar
    - **pregunta**: Pregunta opcional para generar respuesta con IA
    """
    try:
        resultado = await rag_service.consulta_estado_con_rag(
            estado_nombre=consulta.estado,
            parametros=consulta.parametros,
            pregunta=consulta.pregunta
        )
        return resultado
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en consulta de parámetros RAG: {str(e)}"
        )

@rag_router.get("/estado/{estado_nombre}/parametros")
async def obtener_parametros_estado(estado_nombre: str):
    """
    Obtiene información detallada sobre los parámetros disponibles para un estado.
    
    - **estado_nombre**: Nombre del estado
    """
    try:
        # Obtener datos del estado
        estado_data = await rag_service.rag_agent.obtener_datos_estado_completo(estado_nombre)
        if not estado_data:
            raise HTTPException(
                status_code=404,
                detail=f"Estado {estado_nombre} no encontrado"
            )
        
        # Estructurar información de parámetros
        parametros_info = []
        for param_name, param_data in estado_data.parametros.items():
            parametros_info.append({
                "nombre": param_name,
                "id": param_data.id,
                "tiene_texto_analisis": bool(param_data.texto_analisis and param_data.texto_analisis != "No se encontró un texto de análisis para este parámetro."),
                "tiene_datos_apis": len(param_data.datos_apis) > 0,
                "cantidad_apis": len(param_data.datos_apis),
                "apis_disponibles": [api["nombre"] for api in param_data.datos_apis]
            })
        
        return {
            "estado": estado_nombre,
            "parametros": parametros_info,
            "total_parametros": len(parametros_info),
            "parametros_con_datos": len([p for p in parametros_info if p["tiene_datos_apis"]]),
            "rag_habilitado": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo parámetros del estado: {str(e)}"
        )

@rag_router.get("/health-rag")
async def health_check_rag():
    """
    Endpoint de salud para verificar el estado del sistema RAG.
    """
    try:
        # Verificar conexión a Firebase
        firebase_ok = rag_service.rag_agent.db is not None
        
        # Verificar OpenAI
        openai_ok = rag_service.rag_agent.openai_client is not None
        
        # Obtener conteo de estados
        estados = await obtener_estados_disponibles_rag()
        estados_count = len(estados)
        
        return {
            "rag_habilitado": True,
            "firebase_conectado": firebase_ok,
            "openai_disponible": openai_ok,
            "estados_disponibles": estados_count,
            "servicios": {
                "firebase": "✅ Conectado" if firebase_ok else "❌ Desconectado",
                "openai": "✅ Disponible" if openai_ok else "❌ No disponible",
                "rag_agent": "✅ Activo"
            }
        }
        
    except Exception as e:
        return {
            "rag_habilitado": False,
            "error": str(e),
            "servicios": {
                "firebase": "❌ Error",
                "openai": "❌ Error",
                "rag_agent": "❌ Error"
            }
        }

# Función auxiliar para obtener estados disponibles
async def obtener_estados_disponibles_rag():
    """Función auxiliar para obtener estados disponibles"""
    try:
        from rag_agent import obtener_estados_disponibles_rag as _obtener_estados
        return await _obtener_estados()
    except Exception as e:
        print(f"❌ Error obteniendo estados: {e}")
        return []

# Endpoints de compatibilidad con el sistema existente
@rag_router.post("/consulta-dinamica-rag")
async def consulta_dinamica_rag_endpoint(consulta: ConsultaRAGRequest):
    """
    Endpoint de compatibilidad que extiende la funcionalidad de consulta-dinamica existente.
    Mantiene la misma interfaz pero agrega capacidades RAG.
    """
    try:
        # Si no se proporcionan parámetros, usar todos los disponibles
        if not consulta.parametros:
            # Obtener parámetros disponibles del estado
            estado_data = await rag_service.rag_agent.obtener_datos_estado_completo(consulta.estado)
            if estado_data:
                consulta.parametros = list(estado_data.parametros.keys())
        
        resultado = await rag_service.consulta_estado_con_rag(
            estado_nombre=consulta.estado,
            parametros=consulta.parametros,
            pregunta=consulta.pregunta
        )
        return resultado
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en consulta dinámica RAG: {str(e)}"
        )

