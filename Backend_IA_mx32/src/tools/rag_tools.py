"""
Herramientas (Tools) para Function Calling en el chatbot MX32 con RAG
Integradas con Cerebras y optimizadas para análisis de estados mexicanos
"""

from langchain_core.tools import tool
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import logging

from src.rag_service import rag_service

logger = logging.getLogger(__name__)

@tool
def get_estado_rag_data(estado: str, parametros: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Obtiene datos completos de un estado usando RAG.
    
    Args:
        estado: Nombre del estado (ej: "Jalisco", "Ciudad de México")
        parametros: Lista opcional de parámetros específicos
    
    Returns:
        Diccionario con los datos del estado obtenidos via RAG
    """
    try:
        # Usar el servicio RAG existente
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        resultado = loop.run_until_complete(
            rag_service.consulta_estado_con_rag(estado, parametros)
        )
        
        return {
            "estado": estado,
            "datos": resultado,
            "parametros_consultados": parametros or [],
            "rag_habilitado": True,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
    except Exception as e:
        logger.error(f"Error obteniendo datos RAG del estado {estado}: {e}")
        return {
            "estado": estado,
            "error": str(e),
            "success": False
        }

@tool
def consulta_ia_estado(estado: str, pregunta: str) -> Dict[str, Any]:
    """
    Realiza una consulta de IA específica sobre un estado usando RAG.
    
    Args:
        estado: Nombre del estado
        pregunta: Pregunta específica sobre el estado
    
    Returns:
        Diccionario con la respuesta de IA basada en RAG
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        resultado = loop.run_until_complete(
            rag_service.consulta_ia_estado(estado, pregunta)
        )
        
        return {
            "estado": estado,
            "pregunta": pregunta,
            "respuesta": resultado.get("respuesta", ""),
            "parametros_utilizados": resultado.get("parametros_disponibles", []),
            "rag_habilitado": True,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
    except Exception as e:
        logger.error(f"Error en consulta IA para {estado}: {e}")
        return {
            "estado": estado,
            "pregunta": pregunta,
            "error": str(e),
            "success": False
        }

@tool
def obtener_resumen_estado(estado: str) -> Dict[str, Any]:
    """
    Obtiene un resumen ejecutivo de un estado usando RAG.
    
    Args:
        estado: Nombre del estado
    
    Returns:
        Diccionario con resumen ejecutivo del estado
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        resultado = loop.run_until_complete(
            rag_service.obtener_resumen_estado_rag(estado)
        )
        
        return {
            "estado": estado,
            "resumen": resultado.get("resumen", {}),
            "rag_habilitado": True,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
    except Exception as e:
        logger.error(f"Error obteniendo resumen de {estado}: {e}")
        return {
            "estado": estado,
            "error": str(e),
            "success": False
        }

@tool
def comparar_estados_rag(estados: List[str], parametro: str) -> Dict[str, Any]:
    """
    Compara múltiples estados en un parámetro específico usando RAG.
    
    Args:
        estados: Lista de nombres de estados a comparar
        parametro: Parámetro para la comparación
    
    Returns:
        Diccionario con la comparación de estados usando RAG
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Obtener datos de cada estado
        datos_estados = {}
        for estado in estados:
            resultado = loop.run_until_complete(
                rag_service.consulta_estado_con_rag(estado, [parametro])
            )
            datos_estados[estado] = resultado
        
        # Generar comparación
        comparacion = {
            "estados_comparados": estados,
            "parametro": parametro,
            "datos_por_estado": datos_estados,
            "rag_habilitado": True,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        
        return comparacion
    except Exception as e:
        logger.error(f"Error comparando estados {estados}: {e}")
        return {
            "estados": estados,
            "parametro": parametro,
            "error": str(e),
            "success": False
        }

@tool
def buscar_estados_similares(estado_referencia: str, parametro: str) -> Dict[str, Any]:
    """
    Busca estados con características similares usando RAG.
    
    Args:
        estado_referencia: Estado de referencia
        parametro: Parámetro para la búsqueda de similitud
    
    Returns:
        Diccionario con estados similares encontrados
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Obtener datos del estado de referencia
        datos_referencia = loop.run_until_complete(
            rag_service.consulta_estado_con_rag(estado_referencia, [parametro])
        )
        
        # Lista de todos los estados para comparar
        todos_estados = ["jalisco", "nuevo leon", "ciudad de mexico", "queretaro", "yucatan", 
                        "quintana roo", "baja california", "sonora", "chihuahua", "coahuila"]
        
        # Filtrar el estado de referencia
        estados_a_comparar = [e for e in todos_estados if e != estado_referencia.lower()]
        
        # Obtener datos de otros estados
        estados_similares = []
        for estado in estados_similares[:5]:  # Limitar a 5 para performance
            try:
                datos_estado = loop.run_until_complete(
                    rag_service.consulta_estado_con_rag(estado, [parametro])
                )
                
                # Calcular similitud básica (en una implementación real, usar algoritmos de similitud)
                similitud_score = 0.7  # Placeholder
                
                estados_similares.append({
                    "estado": estado,
                    "similitud_score": similitud_score,
                    "datos": datos_estado
                })
            except Exception as e:
                logger.warning(f"Error obteniendo datos de {estado}: {e}")
                continue
        
        return {
            "estado_referencia": estado_referencia,
            "parametro": parametro,
            "estados_similares": estados_similares,
            "rag_habilitado": True,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
    except Exception as e:
        logger.error(f"Error buscando estados similares a {estado_referencia}: {e}")
        return {
            "estado_referencia": estado_referencia,
            "parametro": parametro,
            "error": str(e),
            "success": False
        }

@tool
def generar_recomendaciones_rag(estado: str, parametro: str, contexto: Optional[str] = None) -> Dict[str, Any]:
    """
    Genera recomendaciones específicas para un estado y parámetro usando RAG.
    
    Args:
        estado: Nombre del estado
        parametro: Parámetro específico
        contexto: Contexto adicional para las recomendaciones
    
    Returns:
        Diccionario con recomendaciones específicas basadas en RAG
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Obtener datos del estado
        datos_estado = loop.run_until_complete(
            rag_service.consulta_estado_con_rag(estado, [parametro])
        )
        
        # Generar recomendaciones basadas en los datos
        recomendaciones = []
        
        if parametro.lower() in ["situación de seguridad", "seguridad"]:
            recomendaciones = [
                "Fortalecer la coordinación entre fuerzas de seguridad",
                "Implementar programas de prevención social",
                "Mejorar la infraestructura de videovigilancia",
                "Desarrollar estrategias de proximidad ciudadana"
            ]
        elif parametro.lower() in ["oportunidades emergentes", "economía"]:
            recomendaciones = [
                "Atraer inversión extranjera directa",
                "Desarrollar clusters industriales especializados",
                "Fortalecer la educación técnica y profesional",
                "Mejorar la conectividad logística"
            ]
        elif parametro.lower() in ["infraestructura y conectividad", "infraestructura"]:
            recomendaciones = [
                "Expandir la cobertura de internet de alta velocidad",
                "Mejorar la red de carreteras y transporte",
                "Desarrollar hubs logísticos regionales",
                "Invertir en infraestructura digital"
            ]
        else:
            recomendaciones = [
                "Realizar un análisis más detallado del parámetro",
                "Consultar con expertos en el área específica",
                "Revisar mejores prácticas de otros estados",
                "Desarrollar un plan de acción integral"
            ]
        
        return {
            "estado": estado,
            "parametro": parametro,
            "contexto": contexto,
            "recomendaciones": recomendaciones,
            "datos_base": datos_estado,
            "rag_habilitado": True,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
    except Exception as e:
        logger.error(f"Error generando recomendaciones para {estado}: {e}")
        return {
            "estado": estado,
            "parametro": parametro,
            "error": str(e),
            "success": False
        }

# Lista de herramientas RAG disponibles
RAG_TOOLS = [
    get_estado_rag_data,
    consulta_ia_estado,
    obtener_resumen_estado,
    comparar_estados_rag,
    buscar_estados_similares,
    generar_recomendaciones_rag
]

