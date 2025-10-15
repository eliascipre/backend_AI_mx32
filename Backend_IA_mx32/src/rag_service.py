"""
Servicio de integración RAG para MX32 Backend
Conecta el RAG Agent con los endpoints existentes de FastAPI
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
from rag_agent import rag_agent, consultar_estado_rag, obtener_estados_disponibles_rag

class RAGService:
    """Servicio que integra RAG con los endpoints existentes"""
    
    def __init__(self):
        self.rag_agent = rag_agent
    
    async def consulta_estado_con_rag(self, estado_nombre: str, parametros: List[str] = None, pregunta: str = None) -> Dict[str, Any]:
        """
        Realiza una consulta de estado con capacidades RAG
        Combina los datos estructurados con respuestas generadas por IA
        """
        try:
            # 1. Obtener datos completos del estado usando RAG
            estado_data = await self.rag_agent.obtener_datos_estado_completo(estado_nombre)
            if not estado_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"No se encontraron datos para el estado {estado_nombre}"
                )
            
            # 2. Si se proporciona una pregunta, generar respuesta RAG
            respuesta_rag = None
            if pregunta:
                respuesta_rag = await self.rag_agent.generar_respuesta_rag(pregunta, estado_nombre)
            
            # 3. Filtrar parámetros si se especifican
            parametros_filtrados = {}
            if parametros:
                for param in parametros:
                    if param in estado_data.parametros:
                        parametros_filtrados[param] = estado_data.parametros[param]
            else:
                parametros_filtrados = estado_data.parametros
            
            # 4. Estructurar respuesta en formato compatible con el frontend
            datos_por_parametro = {}
            for param_name, param_data in parametros_filtrados.items():
                datos_por_parametro[param_name] = {
                    "texto_analisis": param_data.texto_analisis,
                    "datos_apis": param_data.datos_apis
                }
            
            # 5. Respuesta final
            respuesta = {
                "estado": estado_nombre,
                "datos_por_parametro": datos_por_parametro,
                "link_pdf_estado": None,  # Se puede agregar después si está disponible
                "rag_habilitado": True,
                "timestamp": estado_data.id  # Usar ID como timestamp único
            }
            
            # 6. Agregar respuesta RAG si existe
            if respuesta_rag and not respuesta_rag.get('error'):
                respuesta["respuesta_ia"] = {
                    "pregunta": pregunta,
                    "respuesta": respuesta_rag.get('respuesta'),
                    "parametros_utilizados": respuesta_rag.get('parametros_disponibles', []),
                    "datos_utilizados": respuesta_rag.get('datos_utilizados', {})
                }
            
            return respuesta
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Error en consulta RAG: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error interno del servidor: {str(e)}"
            )
    
    async def obtener_estados_con_rag(self) -> Dict[str, Any]:
        """Obtiene lista de estados con información adicional del RAG"""
        try:
            estados = await obtener_estados_disponibles_rag()
            
            # Agregar información adicional de cada estado
            estados_info = []
            for estado in estados[:10]:  # Limitar a 10 para evitar sobrecarga
                try:
                    resumen = await self.rag_agent.obtener_resumen_estado(estado)
                    if not resumen.get('error'):
                        estados_info.append({
                            "nombre": estado,
                            "parametros_disponibles": resumen.get('parametros_disponibles', []),
                            "parametros_con_datos": resumen.get('parametros_con_datos_apis', 0),
                            "total_parametros": resumen.get('total_parametros', 0),
                            "rag_habilitado": True
                        })
                    else:
                        estados_info.append({
                            "nombre": estado,
                            "rag_habilitado": False,
                            "error": resumen.get('error')
                        })
                except Exception as e:
                    print(f"⚠️ Error obteniendo info de {estado}: {e}")
                    estados_info.append({
                        "nombre": estado,
                        "rag_habilitado": False,
                        "error": str(e)
                    })
            
            return {
                "estados": estados_info,
                "total_estados": len(estados),
                "rag_habilitado": True
            }
            
        except Exception as e:
            print(f"❌ Error obteniendo estados con RAG: {e}")
            return {
                "estados": [],
                "total_estados": 0,
                "rag_habilitado": False,
                "error": str(e)
            }
    
    async def consulta_ia_estado(self, estado_nombre: str, pregunta: str) -> Dict[str, Any]:
        """Endpoint específico para consultas de IA sobre un estado"""
        try:
            respuesta_rag = await consultar_estado_rag(estado_nombre, pregunta)
            
            if respuesta_rag.get('error'):
                raise HTTPException(
                    status_code=400,
                    detail=respuesta_rag.get('error')
                )
            
            return {
                "estado": estado_nombre,
                "pregunta": pregunta,
                "respuesta": respuesta_rag.get('respuesta'),
                "parametros_disponibles": respuesta_rag.get('parametros_disponibles', []),
                "datos_utilizados": respuesta_rag.get('datos_utilizados', {}),
                "timestamp": respuesta_rag.get('timestamp'),
                "rag_habilitado": True
            }
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Error en consulta IA: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error procesando consulta de IA: {str(e)}"
            )
    
    async def obtener_resumen_estado_rag(self, estado_nombre: str) -> Dict[str, Any]:
        """Obtiene un resumen ejecutivo de un estado usando RAG"""
        try:
            resumen = await self.rag_agent.obtener_resumen_estado(estado_nombre)
            
            if resumen.get('error'):
                raise HTTPException(
                    status_code=404,
                    detail=resumen.get('error')
                )
            
            return {
                "estado": estado_nombre,
                "resumen": resumen,
                "rag_habilitado": True
            }
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Error obteniendo resumen RAG: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error obteniendo resumen: {str(e)}"
            )

# Instancia global del servicio
rag_service = RAGService()

# Funciones de utilidad para integración directa
async def consulta_estado_rag_service(estado_nombre: str, parametros: List[str] = None, pregunta: str = None) -> Dict[str, Any]:
    """Función de utilidad para consultas de estado con RAG"""
    return await rag_service.consulta_estado_con_rag(estado_nombre, parametros, pregunta)

async def consulta_ia_estado_service(estado_nombre: str, pregunta: str) -> Dict[str, Any]:
    """Función de utilidad para consultas de IA"""
    return await rag_service.consulta_ia_estado(estado_nombre, pregunta)

async def obtener_estados_rag_service() -> Dict[str, Any]:
    """Función de utilidad para obtener estados con RAG"""
    return await rag_service.obtener_estados_con_rag()

if __name__ == "__main__":
    # Ejemplo de uso del servicio
    async def ejemplo_servicio():
        print("🔧 Iniciando ejemplo de uso del RAG Service...")
        
        # Obtener estados
        estados = await obtener_estados_rag_service()
        print(f"📋 Estados con RAG: {json.dumps(estados, indent=2, ensure_ascii=False)}")
        
        if estados.get('estados'):
            estado_ejemplo = estados['estados'][0]['nombre']
            print(f"\n🔍 Probando consulta con: {estado_ejemplo}")
            
            # Consulta normal con RAG
            consulta = await consulta_estado_rag_service(estado_ejemplo, ["Situación de Seguridad"])
            print(f"📊 Consulta normal: {json.dumps(consulta, indent=2, ensure_ascii=False)}")
            
            # Consulta de IA
            pregunta = "¿Cuál es la situación de seguridad en este estado?"
            respuesta_ia = await consulta_ia_estado_service(estado_ejemplo, pregunta)
            print(f"🤖 Respuesta IA: {json.dumps(respuesta_ia, indent=2, ensure_ascii=False)}")
    
    # Ejecutar ejemplo
    asyncio.run(ejemplo_servicio())

