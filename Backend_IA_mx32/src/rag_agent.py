"""
Agente RAG (Retrieval-Augmented Generation) para MX32
Integra con Firebase y proporciona respuestas contextuales sobre estados mexicanos
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore
from openai import AsyncOpenAI

# Configuración
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FIREBASE_PROJECT_ID = "mx32-76c52"

@dataclass
class EstadoData:
    """Estructura para datos de un estado"""
    id: str
    nombre: str
    state_id_replacement: str
    parametros: Dict[str, Any]

@dataclass
class ParametroData:
    """Estructura para datos de un parámetro"""
    id: str
    nombre: str
    texto_analisis: str
    datos_apis: List[Dict[str, Any]]

class RAGAgent:
    """Agente RAG que se alimenta de datos de Firebase"""
    
    def __init__(self):
        self.db = None
        self.openai_client = None
        self.estados_cache = {}
        self.parametros_cache = {}
        self._initialize_services()
    
    def _initialize_services(self):
        """Inicializa Firebase y OpenAI"""
        try:
            # Inicializar Firebase
            if not firebase_admin._apps:
                firebase_admin.initialize_app()
            self.db = firestore.client()
            print("✅ Firebase inicializado para RAG Agent")
            
            # Inicializar OpenAI
            if OPENAI_API_KEY:
                self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
                print("✅ OpenAI inicializado para RAG Agent")
            else:
                print("⚠️ OPENAI_API_KEY no configurada")
                
        except Exception as e:
            print(f"❌ Error inicializando servicios: {e}")
            raise
    
    async def obtener_datos_estado_completo(self, estado_nombre: str) -> Optional[EstadoData]:
        """
        Obtiene todos los datos de un estado desde Firebase
        Similar a la función del backend pero optimizada para RAG
        """
        try:
            print(f"🔍 RAG: Obteniendo datos completos para {estado_nombre}")
            
            # 1. Obtener datos del estado
            estados_ref = self.db.collection('states')
            estado_docs = estados_ref.where('states_name', '==', estado_nombre.lower()).limit(1).stream()
            
            estado_data = None
            for doc in estado_docs:
                estado_data = doc.to_dict()
                estado_data['id'] = doc.id
                break
            
            if not estado_data:
                print(f"❌ Estado no encontrado: {estado_nombre}")
                return None
            
            estado_id = estado_data['id']
            state_id_replacement = estado_data.get('state_id_replacement')
            
            # 2. Obtener todos los parámetros disponibles
            parametros_ref = self.db.collection('parameters')
            param_docs = parametros_ref.stream()
            
            parametros_data = {}
            
            for param_doc in param_docs:
                param_data = param_doc.to_dict()
                param_data['id'] = param_doc.id
                param_name = param_data.get('name')
                
                if not param_name:
                    continue
                
                # 3. Obtener texto de análisis específico
                special_texts_ref = self.db.collection('special_text')
                text_docs = special_texts_ref.where('states_r', '==', estado_id).where('parameter_r', '==', param_doc.id).limit(1).stream()
                
                texto_analisis = "No se encontró un texto de análisis para este parámetro."
                for text_doc in text_docs:
                    text_data = text_doc.to_dict()
                    texto_analisis = text_data.get('added_text', texto_analisis)
                    break
                
                # 4. Obtener datos de APIs relacionadas
                related_api_ids = param_data.get('related_apis', [])
                datos_apis = []
                
                if related_api_ids and state_id_replacement:
                    # Obtener APIs y hacer llamadas
                    apis_ref = self.db.collection('apis')
                    for api_id in related_api_ids:
                        api_doc = apis_ref.document(api_id).get()
                        if api_doc.exists:
                            api_data = api_doc.to_dict()
                            api_url = api_data.get('dynamic_url', '').replace('{state_id}', state_id_replacement)
                            
                            if api_url:
                                # Hacer llamada a la API
                                try:
                                    async with aiohttp.ClientSession() as session:
                                        async with session.get(api_url) as response:
                                            if response.status == 200:
                                                api_data_response = await response.json()
                                                datos_apis.append({
                                                    "nombre": api_data.get('apis_name', ''),
                                                    "url": api_url,
                                                    "datos": api_data_response
                                                })
                                            else:
                                                print(f"⚠️ Error en API {api_data.get('apis_name', '')}: {response.status}")
                                except Exception as e:
                                    print(f"⚠️ Error llamando API {api_data.get('apis_name', '')}: {e}")
                
                parametros_data[param_name] = ParametroData(
                    id=param_doc.id,
                    nombre=param_name,
                    texto_analisis=texto_analisis,
                    datos_apis=datos_apis
                )
            
            return EstadoData(
                id=estado_id,
                nombre=estado_nombre,
                state_id_replacement=state_id_replacement,
                parametros=parametros_data
            )
            
        except Exception as e:
            print(f"❌ Error obteniendo datos del estado: {e}")
            return None
    
    def _crear_contexto_estado(self, estado_data: EstadoData) -> str:
        """Crea un contexto estructurado con todos los datos del estado"""
        contexto = f"""
# INFORMACIÓN COMPLETA DEL ESTADO: {estado_data.nombre.upper()}

## DATOS GENERALES
- ID del Estado: {estado_data.id}
- ID para APIs: {estado_data.state_id_replacement}

## PARÁMETROS DE ANÁLISIS
"""
        
        for param_name, param_data in estado_data.parametros.items():
            contexto += f"""
### {param_name}
**Análisis:** {param_data.texto_analisis}

**Datos de APIs:**
"""
            if param_data.datos_apis:
                for api_data in param_data.datos_apis:
                    contexto += f"- **{api_data['nombre']}**: {json.dumps(api_data['datos'], ensure_ascii=False, indent=2)}\n"
            else:
                contexto += "- No hay datos de APIs disponibles\n"
            
            contexto += "\n"
        
        return contexto
    
    async def generar_respuesta_rag(self, pregunta: str, estado_nombre: str) -> Dict[str, Any]:
        """
        Genera una respuesta usando RAG basada en los datos del estado
        """
        if not self.openai_client:
            return {
                "error": "OpenAI no configurado",
                "respuesta": "Lo siento, el servicio de IA no está disponible en este momento."
            }
        
        try:
            # 1. Obtener datos completos del estado
            estado_data = await self.obtener_datos_estado_completo(estado_nombre)
            if not estado_data:
                return {
                    "error": "Estado no encontrado",
                    "respuesta": f"No se encontraron datos para el estado {estado_nombre}."
                }
            
            # 2. Crear contexto estructurado
            contexto = self._crear_contexto_estado(estado_data)
            
            # 3. Crear prompt para el LLM
            prompt = f"""
Eres un asistente especializado en análisis económico y de inversión para estados mexicanos. 
Tienes acceso a datos completos y actualizados sobre {estado_nombre}.

CONTEXTO DISPONIBLE:
{contexto}

PREGUNTA DEL USUARIO: {pregunta}

INSTRUCCIONES:
1. Responde de manera clara y profesional en español
2. Usa los datos específicos del estado cuando sea relevante
3. Si la pregunta es sobre un parámetro específico, enfócate en esos datos
4. Si no tienes información específica, indícalo claramente
5. Mantén un tono profesional pero accesible
6. Incluye datos numéricos cuando estén disponibles

RESPUESTA:
"""
            
            # 4. Generar respuesta con OpenAI
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres un experto en análisis económico de estados mexicanos."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            respuesta = response.choices[0].message.content
            
            return {
                "estado": estado_nombre,
                "pregunta": pregunta,
                "respuesta": respuesta,
                "timestamp": datetime.now().isoformat(),
                "parametros_disponibles": list(estado_data.parametros.keys()),
                "datos_utilizados": {
                    "estado_id": estado_data.id,
                    "parametros_con_datos": len([p for p in estado_data.parametros.values() if p.datos_apis])
                }
            }
            
        except Exception as e:
            print(f"❌ Error generando respuesta RAG: {e}")
            return {
                "error": "Error interno",
                "respuesta": "Lo siento, ocurrió un error al procesar tu consulta. Por favor, intenta de nuevo."
            }
    
    async def obtener_resumen_estado(self, estado_nombre: str) -> Dict[str, Any]:
        """Obtiene un resumen ejecutivo del estado"""
        estado_data = await self.obtener_datos_estado_completo(estado_nombre)
        if not estado_data:
            return {"error": "Estado no encontrado"}
        
        resumen = {
            "estado": estado_nombre,
            "parametros_disponibles": list(estado_data.parametros.keys()),
            "parametros_con_datos_apis": len([p for p in estado_data.parametros.values() if p.datos_apis]),
            "total_parametros": len(estado_data.parametros),
            "estado_id": estado_data.id,
            "api_id": estado_data.state_id_replacement
        }
        
        return resumen

# Instancia global del agente
rag_agent = RAGAgent()

# Funciones de utilidad para integración con FastAPI
async def consultar_estado_rag(estado_nombre: str, pregunta: str = None) -> Dict[str, Any]:
    """Función de utilidad para consultar el agente RAG"""
    if pregunta:
        return await rag_agent.generar_respuesta_rag(pregunta, estado_nombre)
    else:
        return await rag_agent.obtener_resumen_estado(estado_nombre)

async def obtener_estados_disponibles_rag() -> List[str]:
    """Obtiene lista de estados disponibles para el RAG"""
    try:
        estados_ref = rag_agent.db.collection('states')
        estados_docs = estados_ref.stream()
        
        estados = []
        for doc in estados_docs:
            estado_data = doc.to_dict()
            estados.append(estado_data.get('states_name', ''))
        
        return [e for e in estados if e]  # Filtrar valores vacíos
    except Exception as e:
        print(f"❌ Error obteniendo estados: {e}")
        return []

if __name__ == "__main__":
    # Ejemplo de uso
    async def ejemplo_uso():
        print("🤖 Iniciando ejemplo de uso del RAG Agent...")
        
        # Obtener lista de estados
        estados = await obtener_estados_disponibles_rag()
        print(f"📋 Estados disponibles: {estados[:5]}...")  # Mostrar solo los primeros 5
        
        if estados:
            # Ejemplo con el primer estado
            estado_ejemplo = estados[0]
            print(f"\n🔍 Analizando: {estado_ejemplo}")
            
            # Obtener resumen
            resumen = await consultar_estado_rag(estado_ejemplo)
            print(f"📊 Resumen: {json.dumps(resumen, indent=2, ensure_ascii=False)}")
            
            # Ejemplo de pregunta
            pregunta = "¿Cuáles son las principales oportunidades de inversión en este estado?"
            respuesta = await consultar_estado_rag(estado_ejemplo, pregunta)
            print(f"\n❓ Pregunta: {pregunta}")
            print(f"🤖 Respuesta: {respuesta.get('respuesta', 'Error')}")
    
    # Ejecutar ejemplo
    asyncio.run(ejemplo_uso())

