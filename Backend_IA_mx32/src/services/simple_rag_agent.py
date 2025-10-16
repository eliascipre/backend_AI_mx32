"""
Agente RAG simplificado con Cerebras
Versión simplificada sin LangChain complejo
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from src.models.chatbot import UserContext, UserRole, MessageRole, ChatResponseStructured
from src.services.cerebras_client import get_cerebras_client
from src.rag_service import rag_service

logger = logging.getLogger(__name__)

class SimpleRAGAgent:
    """
    Agente RAG simplificado que usa Cerebras
    """
    
    def __init__(self):
        # Configurar Cerebras
        self.cerebras_client = get_cerebras_client()
        
        # Memoria simple
        self.conversation_memory: Dict[str, List[Dict[str, str]]] = {}
        
    def _get_conversation_memory(self, session_id: str) -> List[Dict[str, str]]:
        """Obtiene la memoria de conversación"""
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = []
        return self.conversation_memory[session_id]
    
    def _extract_entities_from_message(self, message: str) -> List[str]:
        """Extrae entidades del mensaje del usuario"""
        entities = []
        
        # Palabras clave relacionadas con México
        mexico_keywords = [
            "mexico", "méxico", "estados", "estado", "ciudad", "municipio",
            "seguridad", "economia", "economía", "desarrollo", "infraestructura",
            "educacion", "educación", "salud", "empleo", "pobreza", "crimen"
        ]
        
        message_lower = message.lower()
        for keyword in mexico_keywords:
            if keyword in message_lower:
                entities.append(keyword)
        
        return entities
    
    def _check_mexico_restriction(self, message: str) -> tuple[bool, str]:
        """Verifica si la pregunta es sobre datos de México o externos"""
        message_lower = message.lower()
        
        # Palabras clave que indican países o regiones externas
        external_countries = [
            "estados unidos", "usa", "eeuu", "united states", "america",
            "canada", "canadá", "brasil", "argentina", "colombia", "chile",
            "peru", "perú", "venezuela", "ecuador", "bolivia", "paraguay",
            "uruguay", "españa", "spain", "francia", "france", "alemania",
            "germany", "italia", "italy", "china", "japon", "japón",
            "corea", "korea", "india", "rusia", "russia", "australia",
            "nueva zelanda", "new zealand", "reino unido", "uk", "inglaterra"
        ]
        
        # Verificar si menciona países externos
        for country in external_countries:
            if country in message_lower:
                return False, f"Lo siento, pero solo puedo proporcionar información sobre datos de México. Mi base de datos contiene únicamente información de los 32 estados mexicanos que puedes visualizar en las gráficas de la plataforma. ¿Te gustaría que te ayude con información sobre algún estado específico de México? 🇲🇽"
        
        # Verificar si es una pregunta sobre México o estados mexicanos
        mexico_indicators = [
            "mexico", "méxico", "estados", "estado", "jalisco", "nuevo león",
            "ciudad de méxico", "cdmx", "yucatán", "quintana roo", "campeche",
            "tabasco", "chiapas", "oaxaca", "veracruz", "puebla", "tlaxcala",
            "hidalgo", "querétaro", "san luis potosí", "tamaulipas", "coahuila",
            "durango", "zacatecas", "aguascalientes", "guanajuato", "michoacán",
            "guerrero", "morelos", "mexico", "baja california", "baja california sur",
            "sonora", "sinaloa", "nayarit", "colima"
        ]
        
        has_mexico_context = any(indicator in message_lower for indicator in mexico_indicators)
        
        if not has_mexico_context:
            return False, "Solo puedo ayudarte con información sobre los 32 estados de México. Mi base de datos contiene datos específicos de seguridad, economía, infraestructura, educación y otros parámetros de los estados mexicanos que puedes ver en las gráficas de la plataforma. ¿Sobre qué estado de México te gustaría saber? 🇲🇽"
        
        return True, ""
    
    async def process_chat_with_rag(
        self,
        messages: List[Dict[str, str]],
        user_context: UserContext,
        api_data: Optional[Dict] = None
    ) -> ChatResponseStructured:
        """
        Procesa un mensaje de chat usando RAG y Cerebras
        """
        try:
            # Obtener el último mensaje
            last_message = messages[-1] if messages else {"role": "user", "content": ""}
            user_message = last_message["content"]
            
            # Verificar restricción de México
            is_mexico_related, restriction_message = self._check_mexico_restriction(user_message)
            if not is_mexico_related:
                return ChatResponseStructured(
                    response=restriction_message,
                    confidence=1.0,
                    sources=["Sistema de restricciones MX32"],
                    suggested_actions=[
                        "Ver datos de estados mexicanos",
                        "Explorar parámetros disponibles",
                        "Consultar información de seguridad",
                        "Analizar datos económicos"
                    ],
                    follow_up_questions=[
                        "¿Sobre qué estado de México te gustaría saber?",
                        "¿Qué parámetro te interesa analizar?",
                        "¿Te gustaría ver una comparación entre estados?"
                    ],
                    model_used="cerebras-gpt-oss-120b",
                    session_id=user_context.session_id or "default",
                    memory_used=True,
                    rag_enabled=False
                )
            
            # Obtener memoria de conversación
            session_id = user_context.session_id or "default"
            conversation_memory = self._get_conversation_memory(session_id)
            
            # Extraer entidades
            entities = self._extract_entities_from_message(user_message)
            
            # Determinar si usar RAG
            use_rag = user_context.rag_enabled and any(keyword in user_message.lower() for keyword in [
                "estado", "jalisco", "mexico", "análisis", "datos", "parámetro", "rag"
            ])
            
            if use_rag:
                # Usar RAG para obtener datos
                estado = user_context.current_state or "jalisco"  # Default
                try:
                    rag_data = await rag_service.consulta_estado_con_rag(estado)
                    
                    # Crear prompt con datos RAG
                    system_prompt = f"""Eres un experto en análisis de datos de México con acceso a información RAG actualizada.
                    
                    Datos RAG disponibles para {estado}:
                    {json.dumps(rag_data, ensure_ascii=False, indent=2)}
                    
                    Responde la pregunta del usuario basándote en estos datos reales y actualizados.
                    Siempre menciona que la información proviene de datos RAG actualizados.
                    """
                    
                    response_text = await self.cerebras_client.generate_response(
                        system_prompt=system_prompt,
                        user_message=user_message,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    
                    rag_enabled = True
                    sources = ["Base de datos MX32", "RAG Analysis", "Cerebras AI"]
                    confidence = 0.9
                    
                except Exception as e:
                    logger.error(f"Error en RAG: {e}")
                    # Fallback sin RAG
                    system_prompt = "Eres un experto en análisis de datos de México. Responde la pregunta del usuario."
                    response_text = await self.cerebras_client.generate_response(
                        system_prompt=system_prompt,
                        user_message=user_message,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    rag_enabled = False
                    sources = ["Base de datos MX32", "Cerebras AI"]
                    confidence = 0.8
            else:
                # Usar solo Cerebras sin RAG
                system_prompt = """Eres un experto en análisis de datos de México. 
                Responde la pregunta del usuario de manera profesional y útil.
                Siempre menciona que solo tienes información sobre los 32 estados de México."""
                
                response_text = await self.cerebras_client.generate_response(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                rag_enabled = False
                sources = ["Base de datos MX32", "Cerebras AI"]
                confidence = 0.8
            
            # Generar acciones sugeridas
            suggested_actions = []
            if user_context.current_state:
                suggested_actions.append(f"Ver datos completos de {user_context.current_state}")
                suggested_actions.append(f"Comparar {user_context.current_state} con otros estados")
            
            if "seguridad" in entities:
                suggested_actions.append("Ver análisis detallado de seguridad")
                suggested_actions.append("Comparar indicadores de seguridad")
            
            if "economia" in entities or "economía" in entities:
                suggested_actions.append("Analizar indicadores económicos")
                suggested_actions.append("Ver datos de empleo")
            
            # Generar preguntas de seguimiento
            follow_up_questions = []
            if user_context.current_state:
                follow_up_questions.append(f"¿Te gustaría comparar {user_context.current_state} con otros estados?")
                follow_up_questions.append(f"¿Qué otros parámetros te interesan de {user_context.current_state}?")
            
            if "seguridad" in entities:
                follow_up_questions.append("¿Te interesa analizar las tendencias de seguridad?")
            
            # Actualizar memoria
            conversation_memory.append({"role": "user", "content": user_message})
            conversation_memory.append({"role": "assistant", "content": response_text})
            
            # Mantener solo los últimos 10 intercambios
            if len(conversation_memory) > 20:
                conversation_memory = conversation_memory[-20:]
                self.conversation_memory[session_id] = conversation_memory
            
            return ChatResponseStructured(
                response=response_text,
                confidence=confidence,
                sources=sources,
                suggested_actions=suggested_actions[:5],
                follow_up_questions=follow_up_questions[:3],
                model_used="cerebras-gpt-oss-120b",
                session_id=session_id,
                memory_used=True,
                rag_enabled=rag_enabled,
                entity_data={
                    "states_mentioned": entities,
                    "entities": {"general": entities},
                    "intent": "rag_analysis" if rag_enabled else "general_analysis"
                }
            )
            
        except Exception as e:
            logger.error(f"Error procesando chat con RAG: {e}")
            return ChatResponseStructured(
                response="Lo siento, ocurrió un error al procesar tu consulta. Por favor, inténtalo de nuevo.",
                confidence=0.0,
                sources=[],
                model_used="cerebras-gpt-oss-120b",
                session_id=user_context.session_id or "default",
                rag_enabled=False
            )

# Instancia global del agente
_simple_rag_agent = None

def get_simple_rag_agent() -> SimpleRAGAgent:
    """Obtiene la instancia global del agente RAG simplificado"""
    global _simple_rag_agent
    if _simple_rag_agent is None:
        _simple_rag_agent = SimpleRAGAgent()
    return _simple_rag_agent
