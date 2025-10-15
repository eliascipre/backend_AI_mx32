"""
Agente de chatbot avanzado con LangChain, RAG y Cerebras
Integra todas las características avanzadas para análisis de estados mexicanos
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import asyncio

from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain.memory import ConversationBufferWindowMemory, ConversationEntityMemory
from langchain_core.tools import Tool

from src.models.chatbot import ChatResponseStructured
from src.models.chatbot import UserContext, UserRole, MessageRole
from src.prompts.few_shot_templates import FewShotTemplates
from src.tools.rag_tools import RAG_TOOLS
from src.services.cerebras_client import get_cerebras_client
from src.rag_service import rag_service

logger = logging.getLogger(__name__)

class RAGLangChainAgent:
    """
    Agente de chatbot que combina LangChain, RAG y Cerebras
    """
    
    def __init__(self):
        # Configurar Cerebras
        self.cerebras_client = get_cerebras_client()
        
        # Configurar herramientas (RAG + tradicionales)
        self.tools = self._setup_tools()
        
        # Configurar memorias
        self.conversation_memory: Dict[str, ConversationBufferWindowMemory] = {}
        self.entity_memory: Dict[str, ConversationEntityMemory] = {}
        
        # Configurar templates
        self.few_shot_templates = FewShotTemplates()
        
        # Configurar output parsers
        self.output_parsers = self._setup_output_parsers()
        
        # Crear cadenas LCEL
        self.chat_chain = self._create_chat_chain()
        self.rag_chain = self._create_rag_chain()
        self.analysis_chain = self._create_analysis_chain()
        
    def _setup_tools(self) -> List[Tool]:
        """Configura las herramientas disponibles (RAG + tradicionales)"""
        return RAG_TOOLS
    
    def _setup_output_parsers(self) -> Dict[str, PydanticOutputParser]:
        """Configura los output parsers"""
        return {
            "rag_analysis": PydanticOutputParser(pydantic_object=RAGAnalysisResponse),
            "structured_analysis": PydanticOutputParser(pydantic_object=StructuredAnalysisResponse),
            "comparison": PydanticOutputParser(pydantic_object=ComparisonResponse),
            "trend_analysis": PydanticOutputParser(pydantic_object=TrendAnalysisResponse),
            "entity_extraction": PydanticOutputParser(pydantic_object=EntityExtractionResponse),
            "function_call": PydanticOutputParser(pydantic_object=FunctionCallResponse)
        }
    
    def _create_chat_chain(self):
        """Crea la cadena LCEL para chat general"""
        # Template del sistema con RAG
        system_template = """Eres un asistente especializado en análisis de datos de México (MX32) con capacidades RAG avanzadas.
        Tu función es ayudar a los usuarios a analizar datos de estados y parámetros usando información actualizada de Firebase.
        
        CAPACIDADES RAG:
        - Acceso a datos completos de Firebase
        - Análisis contextual de estados mexicanos
        - Respuestas basadas en datos reales y actualizados
        
        RESTRICCIÓN IMPORTANTE:
        - SOLO puedes proporcionar información sobre los 32 estados de México
        - Usa datos RAG cuando sea posible para mayor precisión
        - Si el usuario pregunta sobre otros países, explica que solo tienes datos de México
        
        Contexto del usuario:
        - Página: {current_page}
        - Estado: {current_state}
        - Parámetro: {current_parameter}
        
        Herramientas disponibles: {tool_names}
        
        Instrucciones:
        - Siempre responde en español
        - Usa un tono profesional pero accesible
        - Proporciona datos específicos cuando sea posible
        - Sugiere acciones concretas basadas en los datos
        - Incluye la bandera de México 🇲🇽 cuando menciones datos del país
        """
        
        # Template de mensajes
        chat_template = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Cadena LCEL
        return (
            {
                "input": RunnablePassthrough(),
                "current_page": RunnableLambda(lambda x: x.get("current_page", "N/A")),
                "current_state": RunnableLambda(lambda x: x.get("current_state", "N/A")),
                "current_parameter": RunnableLambda(lambda x: x.get("current_parameter", "N/A")),
                "tool_names": RunnableLambda(lambda x: ", ".join([tool.name for tool in self.tools])),
                "chat_history": RunnableLambda(lambda x: self._get_chat_history(x.get("session_id", "default"))),
                "agent_scratchpad": RunnableLambda(lambda x: [])
            }
            | chat_template
            | RunnableLambda(lambda x: self._process_with_cerebras(x))
        )
    
    def _create_rag_chain(self):
        """Crea la cadena LCEL específica para RAG"""
        rag_template = """Analiza la siguiente consulta usando datos RAG:
        
        Estado: {estado}
        Pregunta: {pregunta}
        Datos RAG: {rag_data}
        
        Proporciona:
        1. Respuesta basada en datos RAG
        2. Insights específicos del estado
        3. Recomendaciones contextuales
        4. Nivel de confianza basado en datos
        
        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_template(rag_template)
        
        return (
            {
                "estado": RunnablePassthrough(),
                "pregunta": RunnablePassthrough(),
                "rag_data": RunnablePassthrough(),
                "format_instructions": RunnableLambda(lambda x: self.output_parsers["rag_analysis"].get_format_instructions())
            }
            | prompt
            | RunnableLambda(lambda x: self._process_with_cerebras(x))
            | self.output_parsers["rag_analysis"]
        )
    
    def _create_analysis_chain(self):
        """Crea la cadena LCEL para análisis estructurado"""
        analysis_template = """Analiza los siguientes datos y proporciona un análisis estructurado:
        
        Estado: {state}
        Parámetro: {parameter}
        Datos: {data}
        
        Proporciona:
        1. Análisis principal
        2. Insights clave (máximo 5)
        3. Recomendaciones específicas (máximo 5)
        4. Métricas clave extraídas
        5. Nivel de confianza del análisis
        6. Tipo de análisis realizado
        7. Contexto específico del estado
        
        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_template(analysis_template)
        
        return (
            {
                "state": RunnablePassthrough(),
                "parameter": RunnablePassthrough(),
                "data": RunnablePassthrough(),
                "format_instructions": RunnableLambda(lambda x: self.output_parsers["structured_analysis"].get_format_instructions())
            }
            | prompt
            | RunnableLambda(lambda x: self._process_with_cerebras(x))
            | self.output_parsers["structured_analysis"]
        )
    
    async def _process_with_cerebras(self, prompt_data: Dict[str, Any]) -> str:
        """Procesa un prompt usando Cerebras"""
        try:
            if isinstance(prompt_data, dict) and "messages" in prompt_data:
                # Si ya es un formato de mensajes
                messages = prompt_data["messages"]
            else:
                # Convertir a formato de mensajes
                system_prompt = prompt_data.get("system", "")
                user_input = prompt_data.get("human", "")
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
            
            response = await self.cerebras_client.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=False
            )
            
            if "error" in response:
                return f"Error: {response['error']}"
            
            try:
                return response['choices'][0]['message']['content']
            except (KeyError, IndexError):
                return "Error: No se pudo obtener la respuesta"
                
        except Exception as e:
            logger.error(f"Error procesando con Cerebras: {e}")
            return f"Error: {str(e)}"
    
    def _get_conversation_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Obtiene o crea la memoria de conversación"""
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = ConversationBufferWindowMemory(
                k=10,
                return_messages=True
            )
        return self.conversation_memory[session_id]
    
    def _get_entity_memory(self, session_id: str) -> ConversationEntityMemory:
        """Obtiene o crea la memoria de entidades"""
        if session_id not in self.entity_memory:
            # Crear un LLM dummy para la memoria de entidades
            from langchain_openai import ChatOpenAI
            dummy_llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                openai_api_key="dummy"
            )
            
            self.entity_memory[session_id] = ConversationEntityMemory(
                llm=dummy_llm,
                memory_key="entity_memory",
                return_messages=True
            )
        return self.entity_memory[session_id]
    
    def _get_chat_history(self, session_id: str) -> List:
        """Obtiene el historial de chat"""
        memory = self._get_conversation_memory(session_id)
        if memory.chat_memory.messages:
            messages = memory.chat_memory.messages[-10:]
            return messages
        return []
    
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
    
    def _determine_intent(self, message: str) -> str:
        """Determina la intención del mensaje del usuario"""
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in ["comparar", "comparación", "vs", "versus"]):
            return "comparison"
        elif any(keyword in message_lower for keyword in ["tendencia", "histórico", "evolución", "cambio"]):
            return "trend_analysis"
        elif any(keyword in message_lower for keyword in ["recomendación", "sugerencia", "qué hacer"]):
            return "recommendations"
        elif any(keyword in message_lower for keyword in ["similar", "parecido", "como"]):
            return "similar_states"
        elif any(keyword in message_lower for keyword in ["rag", "datos", "firebase", "análisis"]):
            return "rag_analysis"
        else:
            return "general_analysis"
    
    async def process_chat_with_rag(
        self,
        messages: List[Dict[str, str]],
        user_context: UserContext,
        api_data: Optional[Dict] = None
    ) -> ChatResponseStructured:
        """
        Procesa un mensaje de chat usando RAG, LangChain y Cerebras
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
            
            # Determinar intención
            intent = self._determine_intent(user_message)
            
            # Obtener memorias
            session_id = user_context.session_id or "default"
            conversation_memory = self._get_conversation_memory(session_id)
            entity_memory = self._get_entity_memory(session_id)
            
            # Extraer entidades
            entities = self._extract_entities_from_message(user_message)
            
            # Procesar según la intención
            if intent == "rag_analysis" or user_context.rag_enabled:
                response = await self._handle_rag_analysis_intent(user_message, user_context, api_data)
            elif intent == "comparison":
                response = await self._handle_comparison_intent(user_message, user_context, api_data)
            elif intent == "trend_analysis":
                response = await self._handle_trend_analysis_intent(user_message, user_context, api_data)
            elif intent == "recommendations":
                response = await self._handle_recommendations_intent(user_message, user_context, api_data)
            elif intent == "similar_states":
                response = await self._handle_similar_states_intent(user_message, user_context, api_data)
            else:
                response = await self._handle_general_analysis_intent(user_message, user_context, api_data)
            
            # Actualizar memorias
            conversation_memory.chat_memory.add_user_message(user_message)
            conversation_memory.chat_memory.add_ai_message(response.response)
            
            # Actualizar memoria de entidades
            entity_memory.save_context(
                {"input": user_message},
                {"output": response.response}
            )
            
            return response
            
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
    
    async def _handle_rag_analysis_intent(self, user_message: str, user_context: UserContext, api_data: Optional[Dict]) -> ChatResponseStructured:
        """Maneja intenciones de análisis RAG"""
        try:
            # Obtener datos RAG
            estado = user_context.current_state or "jalisco"  # Default
            rag_data = await rag_service.consulta_estado_con_rag(estado)
            
            # Usar few-shot prompting para RAG
            few_shot_template = self.few_shot_templates.get_rag_template()
            prompt = few_shot_template.format(input=user_message)
            
            # Generar respuesta con Cerebras
            response_text = await self.cerebras_client.generate_response(
                system_prompt="Eres un experto en análisis de datos de México con capacidades RAG.",
                user_message=prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            # Extraer entidades
            entities = self._extract_entities_from_message(user_message)
            
            return ChatResponseStructured(
                response=response_text,
                confidence=0.9,
                sources=["Base de datos MX32", "RAG Analysis", "Cerebras AI"],
                suggested_actions=self._generate_suggested_actions(user_context, entities),
                follow_up_questions=self._generate_follow_up_questions(user_context, entities),
                model_used="cerebras-gpt-oss-120b",
                session_id=user_context.session_id or "default",
                memory_used=True,
                rag_enabled=True,
                rag_data={
                    "estado": estado,
                    "datos_utilizados": rag_data,
                    "parametros_consultados": list(rag_data.get("datos_por_parametro", {}).keys()) if rag_data else []
                },
                entity_data=EntityExtractionResponse(
                    states_mentioned=entities,
                    parameters_mentioned=[],
                    metrics_mentioned=[],
                    time_periods=[],
                    entities={"rag": entities},
                    intent="rag_analysis"
                )
            )
            
        except Exception as e:
            logger.error(f"Error en análisis RAG: {e}")
            return ChatResponseStructured(
                response="Lo siento, no pude acceder a los datos RAG en este momento.",
                confidence=0.5,
                sources=["Sistema RAG"],
                model_used="cerebras-gpt-oss-120b",
                session_id=user_context.session_id or "default",
                rag_enabled=False
            )
    
    async def _handle_general_analysis_intent(self, user_message: str, user_context: UserContext, api_data: Optional[Dict]) -> ChatResponseStructured:
        """Maneja intenciones de análisis general"""
        try:
            # Usar few-shot prompting
            few_shot_template = self.few_shot_templates.get_template_by_intent(user_message)
            prompt = few_shot_template.format(input=user_message)
            
            # Generar respuesta con Cerebras
            response_text = await self.cerebras_client.generate_response(
                system_prompt="Eres un experto en análisis de datos de México.",
                user_message=prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            # Extraer entidades
            entities = self._extract_entities_from_message(user_message)
            
            return ChatResponseStructured(
                response=response_text,
                confidence=0.8,
                sources=["Base de datos MX32", "Análisis en tiempo real"],
                suggested_actions=self._generate_suggested_actions(user_context, entities),
                follow_up_questions=self._generate_follow_up_questions(user_context, entities),
                model_used="cerebras-gpt-oss-120b",
                session_id=user_context.session_id or "default",
                memory_used=True,
                rag_enabled=False,
                entity_data=EntityExtractionResponse(
                    states_mentioned=entities,
                    parameters_mentioned=[],
                    metrics_mentioned=[],
                    time_periods=[],
                    entities={"general": entities},
                    intent="general_analysis"
                )
            )
            
        except Exception as e:
            logger.error(f"Error en análisis general: {e}")
            return ChatResponseStructured(
                response="Lo siento, ocurrió un error al procesar tu consulta.",
                confidence=0.5,
                sources=["Sistema de análisis"],
                model_used="cerebras-gpt-oss-120b",
                session_id=user_context.session_id or "default",
                rag_enabled=False
            )
    
    async def _handle_comparison_intent(self, user_message: str, user_context: UserContext, api_data: Optional[Dict]) -> ChatResponseStructured:
        """Maneja intenciones de comparación"""
        return ChatResponseStructured(
            response="Funcionalidad de comparación en desarrollo...",
            confidence=0.7,
            sources=["Base de datos MX32"],
            model_used="cerebras-gpt-oss-120b",
            session_id=user_context.session_id or "default",
            rag_enabled=False
        )
    
    async def _handle_trend_analysis_intent(self, user_message: str, user_context: UserContext, api_data: Optional[Dict]) -> ChatResponseStructured:
        """Maneja intenciones de análisis de tendencias"""
        return ChatResponseStructured(
            response="Funcionalidad de análisis de tendencias en desarrollo...",
            confidence=0.7,
            sources=["Base de datos MX32"],
            model_used="cerebras-gpt-oss-120b",
            session_id=user_context.session_id or "default",
            rag_enabled=False
        )
    
    async def _handle_recommendations_intent(self, user_message: str, user_context: UserContext, api_data: Optional[Dict]) -> ChatResponseStructured:
        """Maneja intenciones de recomendaciones"""
        return ChatResponseStructured(
            response="Funcionalidad de recomendaciones en desarrollo...",
            confidence=0.7,
            sources=["Base de datos MX32"],
            model_used="cerebras-gpt-oss-120b",
            session_id=user_context.session_id or "default",
            rag_enabled=False
        )
    
    async def _handle_similar_states_intent(self, user_message: str, user_context: UserContext, api_data: Optional[Dict]) -> ChatResponseStructured:
        """Maneja intenciones de búsqueda de estados similares"""
        return ChatResponseStructured(
            response="Funcionalidad de estados similares en desarrollo...",
            confidence=0.7,
            sources=["Base de datos MX32"],
            model_used="cerebras-gpt-oss-120b",
            session_id=user_context.session_id or "default",
            rag_enabled=False
        )
    
    def _generate_suggested_actions(self, user_context: UserContext, entities: List[str]) -> List[str]:
        """Genera acciones sugeridas"""
        actions = []
        
        if user_context.current_state:
            actions.append(f"Ver datos completos de {user_context.current_state}")
            actions.append(f"Comparar {user_context.current_state} con otros estados")
        
        if "seguridad" in entities:
            actions.append("Ver análisis detallado de seguridad")
            actions.append("Comparar indicadores de seguridad")
        
        if "economia" in entities or "economía" in entities:
            actions.append("Analizar indicadores económicos")
            actions.append("Ver datos de empleo")
        
        return actions[:5]
    
    def _generate_follow_up_questions(self, user_context: UserContext, entities: List[str]) -> List[str]:
        """Genera preguntas de seguimiento"""
        questions = []
        
        if user_context.current_state:
            questions.append(f"¿Te gustaría comparar {user_context.current_state} con otros estados?")
            questions.append(f"¿Qué otros parámetros te interesan de {user_context.current_state}?")
        
        if "seguridad" in entities:
            questions.append("¿Te interesa analizar las tendencias de seguridad?")
        
        return questions[:3]

# Instancia global del agente
_rag_langchain_agent = None

def get_rag_langchain_agent() -> RAGLangChainAgent:
    """Obtiene la instancia global del agente RAG-LangChain"""
    global _rag_langchain_agent
    if _rag_langchain_agent is None:
        _rag_langchain_agent = RAGLangChainAgent()
    return _rag_langchain_agent

