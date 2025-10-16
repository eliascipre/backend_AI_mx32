# Plan de Aplicación de LangChain en Rama Dev

## 🎯 Objetivo

Aplicar la integración completa de LangChain del proyecto `mx32-backend` en la rama `dev` del proyecto `Backend_IA_mx32`, manteniendo la compatibilidad con el sistema RAG existente.

## 📋 Análisis de la Situación Actual

### **Rama Actual (Backend_IA_mx32)**:
- ✅ Sistema RAG implementado
- ✅ Integración con Firebase
- ✅ Agente RAG funcional
- ❌ Sin características avanzadas de LangChain

### **Rama de Referencia (mx32-backend)**:
- ✅ LangChain Expression Language (LCEL)
- ✅ Output Parsers estructurados
- ✅ Few-Shot Prompting
- ✅ Conversation Entity Memory
- ✅ Function Calling
- ✅ Restricciones de datos específicas

## 🚀 Plan de Implementación

### **Fase 1: Preparación y Análisis**

#### 1.1 Backup y Preparación
```bash
# Crear backup de la rama dev actual
git checkout dev
git branch dev-backup-$(date +%Y%m%d)
git push origin dev-backup-$(date +%Y%m%d)

# Verificar estado actual
git status
git log --oneline -10
```

#### 1.2 Análisis de Compatibilidad
- [ ] Verificar compatibilidad entre sistemas RAG y LangChain
- [ ] Identificar conflictos potenciales
- [ ] Planificar estrategia de integración

### **Fase 2: Instalación de Dependencias**

#### 2.1 Actualizar requirements.txt
```python
# Agregar dependencias de LangChain al requirements_rag.txt existente
langchain==0.3.7
langchain-openai==0.2.8
langchain-core==0.3.15
langchain-community==0.3.7
```

#### 2.2 Instalación
```bash
pip install -r requirements_rag.txt
```

### **Fase 3: Implementación de Modelos**

#### 3.1 Crear Output Parsers
**Archivo**: `Backend_IA_mx32/models/output_parsers.py`

```python
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
    RAG_ANALYSIS = "rag_analysis"  # Nuevo para RAG

class ConfidenceLevel(str, Enum):
    """Niveles de confianza"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RAGAnalysisResponse(BaseModel):
    """Respuesta específica para análisis RAG"""
    estado: str = Field(description="Estado analizado")
    pregunta: str = Field(description="Pregunta original")
    respuesta: str = Field(description="Respuesta generada por RAG")
    parametros_utilizados: List[str] = Field(description="Parámetros utilizados en el análisis")
    datos_utilizados: Dict[str, Any] = Field(description="Datos específicos utilizados")
    confidence: ConfidenceLevel = Field(description="Nivel de confianza del análisis RAG")
    timestamp: datetime = Field(default_factory=datetime.now)

# ... resto de modelos existentes + nuevos para RAG
```

#### 3.2 Crear Modelos de Chatbot
**Archivo**: `Backend_IA_mx32/models/chatbot.py`

```python
"""
Modelos de datos para el chatbot MX32 con integración RAG
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
```

### **Fase 4: Implementación de Templates**

#### 4.1 Crear Few-Shot Templates
**Archivo**: `Backend_IA_mx32/prompts/few_shot_templates.py`

```python
"""
Templates de Few-Shot Prompting para el chatbot MX32 con RAG
"""

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from typing import List, Dict, Any

class FewShotTemplates:
    """Clase para gestionar templates de few-shot prompting con RAG"""
    
    def __init__(self):
        self.examples = self._load_examples()
    
    def _load_examples(self) -> Dict[str, List[Dict[str, str]]]:
        """Carga ejemplos para diferentes tipos de consultas con RAG"""
        return {
            "rag_analysis": [
                {
                    "input": "¿Cuáles son las principales oportunidades de inversión en Jalisco?",
                    "output": """## Análisis RAG de Oportunidades de Inversión en Jalisco

**Datos Utilizados:**
- Parámetros: Oportunidades Emergentes, Mapa de Sectores Claves
- APIs consultadas: 5 fuentes de datos económicos
- Texto de análisis: Análisis específico de Jalisco

**Insights Clave:**
- Sector tecnológico en crecimiento del 15% anual
- Clusters industriales bien desarrollados
- Infraestructura de conectividad superior al promedio nacional

**Recomendaciones Específicas:**
1. Invertir en parques tecnológicos especializados
2. Desarrollar clusters de manufactura avanzada
3. Fortalecer la educación técnica especializada

**Nivel de Confianza:** Alto (basado en datos RAG actualizados)"""
                }
            ],
            # ... otros templates existentes + nuevos para RAG
        }
    
    def get_rag_template(self) -> FewShotPromptTemplate:
        """Template específico para análisis RAG"""
        # Implementación del template RAG
        pass
```

### **Fase 5: Implementación de Herramientas**

#### 5.1 Crear Herramientas RAG
**Archivo**: `Backend_IA_mx32/tools/rag_tools.py`

```python
"""
Herramientas (Tools) para Function Calling en el chatbot MX32 con RAG
"""

from langchain_core.tools import tool
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import logging

from rag_service import rag_service

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

# Lista de herramientas RAG
RAG_TOOLS = [
    get_estado_rag_data,
    consulta_ia_estado,
]
```

### **Fase 6: Implementación del Agente Avanzado**

#### 6.1 Crear Agente RAG-LangChain
**Archivo**: `Backend_IA_mx32/services/rag_langchain_agent.py`

```python
"""
Agente de chatbot avanzado con LangChain y RAG integrado
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain.memory import ConversationBufferWindowMemory, ConversationEntityMemory
from langchain_core.tools import Tool

from models.output_parsers import RAGAnalysisResponse, StructuredAnalysisResponse
from models.chatbot import UserContext, UserRole, MessageRole
from prompts.few_shot_templates import FewShotTemplates
from tools.rag_tools import RAG_TOOLS
from rag_service import rag_service

logger = logging.getLogger(__name__)

class RAGLangChainAgent:
    """
    Agente de chatbot que combina LangChain con RAG
    """
    
    def __init__(self):
        # Configurar LLM
        self.llm = self._setup_llm()
        
        # Configurar herramientas (RAG + tradicionales)
        self.tools = self._setup_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
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
    
    def _setup_llm(self) -> ChatOpenAI:
        """Configura el modelo de lenguaje"""
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def _setup_tools(self) -> List[Tool]:
        """Configura las herramientas disponibles (RAG + tradicionales)"""
        return RAG_TOOLS
    
    def _setup_output_parsers(self) -> Dict[str, PydanticOutputParser]:
        """Configura los output parsers"""
        return {
            "rag_analysis": PydanticOutputParser(pydantic_object=RAGAnalysisResponse),
            "structured_analysis": PydanticOutputParser(pydantic_object=StructuredAnalysisResponse),
        }
    
    def _create_chat_chain(self):
        """Crea la cadena LCEL para chat general"""
        # Template del sistema con RAG
        system_template = """Eres un asistente especializado en análisis de datos de México (MX32) con capacidades RAG.
        Tu función es ayudar a los usuarios a analizar datos de estados y parámetros usando información actualizada.
        
        CAPACIDADES RAG:
        - Acceso a datos completos de Firebase
        - Análisis contextual de estados mexicanos
        - Respuestas basadas en datos reales y actualizados
        
        RESTRICCIÓN IMPORTANTE:
        - SOLO puedes proporcionar información sobre los 32 estados de México
        - Usa datos RAG cuando sea posible para mayor precisión
        
        Contexto del usuario:
        - Página: {current_page}
        - Estado: {current_state}
        - Parámetro: {current_parameter}
        
        Herramientas disponibles: {tool_names}
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
            | self.llm_with_tools
            | StrOutputParser()
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
            | self.llm
            | self.output_parsers["rag_analysis"]
        )
    
    async def process_chat_with_rag(
        self,
        messages: List[Dict[str, str]],
        user_context: UserContext,
        api_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Procesa un mensaje de chat usando RAG y LangChain
        """
        try:
            # Obtener el último mensaje
            last_message = messages[-1] if messages else {"role": "user", "content": ""}
            user_message = last_message["content"]
            
            # Determinar si usar RAG
            use_rag = user_context.rag_enabled and any(keyword in user_message.lower() for keyword in [
                "estado", "jalisco", "mexico", "análisis", "datos", "parámetro"
            ])
            
            if use_rag:
                # Usar RAG para obtener datos
                estado = user_context.current_state or "jalisco"  # Default
                rag_data = await rag_service.consulta_estado_con_rag(estado)
                
                # Procesar con cadena RAG
                rag_result = self.rag_chain.invoke({
                    "estado": estado,
                    "pregunta": user_message,
                    "rag_data": rag_data
                })
                
                return {
                    "response": rag_result.respuesta,
                    "rag_data": rag_data,
                    "confidence": 0.9,
                    "sources": ["Base de datos MX32", "RAG Analysis"],
                    "model_used": "gpt-4o-mini",
                    "rag_enabled": True
                }
            else:
                # Usar chat normal con LangChain
                context = {
                    "input": user_message,
                    "current_page": user_context.current_page,
                    "current_state": user_context.current_state,
                    "current_parameter": user_context.current_parameter,
                    "session_id": user_context.session_id,
                }
                
                response = self.chat_chain.invoke(context)
                
                return {
                    "response": response,
                    "confidence": 0.8,
                    "sources": ["Base de datos MX32"],
                    "model_used": "gpt-4o-mini",
                    "rag_enabled": False
                }
                
        except Exception as e:
            logger.error(f"Error procesando chat con RAG: {e}")
            return {
                "response": "Lo siento, ocurrió un error al procesar tu consulta.",
                "confidence": 0.0,
                "sources": [],
                "model_used": "gpt-4o-mini",
                "rag_enabled": False
            }

# Instancia global del agente
_rag_langchain_agent = None

def get_rag_langchain_agent() -> RAGLangChainAgent:
    """Obtiene la instancia global del agente RAG-LangChain"""
    global _rag_langchain_agent
    if _rag_langchain_agent is None:
        _rag_langchain_agent = RAGLangChainAgent()
    return _rag_langchain_agent
```

### **Fase 7: Implementación de Endpoints**

#### 7.1 Crear Endpoints RAG-LangChain
**Archivo**: `Backend_IA_mx32/rag_langchain_endpoints.py`

```python
"""
Endpoints RAG-LangChain para MX32 Backend
Combina capacidades RAG con características avanzadas de LangChain
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from services.rag_langchain_agent import get_rag_langchain_agent
from models.chatbot import ChatRequest, ChatResponse, UserContext, UserRole

# Router para endpoints RAG-LangChain
rag_langchain_router = APIRouter()

@rag_langchain_router.post("/chat-rag-langchain")
async def chat_with_rag_langchain(request: ChatRequest):
    """
    Endpoint principal para chat con RAG y LangChain.
    Combina las capacidades RAG existentes con características avanzadas de LangChain.
    """
    try:
        agent = get_rag_langchain_agent()
        
        # Crear contexto del usuario
        user_context = UserContext(
            user_id=request.user_id,
            user_type=UserRole.ANALYST,
            current_page=request.context.get("current_page") if request.context else None,
            current_state=request.context.get("current_state") if request.context else None,
            current_parameter=request.context.get("current_parameter") if request.context else None,
            session_id=request.session_id,
            preferences=request.context.get("preferences") if request.context else None,
            rag_enabled=request.use_rag
        )
        
        # Procesar con agente RAG-LangChain
        result = await agent.process_chat_with_rag(
            [{"role": "user", "content": request.message}],
            user_context,
            request.context.get("api_data") if request.context else None
        )
        
        return ChatResponse(
            response=result["response"],
            rag_data=result.get("rag_data"),
            confidence=result["confidence"],
            sources=result["sources"],
            model_used=result["model_used"],
            session_id=user_context.session_id,
            rag_enabled=result["rag_enabled"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en chat RAG-LangChain: {str(e)}"
        )

@rag_langchain_router.get("/health-rag-langchain")
async def health_check_rag_langchain():
    """
    Endpoint de salud para verificar el estado del sistema RAG-LangChain.
    """
    try:
        agent = get_rag_langchain_agent()
        
        return {
            "rag_langchain_habilitado": True,
            "rag_disponible": True,
            "langchain_disponible": True,
            "herramientas_disponibles": len(agent.tools),
            "memoria_activa": len(agent.conversation_memory),
            "servicios": {
                "rag": "✅ Activo",
                "langchain": "✅ Activo",
                "output_parsers": "✅ Activo",
                "few_shot": "✅ Activo",
                "function_calling": "✅ Activo"
            }
        }
        
    except Exception as e:
        return {
            "rag_langchain_habilitado": False,
            "error": str(e),
            "servicios": {
                "rag": "❌ Error",
                "langchain": "❌ Error"
            }
        }
```

### **Fase 8: Integración con Sistema Existente**

#### 8.1 Actualizar Servidor Principal
**Archivo**: `Backend_IA_mx32/rag_server_integrated.py`

```python
"""
Servidor FastAPI integrado con RAG y LangChain
"""

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Importar endpoints
from rag_endpoints import rag_router
from rag_langchain_endpoints import rag_langchain_router

def create_integrated_app() -> FastAPI:
    """Crear aplicación FastAPI integrada"""
    
    app = FastAPI(
        title="MX32 RAG + LangChain API",
        description="Sistema integrado de RAG y LangChain para análisis de estados mexicanos",
        version="2.0.0",
        debug=True
    )
    
    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Endpoint raíz
    @app.get("/", tags=["root"])
    def root():
        return {
            "message": "MX32 RAG + LangChain API",
            "version": "2.0.0",
            "features": ["RAG", "LangChain", "Output Parsers", "Few-Shot", "Function Calling"],
            "endpoints": {
                "rag": "/api/rag/",
                "rag_langchain": "/api/rag-langchain/"
            }
        }
    
    # Incluir routers
    app.include_router(rag_router, prefix="/api/rag", tags=["rag"])
    app.include_router(rag_langchain_router, prefix="/api/rag-langchain", tags=["rag-langchain"])
    
    return app

# Crear aplicación
app = create_integrated_app()

if __name__ == "__main__":
    logger.info("🚀 Iniciando MX32 RAG + LangChain API")
    uvicorn.run(
        "rag_server_integrated:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
```

### **Fase 9: Testing y Validación**

#### 9.1 Crear Tests de Integración
**Archivo**: `Backend_IA_mx32/test_rag_langchain_integration.py`

```python
"""
Tests de integración RAG + LangChain
"""

import asyncio
import pytest
from services.rag_langchain_agent import get_rag_langchain_agent
from models.chatbot import UserContext, UserRole

async def test_rag_langchain_integration():
    """Test de integración RAG + LangChain"""
    
    agent = get_rag_langchain_agent()
    
    # Test 1: Chat con RAG habilitado
    user_context = UserContext(
        user_id="test_user",
        user_type=UserRole.ANALYST,
        current_state="jalisco",
        current_parameter="Situación de Seguridad",
        session_id="test_session",
        rag_enabled=True
    )
    
    result = await agent.process_chat_with_rag(
        [{"role": "user", "content": "¿Cuál es la situación de seguridad en Jalisco?"}],
        user_context
    )
    
    assert result["rag_enabled"] == True
    assert "response" in result
    assert result["confidence"] > 0.0
    
    print("✅ Test RAG + LangChain exitoso")

if __name__ == "__main__":
    asyncio.run(test_rag_langchain_integration())
```

### **Fase 10: Documentación y Deployment**

#### 10.1 Actualizar README
**Archivo**: `Backend_IA_mx32/README_RAG_LANGCHAIN.md`

```markdown
# MX32 RAG + LangChain Integration

## 🚀 Características

- **RAG (Retrieval-Augmented Generation)**: Acceso a datos reales de Firebase
- **LangChain Expression Language (LCEL)**: Cadenas de procesamiento elegantes
- **Output Parsers**: Respuestas estructuradas y consistentes
- **Few-Shot Prompting**: Mejores respuestas contextuales
- **Function Calling**: Herramientas especializadas
- **Conversation Memory**: Memoria de conversación persistente

## 🔧 Uso

### Chat con RAG + LangChain
```python
POST /api/rag-langchain/chat-rag-langchain
{
    "message": "¿Cuáles son las oportunidades de inversión en Jalisco?",
    "user_id": "user123",
    "use_rag": true,
    "context": {
        "current_state": "jalisco",
        "current_parameter": "Oportunidades Emergentes"
    }
}
```

### Health Check
```python
GET /api/rag-langchain/health-rag-langchain
```

## 📊 Respuesta Estructurada

```json
{
    "response": "Respuesta del asistente",
    "rag_data": {
        "estado": "jalisco",
        "parametros_utilizados": ["Oportunidades Emergentes"],
        "datos_apis": [...]
    },
    "confidence": 0.9,
    "sources": ["Base de datos MX32", "RAG Analysis"],
    "model_used": "gpt-4o-mini",
    "rag_enabled": true
}
```
```

## 📅 Cronograma de Implementación

### **Semana 1: Preparación**
- [ ] Backup de rama dev
- [ ] Análisis de compatibilidad
- [ ] Instalación de dependencias

### **Semana 2: Modelos y Templates**
- [ ] Implementar output parsers
- [ ] Crear modelos de chatbot
- [ ] Implementar few-shot templates

### **Semana 3: Herramientas y Agente**
- [ ] Crear herramientas RAG
- [ ] Implementar agente RAG-LangChain
- [ ] Integrar con sistema existente

### **Semana 4: Endpoints y Testing**
- [ ] Crear endpoints integrados
- [ ] Implementar tests
- [ ] Documentación final

## 🎯 Resultado Esperado

Al final de la implementación, tendremos:

1. **Sistema RAG existente** ✅ (mantenido)
2. **Características LangChain** ✅ (nuevas)
3. **Integración perfecta** ✅ (objetivo)
4. **Compatibilidad total** ✅ (requisito)
5. **Documentación completa** ✅ (necesario)

## 🚨 Consideraciones Importantes

1. **Compatibilidad**: Mantener funcionalidad RAG existente
2. **Performance**: Optimizar para respuestas rápidas
3. **Testing**: Validar todas las funcionalidades
4. **Documentación**: Actualizar README y docs
5. **Deployment**: Preparar para producción

---

**Estado**: 📋 **PLAN PREPARADO**  
**Fecha**: Diciembre 2024  
**Versión**: 2.0.0

