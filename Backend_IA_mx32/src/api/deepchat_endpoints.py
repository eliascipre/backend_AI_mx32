"""
Endpoints para Deep Chat con RAG, LangChain y Cerebras
Optimizados para el frontend de mx32-frontend
"""

import os
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.models.chatbot import (
    ChatRequest, DeepChatRequest, DeepChatMessage, ChatResponse, 
    UserContext, UserRole, ChatMessage, ChatResponseStructured
)
from src.services.simple_rag_agent import get_simple_rag_agent

logger = logging.getLogger(__name__)

# Funciones de conversión de markdown a HTML
def convert_markdown_table_to_html(text: str) -> str:
    """Convierte tablas markdown a HTML"""
    lines = text.split('\n')
    html_lines = []
    in_table = False
    
    for line in lines:
        if '|' in line and line.strip():
            if not in_table:
                html_lines.append('<div class="table-responsive">')
                html_lines.append('<table class="table table-striped table-bordered">')
                in_table = True
            
            # Limpiar la línea
            clean_line = line.strip()
            if clean_line.startswith('|') and clean_line.endswith('|'):
                cells = [cell.strip() for cell in clean_line[1:-1].split('|')]
                
                if '---' in line or '===' in line:
                    # Es una línea de separación, la saltamos
                    continue
                else:
                    # Es una fila de datos
                    html_cells = ''.join([f'<td class="table-cell-content">{cell}</td>' for cell in cells])
                    html_lines.append(f'<tr>{html_cells}</tr>')
        else:
            if in_table:
                html_lines.append('</table>')
                html_lines.append('</div>')
                in_table = False
            html_lines.append(line)
    
    if in_table:
        html_lines.append('</table>')
        html_lines.append('</div>')
    
    return '\n'.join(html_lines)

def convert_markdown_to_html(text: str) -> str:
    """Convierte elementos markdown básicos a HTML"""
    # Convertir negritas
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.*?)__', r'<strong>\1</strong>', text)
    
    # Convertir cursivas
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.*?)_', r'<em>\1</em>', text)
    
    # Convertir títulos
    text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    
    # Convertir listas
    text = re.sub(r'^\- (.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\. (.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    
    # Agrupar listas
    text = re.sub(r'(<li>.*?</li>)', r'<ul>\1</ul>', text, flags=re.DOTALL)
    
    # Convertir saltos de línea
    text = text.replace('\n', '<br>')
    
    return text

router = APIRouter(prefix="/deepchat", tags=["deepchat"])
security = HTTPBearer()

# Almacenamiento temporal de conversaciones (en producción usar Redis o DB)
conversation_storage: Dict[str, list] = {}

async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Dict[str, Any]:
    """Obtiene información del usuario actual (opcional)"""
    return {
        "id": "mx32-user",
        "role": "analyst",
        "name": "Usuario MX32"
    }

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Obtiene información del usuario actual (requerido)"""
    return {
        "id": "mx32-user",
        "role": "analyst", 
        "name": "Usuario MX32"
    }

@router.post("/chat")
async def deepchat_endpoint(
    request: DeepChatRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user_optional)
):
    """
    Endpoint principal para Deep Chat con RAG, LangChain y Cerebras
    """
    try:
        # Obtener agente avanzado
        agent = get_simple_rag_agent()
        
        # Extraer el último mensaje del usuario
        if not request.messages:
            return {"text": "No se recibió ningún mensaje"}
        
        last_message = request.messages[-1]
        if last_message.role != "user":
            return {"text": "El último mensaje debe ser del usuario"}
        
        # Crear contexto del usuario
        user_context = UserContext(
            user_id=request.user_id or current_user["id"],
            user_type=UserRole.ANALYST,
            current_page=request.context.get("current_page") if request.context else None,
            current_state=request.context.get("current_state") if request.context else None,
            current_parameter=request.context.get("current_parameter") if request.context else None,
            session_id=f"deepchat_{current_user['id']}_{int(datetime.now().timestamp())}",
            preferences=request.context.get("preferences") if request.context else None,
            rag_enabled=request.use_rag
        )
        
        # Convertir mensajes de Deep Chat al formato esperado
        messages = [{"role": msg.role, "content": msg.text} for msg in request.messages]
        
        # Obtener datos de la API si están disponibles
        api_data = request.context.get("api_data") if request.context else None
        
        # Procesar con el agente avanzado
        response = await agent.process_chat_with_rag(messages, user_context, api_data)
        
        # Log de la conversación
        background_tasks.add_task(
            log_conversation,
            user_context.session_id,
            last_message.text,
            response.response,
            user_context
        )
        
        # Procesar la respuesta para renderizar HTML correctamente
        processed_response = response.response
        
        # Verificar si contiene markdown (tablas, listas, etc.)
        has_markdown = any(marker in processed_response for marker in ['|', '**', '*', '#', '- ', '1. '])
        has_html = '<br>' in processed_response or '<br/>' in processed_response
        
        if has_markdown or has_html:
            # Si contiene markdown o HTML, usar HTML
            html_response = processed_response
            
            # Convertir tablas markdown a HTML
            if '|' in html_response:
                html_response = convert_markdown_table_to_html(html_response)
            
            # Convertir otros elementos markdown
            html_response = convert_markdown_to_html(html_response)
            
            return {
                "html": html_response,
                "structured_data": response.structured_data.dict() if response.structured_data else None,
                "rag_data": response.rag_data.dict() if response.rag_data else None,
                "suggested_actions": response.suggested_actions,
                "follow_up_questions": response.follow_up_questions,
                "confidence": response.confidence,
                "sources": response.sources,
                "model_used": response.model_used,
                "rag_enabled": response.rag_enabled,
                "session_id": response.session_id
            }
        else:
            # Si no contiene markdown, usar texto normal
            return {
                "text": processed_response,
                "structured_data": response.structured_data.dict() if response.structured_data else None,
                "rag_data": response.rag_data.dict() if response.rag_data else None,
                "suggested_actions": response.suggested_actions,
                "follow_up_questions": response.follow_up_questions,
                "confidence": response.confidence,
                "sources": response.sources,
                "model_used": response.model_used,
                "rag_enabled": response.rag_enabled,
                "session_id": response.session_id
            }
        
    except Exception as e:
        logger.error(f"Error en deepchat endpoint: {e}")
        return {"text": "Lo siento, ocurrió un error al procesar tu consulta."}

@router.post("/chat-advanced")
async def deepchat_advanced_endpoint(
    request: DeepChatRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user_optional)
):
    """
    Endpoint avanzado para Deep Chat con todas las características de LangChain
    """
    try:
        # Obtener agente avanzado
        agent = get_simple_rag_agent()
        
        # Extraer el último mensaje del usuario
        if not request.messages:
            return {"text": "No se recibió ningún mensaje"}
        
        last_message = request.messages[-1]
        if last_message.role != "user":
            return {"text": "El último mensaje debe ser del usuario"}
        
        # Crear contexto del usuario
        user_context = UserContext(
            user_id=request.user_id or current_user["id"],
            user_type=UserRole.ANALYST,
            current_page=request.context.get("current_page") if request.context else None,
            current_state=request.context.get("current_state") if request.context else None,
            current_parameter=request.context.get("current_parameter") if request.context else None,
            session_id=f"deepchat_advanced_{current_user['id']}_{int(datetime.now().timestamp())}",
            preferences=request.context.get("preferences") if request.context else None,
            rag_enabled=request.use_rag
        )
        
        # Convertir mensajes de Deep Chat al formato esperado
        messages = [{"role": msg.role, "content": msg.text} for msg in request.messages]
        
        # Obtener datos de la API si están disponibles
        api_data = request.context.get("api_data") if request.context else None
        
        # Procesar con el agente avanzado
        response = await agent.process_chat_with_rag(messages, user_context, api_data)
        
        # Log de la conversación
        background_tasks.add_task(
            log_conversation,
            user_context.session_id,
            last_message.text,
            response.response,
            user_context
        )
        
        # Procesar la respuesta para renderizar HTML correctamente
        processed_response = response.response
        
        # Verificar si contiene markdown (tablas, listas, etc.)
        has_markdown = any(marker in processed_response for marker in ['|', '**', '*', '#', '- ', '1. '])
        has_html = '<br>' in processed_response or '<br/>' in processed_response
        
        if has_markdown or has_html:
            # Si contiene markdown o HTML, usar HTML
            html_response = processed_response
            
            # Convertir tablas markdown a HTML
            if '|' in html_response:
                html_response = convert_markdown_table_to_html(html_response)
            
            # Convertir otros elementos markdown
            html_response = convert_markdown_to_html(html_response)
            
            return {
                "html": html_response,
                "structured_data": response.structured_data.dict() if response.structured_data else None,
                "comparison_data": response.comparison_data.dict() if response.comparison_data else None,
                "trend_data": response.trend_data.dict() if response.trend_data else None,
                "entity_data": response.entity_data.dict() if response.entity_data else None,
                "rag_data": response.rag_data.dict() if response.rag_data else None,
                "function_calls": [fc.dict() for fc in response.function_calls],
                "suggested_actions": response.suggested_actions,
                "follow_up_questions": response.follow_up_questions,
                "confidence": response.confidence,
                "sources": response.sources,
                "model_used": response.model_used,
                "rag_enabled": response.rag_enabled,
                "session_id": response.session_id,
                "memory_used": response.memory_used
            }
        else:
            # Si no contiene markdown, usar texto normal
            return {
                "text": processed_response,
                "structured_data": response.structured_data.dict() if response.structured_data else None,
                "comparison_data": response.comparison_data.dict() if response.comparison_data else None,
                "trend_data": response.trend_data.dict() if response.trend_data else None,
                "entity_data": response.entity_data.dict() if response.entity_data else None,
                "rag_data": response.rag_data.dict() if response.rag_data else None,
                "function_calls": [fc.dict() for fc in response.function_calls],
                "suggested_actions": response.suggested_actions,
                "follow_up_questions": response.follow_up_questions,
                "confidence": response.confidence,
                "sources": response.sources,
                "model_used": response.model_used,
                "rag_enabled": response.rag_enabled,
                "session_id": response.session_id,
                "memory_used": response.memory_used
            }
        
    except Exception as e:
        logger.error(f"Error en deepchat avanzado: {e}")
        return {"text": "Lo siento, ocurrió un error al procesar tu consulta."}

@router.get("/conversation/{session_id}")
async def get_conversation_history(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Obtiene el historial de conversación de una sesión
    """
    try:
        if session_id not in conversation_storage:
            return {"messages": [], "session_id": session_id}
        
        messages = conversation_storage[session_id]
        return {
            "session_id": session_id,
            "messages": [msg.dict() for msg in messages],
            "message_count": len(messages),
            "last_activity": messages[-1].timestamp if messages else None
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo historial: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo historial")

@router.delete("/conversation/{session_id}")
async def clear_conversation_history(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Limpia el historial de conversación de una sesión
    """
    try:
        if session_id in conversation_storage:
            del conversation_storage[session_id]
        
        return {"message": "Historial limpiado exitosamente", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"Error limpiando historial: {e}")
        raise HTTPException(status_code=500, detail="Error limpiando historial")

@router.get("/health")
async def deepchat_health():
    """
    Verifica el estado del sistema Deep Chat
    """
    try:
        agent = get_simple_rag_agent()
        return {
            "status": "healthy",
            "model": "cerebras-gpt-oss-120b",
            "rag_enabled": True,
            "langchain_enabled": True,
            "active_sessions": len(conversation_storage),
            "tools_available": len(agent.tools),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def log_conversation(
    session_id: str,
    user_message: str,
    assistant_response: str,
    user_context: UserContext
):
    """
    Registra la conversación en logs (en producción, guardar en base de datos)
    """
    try:
        logger.info(f"Deep Chat session {session_id}: User: {user_message[:100]}... | Assistant: {assistant_response[:100]}...")
        
        # Aquí podrías guardar en base de datos:
        # await save_conversation_to_db(session_id, user_message, assistant_response, user_context)
        
    except Exception as e:
        logger.error(f"Error logging conversation: {e}")

