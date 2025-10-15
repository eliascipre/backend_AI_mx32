"""
Cliente para Cerebras Cloud SDK
Adaptado para el sistema RAG-LangChain de MX32
"""

import os
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CerebrasClient:
    """Cliente para interactuar con Cerebras Cloud API"""
    
    def __init__(self, api_key: Optional[str] = None):
        from src.core.config import settings
        self.api_key = api_key or settings.cerebras_api_key
        self.base_url = settings.cerebras_base_url
        self.model = settings.cerebras_model
        
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False,
        top_p: float = 1.0,
        reasoning_effort: str = "high"
    ) -> Dict[str, Any]:
        """
        Realiza una llamada de chat completion a Cerebras
        
        Args:
            messages: Lista de mensajes en formato OpenAI
            temperature: Temperatura para la generación
            max_tokens: Máximo número de tokens
            stream: Si usar streaming
            top_p: Parámetro top_p
            reasoning_effort: Nivel de esfuerzo de razonamiento
            
        Returns:
            Respuesta de la API de Cerebras
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": messages,
                "model": self.model,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
                "stream": stream,
                "top_p": top_p,
                "reasoning_effort": reasoning_effort
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        if stream:
                            return await self._handle_stream_response(response)
                        else:
                            return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Error en Cerebras API: {response.status} - {error_text}")
                        return {
                            "error": f"API Error: {response.status}",
                            "details": error_text
                        }
                        
        except Exception as e:
            logger.error(f"Error en llamada a Cerebras: {e}")
            return {
                "error": "Connection Error",
                "details": str(e)
            }
    
    async def _handle_stream_response(self, response) -> AsyncGenerator[str, None]:
        """Maneja respuestas en streaming de Cerebras"""
        async for line in response.content:
            if line:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith('data: '):
                    try:
                        data = json.loads(line_str[6:])
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
    
    async def generate_response(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Genera una respuesta simple usando Cerebras
        
        Args:
            system_prompt: Prompt del sistema
            user_message: Mensaje del usuario
            temperature: Temperatura para la generación
            max_tokens: Máximo número de tokens
            
        Returns:
            Respuesta generada
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = await self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        
        if "error" in response:
            return f"Error: {response['error']}"
        
        try:
            return response['choices'][0]['message']['content']
        except (KeyError, IndexError):
            return "Error: No se pudo obtener la respuesta"
    
    async def generate_streaming_response(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        """
        Genera una respuesta en streaming usando Cerebras
        
        Args:
            system_prompt: Prompt del sistema
            user_message: Mensaje del usuario
            temperature: Temperatura para la generación
            max_tokens: Máximo número de tokens
            
        Yields:
            Chunks de la respuesta generada
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": messages,
                "model": self.model,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
                "stream": True,
                "top_p": 1.0,
                "reasoning_effort": "high"
            }
            
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    async for chunk in self._handle_stream_response(response):
                        yield chunk
                else:
                    error_text = await response.text()
                    yield f"Error: {response.status} - {error_text}"

# Instancia global del cliente
_cerebras_client = None

def get_cerebras_client() -> CerebrasClient:
    """Obtiene la instancia global del cliente de Cerebras"""
    global _cerebras_client
    if _cerebras_client is None:
        _cerebras_client = CerebrasClient()
    return _cerebras_client

