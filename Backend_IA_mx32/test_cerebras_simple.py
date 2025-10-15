"""
Test simple de Cerebras API
"""

import asyncio
import aiohttp
import json

async def test_cerebras_simple():
    """Test simple de conexión con Cerebras"""
    
    api_key = "csk-9356538ykt4yvhkfnccdcpft69869kd5njyvw3vf2y5x48f4"
    base_url = "https://api.cerebras.ai/v1"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "Eres un asistente útil. Responde brevemente en español."
            },
            {
                "role": "user", 
                "content": "Hola, ¿cómo estás?"
            }
        ],
        "model": "gpt-oss-120b",
        "temperature": 0.7,
        "max_completion_tokens": 100,
        "stream": False,
        "top_p": 1.0,
        "reasoning_effort": "high"
    }
    
    try:
        print("🧪 Probando conexión con Cerebras API...")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                
                print(f"Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    print("✅ Conexión exitosa!")
                    print(f"Respuesta: {data['choices'][0]['message']['content']}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Error: {response.status}")
                    print(f"Detalles: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_cerebras_simple())
