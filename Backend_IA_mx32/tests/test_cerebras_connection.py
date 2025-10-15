"""
Test de conexión con Cerebras API
"""

import asyncio
import sys
import os

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.services.cerebras_client import get_cerebras_client

async def test_cerebras_connection():
    """Test de conexión con Cerebras"""
    
    print("🧪 Probando conexión con Cerebras API")
    print("=" * 50)
    
    try:
        # Obtener cliente
        client = get_cerebras_client()
        print(f"✅ Cliente inicializado")
        print(f"   API Key: {client.api_key[:20]}...")
        print(f"   Modelo: {client.model}")
        print(f"   Base URL: {client.base_url}")
        
        # Test de conexión simple
        print("\n📡 Probando conexión...")
        
        response = await client.generate_response(
            system_prompt="Eres un asistente útil. Responde brevemente.",
            user_message="Hola, ¿cómo estás?",
            temperature=0.7,
            max_tokens=100
        )
        
        print(f"✅ Respuesta recibida: {response[:100]}...")
        
        # Test de streaming
        print("\n🌊 Probando streaming...")
        
        async for chunk in client.generate_streaming_response(
            system_prompt="Eres un asistente útil. Responde brevemente.",
            user_message="Cuéntame sobre México en una frase.",
            temperature=0.7,
            max_tokens=100
        ):
            print(chunk, end="", flush=True)
        
        print("\n\n✅ Streaming funcionando correctamente")
        
        print("\n🎉 ¡Conexión con Cerebras exitosa!")
        return True
        
    except Exception as e:
        print(f"❌ Error en la conexión: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Función principal"""
    success = await test_cerebras_connection()
    
    if success:
        print("\n✅ Todos los tests pasaron. El sistema está listo.")
    else:
        print("\n❌ Algunos tests fallaron. Revisar configuración.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
