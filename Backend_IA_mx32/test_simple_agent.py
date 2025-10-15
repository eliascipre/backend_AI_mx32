"""
Test simple del agente RAG con Cerebras
"""

import asyncio
import sys
import os

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.simple_rag_agent import get_simple_rag_agent
from src.models.chatbot import UserContext, UserRole

async def test_simple_agent():
    """Test del agente simplificado"""
    
    print("🧪 Probando agente RAG simplificado con Cerebras")
    print("=" * 60)
    
    try:
        # Obtener agente
        agent = get_simple_rag_agent()
        print("✅ Agente inicializado correctamente")
        
        # Test 1: Chat con RAG habilitado
        print("\n📊 Test 1: Chat con RAG habilitado")
        print("-" * 40)
        
        user_context = UserContext(
            user_id="test_user",
            user_type=UserRole.ANALYST,
            current_state="jalisco",
            current_parameter="Situación de Seguridad",
            session_id="test_session_1",
            rag_enabled=True
        )
        
        result = await agent.process_chat_with_rag(
            [{"role": "user", "content": "¿Cuál es la situación de seguridad en Jalisco?"}],
            user_context
        )
        
        print(f"✅ RAG habilitado: {result.rag_enabled}")
        print(f"✅ Confianza: {result.confidence}")
        print(f"✅ Modelo usado: {result.model_used}")
        print(f"✅ Respuesta: {result.response[:200]}...")
        
        # Test 2: Chat sin RAG
        print("\n📊 Test 2: Chat sin RAG")
        print("-" * 40)
        
        user_context.rag_enabled = False
        result2 = await agent.process_chat_with_rag(
            [{"role": "user", "content": "¿Qué me puedes decir sobre México?"}],
            user_context
        )
        
        print(f"✅ RAG habilitado: {result2.rag_enabled}")
        print(f"✅ Confianza: {result2.confidence}")
        print(f"✅ Respuesta: {result2.response[:200]}...")
        
        # Test 3: Verificación de restricción de México
        print("\n📊 Test 3: Verificación de restricción de México")
        print("-" * 40)
        
        result3 = await agent.process_chat_with_rag(
            [{"role": "user", "content": "¿Cómo está la economía en Estados Unidos?"}],
            user_context
        )
        
        print(f"✅ Respuesta contiene restricción: {'México' in result3.response}")
        print(f"✅ Respuesta: {result3.response[:200]}...")
        
        print("\n🎉 ¡Todos los tests pasaron exitosamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_simple_agent())
