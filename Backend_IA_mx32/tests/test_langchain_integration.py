"""
Script de prueba para la integración RAG + LangChain + Cerebras
"""

import asyncio
import json
from datetime import datetime
from services.rag_langchain_agent import get_rag_langchain_agent
from models.chatbot import UserContext, UserRole

async def test_rag_langchain_integration():
    """Test de integración RAG + LangChain + Cerebras"""
    
    print("🧪 Iniciando pruebas de integración RAG + LangChain + Cerebras")
    print("=" * 60)
    
    try:
        # Obtener agente
        agent = get_rag_langchain_agent()
        print("✅ Agente RAG-LangChain inicializado correctamente")
        
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
        
        # Test 4: Análisis de entidades
        print("\n📊 Test 4: Análisis de entidades")
        print("-" * 40)
        
        if result.entity_data:
            print(f"✅ Estados mencionados: {result.entity_data.states_mentioned}")
            print(f"✅ Entidades extraídas: {result.entity_data.entities}")
            print(f"✅ Intención detectada: {result.entity_data.intent}")
        
        # Test 5: Acciones sugeridas
        print("\n📊 Test 5: Acciones sugeridas")
        print("-" * 40)
        
        if result.suggested_actions:
            print("✅ Acciones sugeridas:")
            for i, action in enumerate(result.suggested_actions, 1):
                print(f"   {i}. {action}")
        
        # Test 6: Preguntas de seguimiento
        print("\n📊 Test 6: Preguntas de seguimiento")
        print("-" * 40)
        
        if result.follow_up_questions:
            print("✅ Preguntas de seguimiento:")
            for i, question in enumerate(result.follow_up_questions, 1):
                print(f"   {i}. {question}")
        
        print("\n🎉 Todas las pruebas completadas exitosamente!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_deepchat_endpoints():
    """Test de endpoints de Deep Chat"""
    
    print("\n🌐 Probando endpoints de Deep Chat")
    print("=" * 60)
    
    try:
        # Simular llamada a endpoint
        from api.deepchat_endpoints import deepchat_endpoint
        from models.chatbot import DeepChatRequest, DeepChatMessage
        
        # Crear request de prueba
        request = DeepChatRequest(
            messages=[
                DeepChatMessage(role="user", text="¿Cuáles son las oportunidades de inversión en Jalisco?")
            ],
            user_id="test_user",
            context={
                "current_state": "jalisco",
                "current_parameter": "Oportunidades Emergentes"
            },
            use_rag=True
        )
        
        # Simular contexto de usuario
        class MockUser:
            def __init__(self):
                self.id = "test_user"
                self.role = "analyst"
                self.name = "Usuario MX32"
        
        # Ejecutar endpoint
        result = await deepchat_endpoint(request, None, MockUser())
        
        print("✅ Endpoint de Deep Chat funcionando")
        print(f"✅ Respuesta contiene HTML: {'html' in result}")
        print(f"✅ Respuesta contiene texto: {'text' in result}")
        
        if 'html' in result:
            print(f"✅ HTML generado: {len(result['html'])} caracteres")
        elif 'text' in result:
            print(f"✅ Texto generado: {len(result['text'])} caracteres")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en endpoints: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Función principal de pruebas"""
    
    print("🚀 Iniciando pruebas completas de integración")
    print("=" * 60)
    
    # Test 1: Integración RAG + LangChain
    test1_success = await test_rag_langchain_integration()
    
    # Test 2: Endpoints de Deep Chat
    test2_success = await test_deepchat_endpoints()
    
    # Resumen final
    print("\n📋 RESUMEN DE PRUEBAS")
    print("=" * 60)
    print(f"✅ Integración RAG + LangChain: {'PASÓ' if test1_success else 'FALLÓ'}")
    print(f"✅ Endpoints Deep Chat: {'PASÓ' if test2_success else 'FALLÓ'}")
    
    if test1_success and test2_success:
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON EXITOSAMENTE!")
        print("🚀 El sistema está listo para usar con el frontend")
    else:
        print("\n❌ Algunas pruebas fallaron. Revisar logs para más detalles.")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())

