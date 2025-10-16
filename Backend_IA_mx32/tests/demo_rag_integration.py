"""
Demo de integración RAG con el sistema MX32 existente
Muestra cómo el RAG se integra con los endpoints de /estados/
"""

import asyncio
import json
import requests
from datetime import datetime

class MX32RAGDemo:
    """Demo de integración RAG con MX32"""
    
    def __init__(self, backend_url="http://localhost:8000", rag_url="http://localhost:8001"):
        self.backend_url = backend_url
        self.rag_url = rag_url
    
    def print_header(self, title: str):
        """Imprimir encabezado"""
        print("\n" + "=" * 60)
        print(f"🔧 {title}")
        print("=" * 60)
    
    def print_step(self, step: str, description: str):
        """Imprimir paso"""
        print(f"\n📋 Paso {step}: {description}")
        print("-" * 40)
    
    async def demo_backend_existing(self):
        """Demo del backend existente"""
        self.print_header("BACKEND EXISTENTE - SIN RAG")
        
        self.print_step("1", "Obtener lista de estados")
        try:
            response = requests.get(f"{self.backend_url}/api/consultas/estados")
            if response.status_code == 200:
                estados = response.json()
                print(f"✅ Estados obtenidos: {len(estados.get('estados', []))}")
                if estados.get('estados'):
                    estado_ejemplo = estados['estados'][0]['states_name']
                    print(f"   Ejemplo: {estado_ejemplo}")
                    return estado_ejemplo
            else:
                print(f"❌ Error: {response.status_code}")
        except Exception as e:
            print(f"❌ Error conectando al backend: {e}")
        
        return None
    
    async def demo_consulta_existing(self, estado: str):
        """Demo de consulta existente"""
        self.print_step("2", f"Consulta existente para {estado}")
        
        consulta_data = {
            "estado": estado,
            "parametros": ["Situación de Seguridad", "Geografía y Logística"]
        }
        
        try:
            response = requests.post(
                f"{self.backend_url}/api/consultas/consulta-dinamica",
                json=consulta_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                datos = response.json()
                print(f"✅ Consulta exitosa")
                print(f"   Estado: {datos.get('estado')}")
                print(f"   Parámetros: {len(datos.get('datos_por_parametro', {}))}")
                
                # Mostrar estructura de datos
                for param, data in datos.get('datos_por_parametro', {}).items():
                    print(f"   - {param}: {len(data.get('datos_apis', []))} APIs")
                
                return datos
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error en consulta: {e}")
        
        return None
    
    async def demo_rag_system(self, estado: str):
        """Demo del sistema RAG"""
        self.print_header("SISTEMA RAG - CON IA")
        
        self.print_step("1", "Verificar estado del RAG")
        try:
            response = requests.get(f"{self.rag_url}/api/rag/health-rag")
            if response.status_code == 200:
                health = response.json()
                print(f"✅ RAG disponible: {health.get('rag_habilitado')}")
                print(f"   Firebase: {health.get('servicios', {}).get('firebase')}")
                print(f"   OpenAI: {health.get('servicios', {}).get('openai')}")
            else:
                print(f"❌ RAG no disponible: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error conectando al RAG: {e}")
            return False
        
        self.print_step("2", f"Consulta RAG para {estado}")
        consulta_rag_data = {
            "estado": estado,
            "parametros": ["Situación de Seguridad", "Geografía y Logística"],
            "pregunta": "¿Cuáles son las principales oportunidades de inversión en este estado?"
        }
        
        try:
            response = requests.post(
                f"{self.rag_url}/api/rag/consulta-estado-rag",
                json=consulta_rag_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                datos_rag = response.json()
                print(f"✅ Consulta RAG exitosa")
                print(f"   Estado: {datos_rag.get('estado')}")
                print(f"   RAG habilitado: {datos_rag.get('rag_habilitado')}")
                
                # Mostrar respuesta de IA si existe
                if datos_rag.get('respuesta_ia'):
                    respuesta_ia = datos_rag['respuesta_ia']
                    print(f"   Pregunta IA: {respuesta_ia.get('pregunta')}")
                    print(f"   Respuesta IA: {respuesta_ia.get('respuesta', '')[:200]}...")
                
                return datos_rag
            else:
                print(f"❌ Error RAG: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error en consulta RAG: {e}")
        
        return None
    
    async def demo_consulta_ia_directa(self, estado: str):
        """Demo de consulta de IA directa"""
        self.print_step("3", f"Consulta de IA directa para {estado}")
        
        pregunta = "¿Cómo está la situación de seguridad en este estado y qué oportunidades de inversión ofrece?"
        
        consulta_ia_data = {
            "estado": estado,
            "pregunta": pregunta
        }
        
        try:
            response = requests.post(
                f"{self.rag_url}/api/rag/consulta-ia",
                json=consulta_ia_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                respuesta_ia = response.json()
                print(f"✅ Respuesta IA generada")
                print(f"   Pregunta: {respuesta_ia.get('pregunta')}")
                print(f"   Estado: {respuesta_ia.get('estado')}")
                print(f"   Parámetros utilizados: {len(respuesta_ia.get('parametros_disponibles', []))}")
                print(f"   Respuesta: {respuesta_ia.get('respuesta', '')[:300]}...")
                
                return respuesta_ia
            else:
                print(f"❌ Error IA: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error en consulta IA: {e}")
        
        return None
    
    async def demo_comparacion(self, datos_existing, datos_rag):
        """Demo de comparación entre sistemas"""
        self.print_header("COMPARACIÓN: SISTEMA EXISTENTE vs RAG")
        
        print("\n📊 SISTEMA EXISTENTE:")
        if datos_existing:
            print(f"   ✅ Datos estructurados disponibles")
            print(f"   📈 Parámetros: {len(datos_existing.get('datos_por_parametro', {}))}")
            print(f"   🔗 APIs consultadas: {sum(len(data.get('datos_apis', [])) for data in datos_existing.get('datos_por_parametro', {}).values())}")
            print(f"   📝 Textos de análisis: {sum(1 for data in datos_existing.get('datos_por_parametro', {}).values() if data.get('texto_analisis'))}")
        else:
            print("   ❌ No hay datos disponibles")
        
        print("\n🤖 SISTEMA RAG:")
        if datos_rag:
            print(f"   ✅ Datos estructurados + IA")
            print(f"   📈 Parámetros: {len(datos_rag.get('datos_por_parametro', {}))}")
            print(f"   🔗 APIs consultadas: {sum(len(data.get('datos_apis', [])) for data in datos_rag.get('datos_por_parametro', {}).values())}")
            print(f"   📝 Textos de análisis: {sum(1 for data in datos_rag.get('datos_por_parametro', {}).values() if data.get('texto_analisis'))}")
            print(f"   🤖 Respuesta IA: {'Sí' if datos_rag.get('respuesta_ia') else 'No'}")
        else:
            print("   ❌ No hay datos disponibles")
        
        print("\n🎯 VENTAJAS DEL RAG:")
        print("   • Respuestas contextuales en lenguaje natural")
        print("   • Análisis inteligente de los datos")
        print("   • Integración perfecta con sistema existente")
        print("   • Escalable y personalizable")
    
    async def run_demo(self):
        """Ejecutar demo completo"""
        print("🚀 DEMO DE INTEGRACIÓN RAG CON MX32")
        print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Demo del backend existente
        estado = await self.demo_backend_existing()
        if not estado:
            print("\n❌ No se pudo obtener estado del backend. Deteniendo demo.")
            return
        
        # 2. Consulta existente
        datos_existing = await self.demo_consulta_existing(estado)
        
        # 3. Demo del sistema RAG
        datos_rag = await self.demo_rag_system(estado)
        
        # 4. Consulta de IA directa
        if datos_rag:
            await self.demo_consulta_ia_directa(estado)
        
        # 5. Comparación
        await self.demo_comparacion(datos_existing, datos_rag)
        
        # 6. Resumen final
        self.print_header("RESUMEN FINAL")
        print("\n🎉 Demo completado exitosamente!")
        print("\n📋 Próximos pasos para implementar:")
        print("   1. Ejecutar: python integrate_rag_backend.py")
        print("   2. Instalar dependencias: pip install -r requirements.txt")
        print("   3. Configurar OPENAI_API_KEY")
        print("   4. Iniciar backend: uvicorn app.main:app --reload")
        print("   5. Probar endpoints RAG en /api/rag/")
        
        print(f"\n⏰ Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

async def main():
    """Función principal"""
    demo = MX32RAGDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())

