"""
Script de prueba completo para el sistema RAG de MX32
Verifica todas las funcionalidades del sistema
"""

import asyncio
import json
import os
import sys
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_agent import rag_agent, consultar_estado_rag, obtener_estados_disponibles_rag
from rag_service import rag_service

class RAGTester:
    """Clase para probar el sistema RAG"""
    
    def __init__(self):
        self.results = []
        self.errors = []
    
    def log_result(self, test_name: str, success: bool, message: str = ""):
        """Registra resultado de una prueba"""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")
    
    async def test_firebase_connection(self):
        """Probar conexión a Firebase"""
        try:
            if rag_agent.db is None:
                self.log_result("Firebase Connection", False, "Cliente de Firebase no inicializado")
                return False
            
            # Probar consulta simple
            estados_ref = rag_agent.db.collection('states')
            estados_docs = list(estados_ref.limit(1).stream())
            
            if estados_docs:
                self.log_result("Firebase Connection", True, f"Conectado - {len(estados_docs)} documento(s) encontrado(s)")
                return True
            else:
                self.log_result("Firebase Connection", False, "No se encontraron documentos en la colección 'states'")
                return False
                
        except Exception as e:
            self.log_result("Firebase Connection", False, f"Error: {str(e)}")
            return False
    
    async def test_openai_connection(self):
        """Probar conexión a OpenAI"""
        try:
            if rag_agent.openai_client is None:
                self.log_result("OpenAI Connection", False, "Cliente de OpenAI no inicializado")
                return False
            
            # Probar llamada simple
            response = await rag_agent.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Responde 'OK' si puedes leer esto."}],
                max_tokens=10
            )
            
            if response.choices[0].message.content:
                self.log_result("OpenAI Connection", True, "Conectado y funcionando")
                return True
            else:
                self.log_result("OpenAI Connection", False, "Respuesta vacía de OpenAI")
                return False
                
        except Exception as e:
            self.log_result("OpenAI Connection", False, f"Error: {str(e)}")
            return False
    
    async def test_estados_disponibles(self):
        """Probar obtención de estados disponibles"""
        try:
            estados = await obtener_estados_disponibles_rag()
            
            if estados and len(estados) > 0:
                self.log_result("Estados Disponibles", True, f"{len(estados)} estados encontrados")
                return estados[0]  # Devolver primer estado para pruebas
            else:
                self.log_result("Estados Disponibles", False, "No se encontraron estados")
                return None
                
        except Exception as e:
            self.log_result("Estados Disponibles", False, f"Error: {str(e)}")
            return None
    
    async def test_obtener_datos_estado(self, estado_nombre: str):
        """Probar obtención de datos de un estado"""
        try:
            estado_data = await rag_agent.obtener_datos_estado_completo(estado_nombre)
            
            if estado_data:
                parametros_count = len(estado_data.parametros)
                parametros_con_datos = len([p for p in estado_data.parametros.values() if p.datos_apis])
                
                self.log_result("Obtener Datos Estado", True, 
                    f"Estado: {estado_data.nombre}, Parámetros: {parametros_count}, Con datos: {parametros_con_datos}")
                return estado_data
            else:
                self.log_result("Obtener Datos Estado", False, f"No se encontraron datos para {estado_nombre}")
                return None
                
        except Exception as e:
            self.log_result("Obtener Datos Estado", False, f"Error: {str(e)}")
            return None
    
    async def test_respuesta_rag(self, estado_nombre: str, pregunta: str):
        """Probar generación de respuesta RAG"""
        try:
            respuesta = await consultar_estado_rag(estado_nombre, pregunta)
            
            if respuesta.get('error'):
                self.log_result("Respuesta RAG", False, f"Error: {respuesta.get('error')}")
                return False
            
            respuesta_texto = respuesta.get('respuesta', '')
            if respuesta_texto and len(respuesta_texto) > 10:
                self.log_result("Respuesta RAG", True, f"Respuesta generada: {len(respuesta_texto)} caracteres")
                return True
            else:
                self.log_result("Respuesta RAG", False, "Respuesta vacía o muy corta")
                return False
                
        except Exception as e:
            self.log_result("Respuesta RAG", False, f"Error: {str(e)}")
            return False
    
    async def test_servicio_rag(self, estado_nombre: str):
        """Probar servicio RAG completo"""
        try:
            # Probar consulta básica
            consulta = await rag_service.consulta_estado_con_rag(estado_nombre)
            
            if consulta.get('estado') == estado_nombre:
                self.log_result("Servicio RAG", True, f"Consulta exitosa para {estado_nombre}")
                return True
            else:
                self.log_result("Servicio RAG", False, "Consulta falló")
                return False
                
        except Exception as e:
            self.log_result("Servicio RAG", False, f"Error: {str(e)}")
            return False
    
    async def test_consulta_ia(self, estado_nombre: str, pregunta: str):
        """Probar consulta de IA"""
        try:
            respuesta = await rag_service.consulta_ia_estado(estado_nombre, pregunta)
            
            if respuesta.get('respuesta'):
                self.log_result("Consulta IA", True, f"Respuesta IA generada: {len(respuesta.get('respuesta', ''))} caracteres")
                return True
            else:
                self.log_result("Consulta IA", False, "No se generó respuesta IA")
                return False
                
        except Exception as e:
            self.log_result("Consulta IA", False, f"Error: {str(e)}")
            return False
    
    async def run_all_tests(self):
        """Ejecutar todas las pruebas"""
        print("🧪 Iniciando pruebas del sistema RAG...")
        print("=" * 50)
        
        # 1. Probar conexiones
        firebase_ok = await self.test_firebase_connection()
        openai_ok = await self.test_openai_connection()
        
        if not firebase_ok:
            print("\n❌ Firebase no disponible. Deteniendo pruebas.")
            return
        
        # 2. Probar obtención de estados
        estado_ejemplo = await self.test_estados_disponibles()
        if not estado_ejemplo:
            print("\n❌ No se encontraron estados. Deteniendo pruebas.")
            return
        
        # 3. Probar obtención de datos
        estado_data = await self.test_obtener_datos_estado(estado_ejemplo)
        if not estado_data:
            print("\n❌ No se pudieron obtener datos del estado. Deteniendo pruebas.")
            return
        
        # 4. Probar respuestas RAG (solo si OpenAI está disponible)
        if openai_ok:
            pregunta_ejemplo = "¿Cuáles son las principales características de este estado?"
            await self.test_respuesta_rag(estado_ejemplo, pregunta_ejemplo)
            await self.test_consulta_ia(estado_ejemplo, pregunta_ejemplo)
        else:
            print("\n⚠️ OpenAI no disponible. Saltando pruebas de IA.")
        
        # 5. Probar servicio completo
        await self.test_servicio_rag(estado_ejemplo)
        
        # 6. Resumen de resultados
        self.print_summary()
    
    def print_summary(self):
        """Imprimir resumen de resultados"""
        print("\n" + "=" * 50)
        print("📊 RESUMEN DE PRUEBAS")
        print("=" * 50)
        
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r['success']])
        failed_tests = total_tests - successful_tests
        
        print(f"Total de pruebas: {total_tests}")
        print(f"Exitosas: {successful_tests} ✅")
        print(f"Fallidas: {failed_tests} ❌")
        print(f"Tasa de éxito: {(successful_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ PRUEBAS FALLIDAS:")
            for result in self.results:
                if not result['success']:
                    print(f"  - {result['test']}: {result['message']}")
        
        print("\n" + "=" * 50)
        
        # Guardar resultados en archivo
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print("📁 Resultados guardados en test_results.json")

async def main():
    """Función principal"""
    print("🤖 Sistema RAG MX32 - Pruebas Automatizadas")
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Verificar variables de entorno
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("⚠️ ADVERTENCIA: OPENAI_API_KEY no configurada")
        print("   Algunas pruebas de IA no se ejecutarán")
        print()
    
    # Ejecutar pruebas
    tester = RAGTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())

