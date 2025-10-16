"""
Templates de Few-Shot Prompting para el chatbot MX32 con RAG
Integrados con Cerebras y optimizados para análisis de estados mexicanos
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
                },
                {
                    "input": "¿Cómo está la situación de seguridad en Ciudad de México?",
                    "output": """## Análisis RAG de Seguridad en Ciudad de México

**Datos Utilizados:**
- Parámetros: Situación de Seguridad, Geografía y Logística
- APIs consultadas: 3 fuentes de datos de seguridad
- Texto de análisis: Análisis específico de CDMX

**Insights Clave:**
- Mejora del 12% en indicadores de seguridad en zonas centrales
- Persisten desafíos en periferias urbanas
- Implementación exitosa de programas de proximidad social

**Recomendaciones Específicas:**
1. Replicar estrategias exitosas en zonas periféricas
2. Fortalecer coordinación interinstitucional
3. Mejorar infraestructura de videovigilancia

**Nivel de Confianza:** Alto (basado en datos RAG actualizados)"""
                }
            ],
            
            "security_analysis": [
                {
                    "input": "¿Cuál es la situación de seguridad en Jalisco?",
                    "output": """## Análisis de Seguridad en Jalisco

**Situación Actual:**
Jalisco presenta indicadores mixtos en materia de seguridad. Según los datos más recientes, el estado muestra mejoras en algunos indicadores pero persisten desafíos importantes.

**Insights Clave:**
- Reducción del 15% en homicidios dolosos en los últimos 12 meses
- Aumento del 8% en delitos de alto impacto
- Mejora en percepción de seguridad en zonas metropolitanas

**Recomendaciones:**
1. Fortalecer la coordinación entre fuerzas de seguridad
2. Implementar programas de prevención social
3. Mejorar la infraestructura de videovigilancia

**Nivel de Confianza:** Alto (basado en datos oficiales actualizados)"""
                },
                {
                    "input": "¿Cómo está la seguridad en Ciudad de México comparada con otros estados?",
                    "output": """## Comparación de Seguridad: CDMX vs Otros Estados

**Ranking de Seguridad (2024):**
1. Yucatán - 8.5/10
2. Campeche - 8.2/10
3. Aguascalientes - 7.8/10
4. **CDMX - 6.5/10**
5. Jalisco - 6.2/10

**Análisis de CDMX:**
- Mejora significativa en zonas centrales
- Persisten desafíos en periferias
- Implementación exitosa de programas de proximidad social

**Recomendaciones Específicas:**
1. Replicar estrategias exitosas de Yucatán
2. Fortalecer presencia policial en zonas de alta incidencia
3. Mejorar coordinación interinstitucional"""
                }
            ],
            
            "economic_analysis": [
                {
                    "input": "¿Cuál es el estado de la economía en Nuevo León?",
                    "output": """## Análisis Económico de Nuevo León

**Indicadores Económicos Clave:**
- PIB per cápita: $45,000 USD (2do lugar nacional)
- Tasa de desempleo: 3.2% (por debajo del promedio nacional)
- Crecimiento económico: 4.1% anual

**Sectores Destacados:**
- Manufactura avanzada (automotriz, aeroespacial)
- Servicios financieros y tecnológicos
- Comercio internacional

**Oportunidades de Crecimiento:**
1. Desarrollo de clusters tecnológicos
2. Atracción de inversión extranjera directa
3. Fortalecimiento de la cadena de suministro

**Recomendaciones:**
- Invertir en educación técnica especializada
- Mejorar la conectividad logística
- Desarrollar parques industriales especializados"""
                },
                {
                    "input": "¿Qué estados tienen mejor desempeño económico?",
                    "output": """## Ranking de Desempeño Económico por Estados

**Top 5 Estados Económicamente:**
1. **CDMX** - Centro financiero y de servicios
2. **Nuevo León** - Manufactura y tecnología
3. **Jalisco** - Tecnología y agroindustria
4. **Querétaro** - Manufactura aeronáutica
5. **Yucatán** - Turismo y servicios

**Factores de Éxito Común:**
- Diversificación económica
- Inversión en capital humano
- Infraestructura de clase mundial
- Clima de negocios favorable

**Recomendaciones Estratégicas:**
1. Replicar modelos exitosos en otros estados
2. Fomentar la especialización regional
3. Mejorar la conectividad interregional"""
                }
            ],
            
            "infrastructure_analysis": [
                {
                    "input": "¿Cómo está la infraestructura de conectividad en Chiapas?",
                    "output": """## Análisis de Infraestructura en Chiapas

**Estado Actual de Conectividad:**
- Cobertura de internet: 65% (por debajo del promedio nacional)
- Carreteras federales: 78% en buen estado
- Aeropuertos: 2 internacionales operativos

**Desafíos Identificados:**
- Acceso limitado en zonas rurales
- Infraestructura de telecomunicaciones obsoleta
- Conectividad vial insuficiente

**Oportunidades:**
1. Proyecto del Tren Maya (impacto regional)
2. Desarrollo de corredores logísticos
3. Expansión de fibra óptica rural

**Recomendaciones Prioritarias:**
- Inversión en infraestructura digital
- Mejorar carreteras de acceso
- Desarrollar hubs logísticos regionales"""
                }
            ],
            
            "general_consultation": [
                {
                    "input": "¿Qué me puedes decir sobre el desarrollo en Quintana Roo?",
                    "output": """## Análisis Integral de Quintana Roo

**Perfil del Estado:**
Quintana Roo es líder en turismo nacional e internacional, con una economía basada principalmente en servicios turísticos y una creciente diversificación hacia otros sectores.

**Indicadores Clave:**
- Población: 1.8 millones de habitantes
- PIB per cápita: $28,500 USD
- Crecimiento anual: 3.8%
- Desempleo: 4.1%

**Fortalezas:**
- Destino turístico de clase mundial
- Infraestructura hotelera de primer nivel
- Conectividad aérea internacional
- Diversificación hacia servicios financieros

**Áreas de Oportunidad:**
1. Desarrollo de tecnología turística
2. Sostenibilidad ambiental
3. Diversificación económica
4. Capital humano especializado

**Recomendaciones Estratégicas:**
- Invertir en turismo sostenible
- Desarrollar clusters tecnológicos
- Fortalecer la educación superior
- Mejorar la conectividad terrestre"""
                }
            ]
        }
    
    def get_rag_template(self) -> FewShotPromptTemplate:
        """Template específico para análisis RAG"""
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="""Pregunta: {input}

Respuesta: {output}"""
        )
        
        return FewShotPromptTemplate(
            examples=self.examples["rag_analysis"],
            example_prompt=example_prompt,
            prefix="""Eres un experto en análisis de datos de México con capacidades RAG avanzadas. 
            Utiliza datos reales y actualizados de Firebase para proporcionar análisis precisos y contextuales.
            Siempre incluye fuentes de datos, insights específicos y recomendaciones accionables.
            
            Ejemplos de análisis RAG:""",
            suffix="""Pregunta: {input}

Respuesta:""",
            input_variables=["input"]
        )
    
    def get_security_template(self) -> FewShotPromptTemplate:
        """Template para análisis de seguridad"""
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="""Pregunta: {input}

Respuesta: {output}"""
        )
        
        return FewShotPromptTemplate(
            examples=self.examples["security_analysis"],
            example_prompt=example_prompt,
            prefix="""Eres un experto en análisis de seguridad pública en México. 
            Analiza la información de seguridad de los estados mexicanos y proporciona 
            insights detallados, recomendaciones específicas y comparaciones cuando sea relevante.
            
            Ejemplos de análisis de seguridad:""",
            suffix="""Pregunta: {input}

Respuesta:""",
            input_variables=["input"]
        )
    
    def get_economic_template(self) -> FewShotPromptTemplate:
        """Template para análisis económico"""
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="""Pregunta: {input}

Respuesta: {output}"""
        )
        
        return FewShotPromptTemplate(
            examples=self.examples["economic_analysis"],
            example_prompt=example_prompt,
            prefix="""Eres un experto en análisis económico regional de México. 
            Proporciona análisis detallados sobre indicadores económicos, 
            comparaciones entre estados y recomendaciones estratégicas.
            
            Ejemplos de análisis económico:""",
            suffix="""Pregunta: {input}

Respuesta:""",
            input_variables=["input"]
        )
    
    def get_infrastructure_template(self) -> FewShotPromptTemplate:
        """Template para análisis de infraestructura"""
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="""Pregunta: {input}

Respuesta: {output}"""
        )
        
        return FewShotPromptTemplate(
            examples=self.examples["infrastructure_analysis"],
            example_prompt=example_prompt,
            prefix="""Eres un experto en infraestructura y conectividad en México. 
            Analiza el estado de la infraestructura, identifica brechas y 
            proporciona recomendaciones para el desarrollo.
            
            Ejemplos de análisis de infraestructura:""",
            suffix="""Pregunta: {input}

Respuesta:""",
            input_variables=["input"]
        )
    
    def get_general_template(self) -> FewShotPromptTemplate:
        """Template para consultas generales"""
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="""Pregunta: {input}

Respuesta: {output}"""
        )
        
        return FewShotPromptTemplate(
            examples=self.examples["general_consultation"],
            example_prompt=example_prompt,
            prefix="""Eres un asistente especializado en análisis de datos de México (MX32) con capacidades RAG. 
            Proporciona análisis integrales sobre estados, parámetros y métricas, 
            con insights detallados y recomendaciones específicas basadas en datos reales.
            
            Ejemplos de análisis general:""",
            suffix="""Pregunta: {input}

Respuesta:""",
            input_variables=["input"]
        )
    
    def get_template_by_intent(self, user_message: str) -> FewShotPromptTemplate:
        """Determina el template apropiado basado en la intención del usuario"""
        message_lower = user_message.lower()
        
        if any(keyword in message_lower for keyword in ["rag", "datos", "firebase", "análisis"]):
            return self.get_rag_template()
        elif any(keyword in message_lower for keyword in ["seguridad", "crimen", "violencia", "policia"]):
            return self.get_security_template()
        elif any(keyword in message_lower for keyword in ["economia", "economía", "pib", "empleo", "inversion"]):
            return self.get_economic_template()
        elif any(keyword in message_lower for keyword in ["infraestructura", "conectividad", "carreteras", "internet"]):
            return self.get_infrastructure_template()
        else:
            return self.get_general_template()

