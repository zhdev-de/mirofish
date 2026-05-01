"""
本体生成服务
接口1：分析文本内容，生成适合社会模拟的实体和关系类型定义
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient
from ..utils.locale import get_language_instruction

logger = logging.getLogger(__name__)


def _to_pascal_case(name: str) -> str:
    """将任意格式的名称转换为 PascalCase（如 'works_for' -> 'WorksFor', 'person' -> 'Person'）"""
    # 按非字母数字字符分割
    parts = re.split(r'[^a-zA-Z0-9]+', name)
    # 再按 camelCase 边界分割（如 'camelCase' -> ['camel', 'Case']）
    words = []
    for part in parts:
        words.extend(re.sub(r'([a-z])([A-Z])', r'\1_\2', part).split('_'))
    # 每个词首字母大写，过滤空串
    result = ''.join(word.capitalize() for word in words if word)
    return result if result else 'Unknown'


# 本体生成的系统提示词
ONTOLOGY_SYSTEM_PROMPT = """Du bist Expertin/Experte für die Gestaltung von Knowledge-Graph-Ontologien. Deine Aufgabe ist es, den gegebenen Text und die Simulations-Anforderung zu analysieren und passende Entitäts- und Beziehungstypen für eine **Social-Media-Meinungs-Simulation** zu entwerfen.

**Wichtig: Du musst gültiges JSON ausgeben — sonst nichts.**

## Hintergrund

Wir bauen ein **Social-Media-Meinungs-Simulationssystem**. In diesem System gilt:
- Jede Entität ist ein „Account" oder „Subjekt", das auf Social Media Stimme erheben, interagieren und Informationen verbreiten kann
- Entitäten beeinflussen sich gegenseitig, teilen Beiträge, kommentieren und antworten
- Wir simulieren die Reaktionen verschiedener Beteiligter und Informations-Verbreitungspfade in Meinungs-Ereignissen

Daher müssen **Entitäten reale, auf Social Media handlungsfähige Subjekte** sein:

**Erlaubt sind**:
- Konkrete Einzelpersonen (Persönlichkeiten, Betroffene, Meinungsführer, Fachleute, Privatpersonen)
- Unternehmen (inkl. ihrer offiziellen Accounts)
- Organisationen (Hochschulen, Verbände, NGOs, Gewerkschaften usw.)
- Regierungsbehörden, Aufsichtsbehörden
- Medienorganisationen (Zeitungen, TV-Sender, Self-Media, Webseiten)
- Social-Media-Plattformen selbst
- Repräsentanten spezifischer Gruppen (Alumni-Verein, Fan-Gemeinde, Aktivisten-Gruppe usw.)

**Nicht erlaubt sind**:
- Abstrakte Konzepte (z. B. „öffentliche Meinung", „Stimmung", „Trend")
- Themen (z. B. „Wissenschafts-Integrität", „Bildungsreform")
- Standpunkte/Haltungen (z. B. „Befürworter", „Gegner")

## Ausgabe-Format

Gib JSON in folgender Struktur aus:

```json
{
    "entity_types": [
        {
            "name": "Name des Entitätstyps (Englisch, PascalCase)",
            "description": "Kurze Beschreibung (Englisch, max. 100 Zeichen)",
            "attributes": [
                {
                    "name": "Attribut-Name (Englisch, snake_case)",
                    "type": "text",
                    "description": "Beschreibung des Attributs"
                }
            ],
            "examples": ["Beispiel-Entität 1", "Beispiel-Entität 2"]
        }
    ],
    "edge_types": [
        {
            "name": "Name des Beziehungstyps (Englisch, UPPER_SNAKE_CASE)",
            "description": "Kurze Beschreibung (Englisch, max. 100 Zeichen)",
            "source_targets": [
                {"source": "Quell-Entitätstyp", "target": "Ziel-Entitätstyp"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Kurze Analyse des Text-Inhalts"
}
```

## Design-Leitlinien (sehr wichtig!)

### 1. Entitätstyp-Design — verbindlich

**Anzahl: genau 10 Entitätstypen.**

**Hierarchie (konkrete Typen UND Auffang-Typen müssen enthalten sein):**

Deine 10 Entitätstypen müssen folgende Hierarchie abbilden:

A. **Auffang-Typen (Pflicht, an den letzten 2 Plätzen der Liste)**:
   - `Person`: Auffang-Typ für jede natürliche Person. Wenn ein Mensch zu keinem konkreteren Personen-Typ passt, gehört er hierher.
   - `Organization`: Auffang-Typ für jede Organisation. Wenn eine Organisation zu keinem konkreteren Organisations-Typ passt, gehört sie hierher.

B. **Konkrete Typen (8, basierend auf dem Text-Inhalt)**:
   - Konkretere Typen für die wichtigsten Rollen im Text
   - Beispiel: Bei einem akademischen Ereignis evtl. `Student`, `Professor`, `University`
   - Beispiel: Bei einem Wirtschafts-Ereignis evtl. `Company`, `CEO`, `Employee`

**Warum Auffang-Typen nötig sind**:
- Im Text tauchen vielfältige Personen auf, z. B. „Schullehrer", „Passant", „ein Internet-User"
- Ohne passenden konkreten Typ werden sie als `Person` klassifiziert
- Analog werden kleine oder temporäre Gruppen unter `Organization` gefasst

**Design-Prinzipien für konkrete Typen**:
- Identifiziere die im Text häufig vorkommenden oder zentralen Rollen
- Jeder konkrete Typ braucht klare Abgrenzung — keine Überschneidungen
- description muss klar machen, wie dieser Typ sich vom Auffang-Typ unterscheidet

### 2. Beziehungstyp-Design

- Anzahl: 6–10
- Beziehungen sollen reale Social-Media-Interaktionen abbilden
- source_targets der Beziehungen müssen die definierten Entitätstypen abdecken

### 3. Attribut-Design

- Pro Entitätstyp 1–3 Schlüssel-Attribute
- **Achtung**: Attribut-Namen dürfen NICHT `name`, `uuid`, `group_id`, `created_at`, `summary` heißen (System-reservierte Wörter)
- Empfohlen: `full_name`, `title`, `role`, `position`, `location`, `description` etc.

## Referenz-Entitätstypen

**Personen (konkret)**:
- Student: Studierende/r
- Professor: Professor/Wissenschaftler
- Journalist: Journalist
- Celebrity: Promi/Influencer
- Executive: Führungskraft
- Official: Regierungsbeamte/r
- Lawyer: Anwalt/Anwältin
- Doctor: Arzt/Ärztin

**Personen (Auffang)**:
- Person: jede natürliche Person (wenn kein konkreter Typ passt)

**Organisationen (konkret)**:
- University: Hochschule
- Company: Unternehmen
- GovernmentAgency: Regierungsbehörde
- MediaOutlet: Medienorganisation
- Hospital: Krankenhaus
- School: Schule
- NGO: Nichtregierungsorganisation

**Organisationen (Auffang)**:
- Organization: jede Organisation (wenn kein konkreter Typ passt)

## Referenz-Beziehungstypen

- WORKS_FOR: arbeitet bei
- STUDIES_AT: studiert an
- AFFILIATED_WITH: zugehörig zu
- REPRESENTS: repräsentiert
- REGULATES: reguliert
- REPORTS_ON: berichtet über
- COMMENTS_ON: kommentiert
- RESPONDS_TO: antwortet auf
- SUPPORTS: unterstützt
- OPPOSES: lehnt ab
- COLLABORATES_WITH: kooperiert mit
- COMPETES_WITH: konkurriert mit
"""


class OntologyGenerator:
    """
    本体生成器
    分析文本内容，生成实体和关系类型定义
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
    
    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成本体定义
        
        Args:
            document_texts: 文档文本列表
            simulation_requirement: 模拟需求描述
            additional_context: 额外上下文
            
        Returns:
            本体定义（entity_types, edge_types等）
        """
        # 构建用户消息
        user_message = self._build_user_message(
            document_texts, 
            simulation_requirement,
            additional_context
        )
        
        lang_instruction = get_language_instruction()
        system_prompt = f"{ONTOLOGY_SYSTEM_PROMPT}\n\n{lang_instruction}\nIMPORTANT: Entity type names MUST be in English PascalCase (e.g., 'PersonEntity', 'MediaOrganization'). Relationship type names MUST be in English UPPER_SNAKE_CASE (e.g., 'WORKS_FOR'). Attribute names MUST be in English snake_case. Only description fields and analysis_summary should use the specified language above."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # 调用LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )
        
        # 验证和后处理
        result = self._validate_and_process(result)
        
        return result
    
    # 传给 LLM 的文本最大长度（5万字）
    MAX_TEXT_LENGTH_FOR_LLM = 50000
    
    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """构建用户消息"""
        
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)

        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(Original umfasst {original_length} Zeichen; die ersten {self.MAX_TEXT_LENGTH_FOR_LLM} wurden für die Ontologie-Analyse verwendet)..."

        message = f"""## Simulations-Anforderung

{simulation_requirement}

## Dokument-Inhalt

{combined_text}
"""

        if additional_context:
            message += f"""
## Zusätzliche Hinweise

{additional_context}
"""

        message += """
Entwirf auf Basis der obigen Inhalte passende Entitäts- und Beziehungstypen für die Social-Media-Meinungs-Simulation.

**Verbindliche Regeln**:
1. Es müssen genau 10 Entitätstypen ausgegeben werden
2. Die letzten 2 müssen die Auffang-Typen sein: Person (Auffang für Personen) und Organization (Auffang für Organisationen)
3. Die ersten 8 sind konkrete Typen, abgeleitet aus dem Text-Inhalt
4. Alle Entitätstypen müssen reale, in der Öffentlichkeit handlungsfähige Subjekte sein, keine abstrakten Konzepte
5. Attribut-Namen dürfen nicht name, uuid, group_id usw. (System-reserviert) verwenden — stattdessen full_name, org_name etc.
"""
        
        return message
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证和后处理结果"""
        
        # 确保必要字段存在
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        
        # 验证实体类型
        # 记录原始名称到 PascalCase 的映射，用于后续修正 edge 的 source_targets 引用
        entity_name_map = {}
        for entity in result["entity_types"]:
            # 强制将 entity name 转为 PascalCase（Zep API 要求）
            if "name" in entity:
                original_name = entity["name"]
                entity["name"] = _to_pascal_case(original_name)
                if entity["name"] != original_name:
                    logger.warning(f"Entity type name '{original_name}' auto-converted to '{entity['name']}'")
                entity_name_map[original_name] = entity["name"]
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # 确保description不超过100字符
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
        
        # 验证关系类型
        for edge in result["edge_types"]:
            # 强制将 edge name 转为 SCREAMING_SNAKE_CASE（Zep API 要求）
            if "name" in edge:
                original_name = edge["name"]
                edge["name"] = original_name.upper()
                if edge["name"] != original_name:
                    logger.warning(f"Edge type name '{original_name}' auto-converted to '{edge['name']}'")
            # 修正 source_targets 中的实体名称引用，与转换后的 PascalCase 保持一致
            for st in edge.get("source_targets", []):
                if st.get("source") in entity_name_map:
                    st["source"] = entity_name_map[st["source"]]
                if st.get("target") in entity_name_map:
                    st["target"] = entity_name_map[st["target"]]
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."
        
        # Zep API 限制：最多 10 个自定义实体类型，最多 10 个自定义边类型
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

        # 去重：按 name 去重，保留首次出现的
        seen_names = set()
        deduped = []
        for entity in result["entity_types"]:
            name = entity.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                deduped.append(entity)
            elif name in seen_names:
                logger.warning(f"Duplicate entity type '{name}' removed during validation")
        result["entity_types"] = deduped

        # 兜底类型定义
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }
        
        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }
        
        # 检查是否已有兜底类型
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names
        
        # 需要添加的兜底类型
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)
        
        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)
            
            # 如果添加后会超过 10 个，需要移除一些现有类型
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # 计算需要移除多少个
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # 从末尾移除（保留前面更重要的具体类型）
                result["entity_types"] = result["entity_types"][:-to_remove]
            
            # 添加兜底类型
            result["entity_types"].extend(fallbacks_to_add)
        
        # 最终确保不超过限制（防御性编程）
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]
        
        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]
        
        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        将本体定义转换为Python代码（类似ontology.py）
        
        Args:
            ontology: 本体定义
            
        Returns:
            Python代码字符串
        """
        code_lines = [
            '"""',
            '自定义实体类型定义',
            '由MiroFish自动生成，用于社会舆论模拟',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== 实体类型定义 ==============',
            '',
        ]
        
        # 生成实体类型
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== 关系类型定义 ==============')
        code_lines.append('')
        
        # 生成关系类型
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # 转换为PascalCase类名
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        # 生成类型字典
        code_lines.append('# ============== 类型配置 ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        
        # 生成边的source_targets映射
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)

