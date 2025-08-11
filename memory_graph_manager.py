"""
记忆关联图谱管理器
使用OpenAI智能构建记忆之间的关联关系，实现记忆串联和推理
"""
import json
import time
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger

class RelationType(Enum):
    """关联类型"""
    IDENTITY = "identity"           # 身份关联：张三 → 程序员
    FAMILY = "family"              # 家庭关联：星星 → 弟弟
    LOCATION = "location"          # 地点关联：张三 → 北京
    SKILL = "skill"               # 技能关联：张三 → Python
    PREFERENCE = "preference"      # 偏好关联：张三 → 编程
    WORK = "work"                 # 工作关联：程序员 → 编程
    PHYSICAL = "physical"         # 身体关联：张三 → 180cm
    TEMPORAL = "temporal"         # 时间关联：今天 → 工作
    CAUSAL = "causal"            # 因果关联：喜欢编程 → 学Python
    SEMANTIC = "semantic"         # 语义关联：相似概念

@dataclass
class MemoryNode:
    """记忆节点"""
    id: str
    content: str
    memory_type: str
    importance: float
    entities: List[str]  # 提取的实体
    concepts: List[str]  # 提取的概念
    timestamp: float

@dataclass
class MemoryEdge:
    """记忆边（关联）"""
    id: str
    source_memory_id: str
    target_memory_id: str
    relation_type: RelationType
    confidence: float
    description: str
    timestamp: float
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class GraphSearchResult:
    """图谱搜索结果"""
    primary_memories: List[MemoryNode]
    related_memories: List[MemoryNode]
    relations: List[MemoryEdge]
    inference_chain: List[str]  # 推理链

class MemoryGraphManager:
    """记忆关联图谱管理器"""
    
    def __init__(self, openai_client, db_connection):
        self.openai_client = openai_client
        self.conn = db_connection
        
        # 配置参数
        self.relation_confidence_threshold = 0.7
        self.max_relation_depth = 3
        self.entity_similarity_threshold = 0.8
        
        self._init_graph_storage()
        logger.info("🕸️ 记忆关联图谱管理器初始化完成")
    
    def _init_graph_storage(self):
        """初始化图谱存储"""
        try:
            # 创建记忆节点表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT,
                    importance REAL,
                    entities TEXT,
                    concepts TEXT,
                    timestamp REAL
                )
            """)
            
            # 创建记忆关联表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_edges (
                    id TEXT PRIMARY KEY,
                    source_memory_id TEXT,
                    target_memory_id TEXT,
                    relation_type TEXT,
                    confidence REAL,
                    description TEXT,
                    timestamp REAL,
                    FOREIGN KEY (source_memory_id) REFERENCES memory_nodes (id),
                    FOREIGN KEY (target_memory_id) REFERENCES memory_nodes (id)
                )
            """)
            
            # 创建实体索引表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_index (
                    entity TEXT,
                    memory_id TEXT,
                    entity_type TEXT,
                    confidence REAL,
                    PRIMARY KEY (entity, memory_id)
                )
            """)
            
            self.conn.commit()
            logger.debug("✅ 图谱存储表初始化完成")
            
        except Exception as e:
            logger.error(f"初始化图谱存储失败: {e}")
    
    async def add_memory_to_graph(self, memory_id: str, content: str, 
                                memory_type: str, importance: float) -> bool:
        """将记忆添加到图谱"""
        try:
            logger.debug(f"🕸️ 添加记忆到图谱: {content}")
            
            # 1. 提取实体和概念
            entities, concepts = await self._extract_entities_and_concepts(content)
            
            # 2. 创建记忆节点
            node = MemoryNode(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                entities=entities,
                concepts=concepts,
                timestamp=time.time()
            )
            
            # 3. 存储节点
            await self._store_memory_node(node)
            
            # 4. 构建与现有记忆的关联
            await self._build_relations_for_new_memory(node)
            
            logger.info(f"✅ 记忆图谱添加成功: {len(entities)} 个实体, {len(concepts)} 个概念")
            return True
            
        except Exception as e:
            logger.error(f"添加记忆到图谱失败: {e}")
            return False
    
    async def _extract_entities_and_concepts(self, content: str) -> Tuple[List[str], List[str]]:
        """使用GPT提取实体和概念"""
        try:
            prompt = f"""
请从以下文本中提取关键实体和概念：

文本: {content}

请提取：
1. 实体：人名、地名、物品名、具体事物
2. 概念：抽象概念、关系、属性、状态

请严格按JSON格式输出：
{{
    "entities": ["实体1", "实体2"],
    "concepts": ["概念1", "概念2"]
}}

示例：
文本: "星星是我的弟弟"
输出: {{"entities": ["星星"], "concepts": ["弟弟", "家庭关系"]}}

文本: "我住在北京"
输出: {{"entities": ["北京"], "concepts": ["居住地", "地理位置"]}}
"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            if response.choices:
                result_text = response.choices[0].message.content.strip()
                try:
                    result_data = json.loads(result_text)
                    entities = result_data.get("entities", [])
                    concepts = result_data.get("concepts", [])
                    return entities, concepts
                except json.JSONDecodeError:
                    logger.warning("GPT实体提取JSON解析失败")
                    return self._fallback_entity_extraction(content)
            
            return self._fallback_entity_extraction(content)
            
        except Exception as e:
            logger.error(f"GPT实体提取失败: {e}")
            return self._fallback_entity_extraction(content)
    
    def _fallback_entity_extraction(self, content: str) -> Tuple[List[str], List[str]]:
        """备用实体提取"""
        import re
        
        entities = []
        concepts = []
        
        # 简单的实体识别
        # 人名识别（中文姓名）
        name_matches = re.findall(r'[星月光明亮强伟华建军国庆][星月光明亮强伟华建军国庆]?', content)
        entities.extend(name_matches)
        
        # 地名识别
        location_keywords = ['北京', '上海', '深圳', '广州', '杭州', '成都', '重庆', '武汉', '西安', '南京']
        for location in location_keywords:
            if location in content:
                entities.append(location)
        
        # 概念识别
        if '弟弟' in content or '哥哥' in content or '姐姐' in content or '妹妹' in content:
            concepts.append('家庭关系')
        
        if '喜欢' in content or '爱好' in content:
            concepts.append('偏好')
        
        if '工作' in content or '职业' in content:
            concepts.append('职业')
        
        if '住' in content or '家' in content:
            concepts.append('居住')
        
        return entities, concepts
    
    async def _store_memory_node(self, node: MemoryNode):
        """存储记忆节点"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO memory_nodes 
                (id, content, memory_type, importance, entities, concepts, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                node.id,
                node.content,
                node.memory_type,
                node.importance,
                json.dumps(node.entities),
                json.dumps(node.concepts),
                node.timestamp
            ))
            
            # 同时更新实体索引
            for entity in node.entities:
                self.conn.execute("""
                    INSERT OR REPLACE INTO entity_index 
                    (entity, memory_id, entity_type, confidence)
                    VALUES (?, ?, ?, ?)
                """, (entity, node.id, "entity", 1.0))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"存储记忆节点失败: {e}")
    
    async def _build_relations_for_new_memory(self, new_node: MemoryNode):
        """为新记忆构建与现有记忆的关联"""
        try:
            # 获取所有现有记忆节点
            cursor = self.conn.execute("""
                SELECT id, content, memory_type, entities, concepts 
                FROM memory_nodes WHERE id != ?
            """, (new_node.id,))
            
            existing_nodes = []
            for row in cursor.fetchall():
                node = MemoryNode(
                    id=row[0],
                    content=row[1],
                    memory_type=row[2],
                    importance=0,  # 暂时不需要
                    entities=json.loads(row[3]) if row[3] else [],
                    concepts=json.loads(row[4]) if row[4] else [],
                    timestamp=0  # 暂时不需要
                )
                existing_nodes.append(node)
            
            # 分析关联关系
            for existing_node in existing_nodes:
                relations = await self._analyze_relation_between_memories(new_node, existing_node)
                
                for relation in relations:
                    if relation.confidence >= self.relation_confidence_threshold:
                        await self._store_memory_edge(relation)
            
        except Exception as e:
            logger.error(f"构建记忆关联失败: {e}")
    
    async def _analyze_relation_between_memories(self, memory1: MemoryNode, 
                                               memory2: MemoryNode) -> List[MemoryEdge]:
        """分析两个记忆之间的关联关系"""
        try:
            relations = []
            
            # 1. 实体关联检测
            common_entities = set(memory1.entities) & set(memory2.entities)
            if common_entities:
                for entity in common_entities:
                    relation = await self._create_entity_relation(memory1, memory2, entity)
                    if relation:
                        relations.append(relation)
            
            # 2. 概念关联检测
            common_concepts = set(memory1.concepts) & set(memory2.concepts)
            if common_concepts:
                for concept in common_concepts:
                    relation = await self._create_concept_relation(memory1, memory2, concept)
                    if relation:
                        relations.append(relation)
            
            # 3. 语义关联检测（使用GPT）
            if not relations:  # 如果没有直接关联，尝试语义关联
                semantic_relation = await self._analyze_semantic_relation(memory1, memory2)
                if semantic_relation:
                    relations.append(semantic_relation)
            
            return relations
            
        except Exception as e:
            logger.error(f"分析记忆关联失败: {e}")
            return []
    
    async def _create_entity_relation(self, memory1: MemoryNode, memory2: MemoryNode, 
                                    entity: str) -> Optional[MemoryEdge]:
        """创建实体关联"""
        try:
            # 判断关联类型
            relation_type = RelationType.IDENTITY
            confidence = 0.9
            description = f"都提到了实体: {entity}"
            
            # 根据记忆类型优化关联类型
            if memory1.memory_type == "family" or memory2.memory_type == "family":
                relation_type = RelationType.FAMILY
                description = f"家庭相关实体: {entity}"
            elif memory1.memory_type == "work" or memory2.memory_type == "work":
                relation_type = RelationType.WORK
                description = f"工作相关实体: {entity}"
            
            edge_id = f"{memory1.id}_{memory2.id}_{entity}"
            
            return MemoryEdge(
                id=edge_id,
                source_memory_id=memory1.id,
                target_memory_id=memory2.id,
                relation_type=relation_type,
                confidence=confidence,
                description=description,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"创建实体关联失败: {e}")
            return None
    
    async def _create_concept_relation(self, memory1: MemoryNode, memory2: MemoryNode, 
                                     concept: str) -> Optional[MemoryEdge]:
        """创建概念关联"""
        try:
            relation_type = RelationType.SEMANTIC
            confidence = 0.8
            description = f"共同概念: {concept}"
            
            # 特定概念的关联类型
            if concept == "家庭关系":
                relation_type = RelationType.FAMILY
            elif concept == "职业":
                relation_type = RelationType.WORK
            elif concept == "偏好":
                relation_type = RelationType.PREFERENCE
            
            edge_id = f"{memory1.id}_{memory2.id}_{concept}"
            
            return MemoryEdge(
                id=edge_id,
                source_memory_id=memory1.id,
                target_memory_id=memory2.id,
                relation_type=relation_type,
                confidence=confidence,
                description=description,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"创建概念关联失败: {e}")
            return None
    
    async def _analyze_semantic_relation(self, memory1: MemoryNode, 
                                   memory2: MemoryNode) -> Optional[MemoryEdge]:
        """使用GPT分析语义关联 - 修复版"""
        try:
            prompt = f"""
    分析以下两段记忆是否存在语义关联：

    记忆1: {memory1.content}
    记忆2: {memory2.content}

    请判断它们之间是否存在关联，如果存在，选择最合适的关联类型。

    请严格按JSON格式输出：
    {{
        "has_relation": true,
        "relation_type": "family",
        "confidence": 0.8,
        "description": "关联描述"
    }}

    或者：
    {{
        "has_relation": false
    }}

    关联类型只能是：identity, family, location, skill, preference, work, physical, causal, semantic
    """

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1
            )
            
            if response.choices:
                result_text = response.choices[0].message.content.strip()
                
                # 清理可能的格式问题
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0].strip()
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].strip()
                
                try:
                    result_data = json.loads(result_text)
                    
                    if result_data.get("has_relation", False):
                        relation_type = result_data.get("relation_type", "semantic")
                        
                        # 验证关联类型
                        valid_types = [rt.value for rt in RelationType]
                        if relation_type not in valid_types:
                            relation_type = "semantic"
                        
                        edge_id = f"{memory1.id}_{memory2.id}_semantic"
                        
                        return MemoryEdge(
                            id=edge_id,
                            source_memory_id=memory1.id,
                            target_memory_id=memory2.id,
                            relation_type=RelationType(relation_type),
                            confidence=min(result_data.get("confidence", 0.5), 1.0),
                            description=result_data.get("description", "语义关联"),
                            timestamp=time.time()
                        )
                except json.JSONDecodeError as e:
                    logger.debug(f"语义关联分析JSON解析失败: {result_text[:100]}...")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"语义关联分析失败: {e}")
            return None
    async def enhanced_graph_search(self, question: str, primary_memories: List[Any]) -> Tuple[str, List[str]]:
        """增强图谱搜索 - 多层关联"""
        try:
            logger.info(f"🕸️ 增强图谱搜索: {question}")
            
            # 1. 从问题中提取实体
            question_entities, question_concepts = await self._extract_entities_and_concepts(question)
            logger.debug(f"问题实体: {question_entities}, 概念: {question_concepts}")
            
            # 2. 直接实体搜索
            entity_memories = []
            for entity in question_entities:
                cursor = self.conn.execute("""
                    SELECT memory_id FROM entity_index WHERE entity = ?
                """, (entity,))
                
                for row in cursor.fetchall():
                    memory = await self._get_memory_node_by_id(row[0])
                    if memory and memory not in entity_memories:
                        entity_memories.append(memory)
            
            # 3. 获取所有相关记忆
            all_related_memories = []
            all_relations = []
            
            # 从主要记忆出发
            for memory in primary_memories:
                related_memories, relations = await self._get_related_memories(memory.id)
                all_related_memories.extend(related_memories)
                all_relations.extend(relations)
            
            # 从实体记忆出发
            for memory in entity_memories:
                related_memories, relations = await self._get_related_memories(memory.id)
                all_related_memories.extend(related_memories)
                all_relations.extend(relations)
            
            # 去重
            all_memories = list(set(primary_memories + entity_memories + all_related_memories))
            
            # 4. 构建增强推理链
            inference_chain = await self._build_enhanced_inference_chain(
                question, question_entities, all_memories, all_relations
            )
            
            # 5. 生成回答
            response = await self._generate_enhanced_response(question, all_memories, inference_chain)
            
            used_memory_ids = [m.id for m in all_memories]
            
            logger.info(f"✅ 增强图谱回答: {response}")
            return response, used_memory_ids
            
        except Exception as e:
            logger.error(f"增强图谱搜索失败: {e}")
            return "", []

    async def _get_memory_node_by_id(self, memory_id: str) -> Optional[MemoryNode]:
        """根据ID获取记忆节点"""
        try:
            cursor = self.conn.execute("""
                SELECT content, memory_type, importance, entities, concepts, timestamp
                FROM memory_nodes WHERE id = ?
            """, (memory_id,))
            
            result = cursor.fetchone()
            if result:
                return MemoryNode(
                    id=memory_id,
                    content=result[0],
                    memory_type=result[1],
                    importance=result[2],
                    entities=json.loads(result[3]) if result[3] else [],
                    concepts=json.loads(result[4]) if result[4] else [],
                    timestamp=result[5]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"获取记忆节点失败: {e}")
            return None

    async def _build_enhanced_inference_chain(self, question: str, question_entities: List[str],
                                            memories: List[MemoryNode], 
                                            relations: List[MemoryEdge]) -> List[str]:
        """构建增强推理链"""
        try:
            inference_chain = []
            
            # 实体推理
            for entity in question_entities:
                entity_memories = [m for m in memories if entity in m.entities]
                if entity_memories:
                    inference_chain.append(f"实体'{entity}'相关记忆:")
                    for memory in entity_memories[:3]:  # 最多3条
                        inference_chain.append(f"  - {memory.content}")
            
            # 关联推理
            high_confidence_relations = [r for r in relations if r.confidence > 0.8]
            if high_confidence_relations:
                inference_chain.append("高置信度关联:")
                for relation in high_confidence_relations[:3]:
                    inference_chain.append(f"  - {relation.description}")
            
            return inference_chain
            
        except Exception as e:
            logger.error(f"构建推理链失败: {e}")
            return []

    async def _generate_enhanced_response(self, question: str, memories: List[MemoryNode], 
                                        inference_chain: List[str]) -> str:
        """生成增强回答"""
        try:
            # 构建更丰富的上下文
            memory_context = []
            
            # 按重要性排序
            sorted_memories = sorted(memories, key=lambda x: x.importance, reverse=True)
            
            for memory in sorted_memories[:10]:  # 最多10条记忆
                memory_context.append(f"- {memory.content} (重要性: {memory.importance:.2f})")
            
            context = "\n".join(memory_context)
            inference_text = "\n".join(inference_chain) if inference_chain else "无特殊推理"
            
            prompt = f"""基于以下记忆信息和推理链，回答用户问题。

    问题: {question}

    记忆信息:
    {context}

    推理链:
    {inference_text}

    要求:
    1. 优先使用推理链中的信息
    2. 如果能找到答案，直接回答
    3. 如果需要推理，说明推理过程
    4. 如果确实没有信息，明确说明缺少什么
    5. 1-2句话简洁回答

    请直接给出回答："""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            
            if response.choices:
                return response.choices[0].message.content.strip()
            
            return "抱歉，我无法基于现有信息回答这个问题。"
            
        except Exception as e:
            logger.error(f"生成增强回答失败: {e}")
            return "抱歉，处理问题时出现了错误。"
    
    async def _store_memory_edge(self, edge: MemoryEdge):
        """存储记忆关联"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO memory_edges 
                (id, source_memory_id, target_memory_id, relation_type, confidence, description, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                edge.id,
                edge.source_memory_id,
                edge.target_memory_id,
                edge.relation_type.value,
                edge.confidence,
                edge.description,
                edge.timestamp
            ))
            
            self.conn.commit()
            logger.debug(f"✅ 存储关联: {edge.description}")
            
        except Exception as e:
            logger.error(f"存储记忆关联失败: {e}")
    
    async def graph_search_and_respond(self, question: str, 
                                     primary_memories: List[Any]) -> Tuple[str, List[str]]:
        """基于图谱搜索并生成关联回答"""
        try:
            logger.info(f"🕸️ 图谱搜索问题: {question}")
            # 首先尝试增强搜索
            enhanced_response, enhanced_memory_ids = await self.enhanced_graph_search(question, primary_memories)
            
            if enhanced_response and "抱歉" not in enhanced_response:
                return enhanced_response, enhanced_memory_ids
            # 1. 获取主要记忆的相关记忆
            all_related_memories = []
            all_relations = []
            
            for memory in primary_memories:
                related_memories, relations = await self._get_related_memories(memory.id)
                all_related_memories.extend(related_memories)
                all_relations.extend(relations)
            
            # 2. 构建推理链
            inference_chain = await self._build_inference_chain(question, primary_memories, all_related_memories, all_relations)
            
            # 3. 生成关联回答
            response = await self._generate_graph_response(question, primary_memories, all_related_memories, inference_chain)
            
            # 4. 收集使用的记忆ID
            used_memory_ids = [m.id for m in primary_memories] + [m.id for m in all_related_memories]
            
            logger.info(f"✅ 图谱回答生成: {response}")
            return response, used_memory_ids
            
        except Exception as e:
            logger.error(f"图谱搜索回答失败: {e}")
            return "", []
    
    async def _get_related_memories(self, memory_id: str) -> Tuple[List[MemoryNode], List[MemoryEdge]]:
        """获取相关记忆 - 增强版"""
        try:
            # 1. 获取目标记忆的实体和概念
            cursor = self.conn.execute("""
                SELECT entities, concepts FROM memory_nodes WHERE id = ?
            """, (memory_id,))
            
            result = cursor.fetchone()
            if not result:
                return [], []
            
            target_entities = json.loads(result[0]) if result[0] else []
            target_concepts = json.loads(result[1]) if result[1] else []
            
            # 2. 查找直接关联的记忆
            cursor = self.conn.execute("""
                SELECT e.target_memory_id, e.relation_type, e.confidence, e.description,
                    n.content, n.memory_type, n.importance, n.entities, n.concepts
                FROM memory_edges e
                JOIN memory_nodes n ON e.target_memory_id = n.id
                WHERE e.source_memory_id = ?
                UNION
                SELECT e.source_memory_id, e.relation_type, e.confidence, e.description,
                    n.content, n.memory_type, n.importance, n.entities, n.concepts
                FROM memory_edges e
                JOIN memory_nodes n ON e.source_memory_id = n.id
                WHERE e.target_memory_id = ?
                ORDER BY e.confidence DESC
            """, (memory_id, memory_id))
            
            related_memories = []
            relations = []
            found_memory_ids = set()
            
            for row in cursor.fetchall():
                related_memory = MemoryNode(
                    id=row[0],
                    content=row[4],
                    memory_type=row[5],
                    importance=row[6],
                    entities=json.loads(row[7]) if row[7] else [],
                    concepts=json.loads(row[8]) if row[8] else [],
                    timestamp=0
                )
                related_memories.append(related_memory)
                found_memory_ids.add(row[0])
                
                relation = MemoryEdge(
                    id="",
                    source_memory_id=memory_id,
                    target_memory_id=row[0],
                    relation_type=RelationType(row[1]),
                    confidence=row[2],
                    description=row[3],
                    timestamp=0
                )
                relations.append(relation)
            
            # 3. 查找实体关联的记忆（新增）
            if target_entities:
                for entity in target_entities:
                    cursor = self.conn.execute("""
                        SELECT memory_id FROM entity_index 
                        WHERE entity = ? AND memory_id != ?
                    """, (entity, memory_id))
                    
                    for entity_row in cursor.fetchall():
                        entity_memory_id = entity_row[0]
                        if entity_memory_id not in found_memory_ids:
                            # 获取这个记忆的详细信息
                            memory_cursor = self.conn.execute("""
                                SELECT content, memory_type, importance, entities, concepts
                                FROM memory_nodes WHERE id = ?
                            """, (entity_memory_id,))
                            
                            memory_result = memory_cursor.fetchone()
                            if memory_result:
                                related_memory = MemoryNode(
                                    id=entity_memory_id,
                                    content=memory_result[0],
                                    memory_type=memory_result[1],
                                    importance=memory_result[2],
                                    entities=json.loads(memory_result[3]) if memory_result[3] else [],
                                    concepts=json.loads(memory_result[4]) if memory_result[4] else [],
                                    timestamp=0
                                )
                                related_memories.append(related_memory)
                                found_memory_ids.add(entity_memory_id)
                                
                                # 创建实体关联
                                relation = MemoryEdge(
                                    id="",
                                    source_memory_id=memory_id,
                                    target_memory_id=entity_memory_id,
                                    relation_type=RelationType.IDENTITY,
                                    confidence=0.9,
                                    description=f"共同实体: {entity}",
                                    timestamp=0
                                )
                                relations.append(relation)
            
            return related_memories, relations
            
        except Exception as e:
            logger.error(f"获取相关记忆失败: {e}")
            return [], []
    
    async def _build_inference_chain(self, question: str, primary_memories: List[Any], 
                                   related_memories: List[MemoryNode], 
                                   relations: List[MemoryEdge]) -> List[str]:
        """构建推理链"""
        try:
            inference_chain = []
            
            # 简单的推理链构建
            for primary in primary_memories:
                inference_chain.append(f"已知: {primary.content}")
            
            for relation in relations:
                if relation.confidence > 0.8:
                    inference_chain.append(f"关联: {relation.description}")
            
            for related in related_memories:
                inference_chain.append(f"相关: {related.content}")
            
            return inference_chain
            
        except Exception as e:
            logger.error(f"构建推理链失败: {e}")
            return []
    
    async def _generate_graph_response(self, question: str, primary_memories: List[Any],
                                     related_memories: List[MemoryNode], 
                                     inference_chain: List[str]) -> str:
        """生成基于图谱的关联回答"""
        try:
            # 构建上下文
            context_parts = []
            
            # 主要记忆
            context_parts.append("直接相关记忆:")
            for memory in primary_memories:
                context_parts.append(f"- {memory.content}")
            
            # 相关记忆
            if related_memories:
                context_parts.append("\n关联记忆:")
                for memory in related_memories[:3]:  # 最多3条相关记忆
                    context_parts.append(f"- {memory.content}")
            
            # 推理链
            if inference_chain:
                context_parts.append("\n推理链:")
                for step in inference_chain[:5]:  # 最多5步推理
                    context_parts.append(f"→ {step}")
            
            context = "\n".join(context_parts)
            
            prompt = f"""基于以下记忆图谱信息，回答用户问题。请利用记忆之间的关联关系进行推理。

问题: {question}

记忆图谱信息:
{context}

要求:
1. 基于现有记忆进行关联推理
2. 如果能通过关联推断出答案，请给出推理过程
3. 如果确实缺少关键信息，明确指出缺少什么
4. 1-2句话简洁回答
5. 用第一人称回答

请直接给出回答："""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            
            if response.choices:
                return response.choices[0].message.content.strip()
            
            return "抱歉，我无法基于现有信息回答这个问题。"
            
        except Exception as e:
            logger.error(f"生成图谱回答失败: {e}")
            return "抱歉，处理问题时出现了错误。"
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        try:
            # 节点统计
            cursor = self.conn.execute("SELECT COUNT(*) FROM memory_nodes")
            node_count = cursor.fetchone()[0]
            
            # 边统计
            cursor = self.conn.execute("SELECT COUNT(*) FROM memory_edges")
            edge_count = cursor.fetchone()[0]
            
            # 关联类型统计
            cursor = self.conn.execute("""
                SELECT relation_type, COUNT(*) 
                FROM memory_edges 
                GROUP BY relation_type
                ORDER BY COUNT(*) DESC
            """)
            relation_types = dict(cursor.fetchall())
            
            # 实体统计
            cursor = self.conn.execute("SELECT COUNT(DISTINCT entity) FROM entity_index")
            entity_count = cursor.fetchone()[0]
            
            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "entity_count": entity_count,
                "relation_types": relation_types,
                "graph_density": edge_count / max(node_count * (node_count - 1), 1)
            }
            
        except Exception as e:
            logger.error(f"获取图谱统计失败: {e}")
            return {"error": str(e)}