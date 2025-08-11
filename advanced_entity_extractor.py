"""
高级实体提取器
支持多语言、上下文理解、冲突检测和模糊匹配
"""
import re
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger

class EntityType(Enum):
    """实体类型"""
    PERSON = "person"           # 人名
    FAMILY_ROLE = "family_role" # 家庭角色
    LOCATION = "location"       # 地点
    AGE = "age"                # 年龄
    OCCUPATION = "occupation"   # 职业
    SKILL = "skill"            # 技能
    PREFERENCE = "preference"   # 偏好
    PHYSICAL = "physical"       # 身体特征
    EDUCATION = "education"     # 教育
    TIME = "time"              # 时间

class ConflictType(Enum):
    """冲突类型"""
    CONTRADICTION = "contradiction"  # 直接矛盾
    UPDATE = "update"               # 信息更新
    CLARIFICATION = "clarification" # 澄清补充
    DIFFERENT_CONTEXT = "different_context"  # 不同语境

@dataclass
class ExtractedEntity:
    """提取的实体"""
    text: str
    entity_type: EntityType
    confidence: float
    context: str
    normalized_form: str  # 标准化形式
    aliases: List[str]    # 别名
    metadata: Dict[str, Any]

@dataclass
class ConflictInfo:
    """冲突信息"""
    entity: str
    conflict_type: ConflictType
    old_value: Any
    new_value: Any
    confidence: float
    resolution: str

class AdvancedEntityExtractor:
    """高级实体提取器"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        
        # 多语言支持
        self.language_patterns = {
            "zh": self._init_chinese_patterns(),
            "en": self._init_english_patterns(),
        }
        
        # 实体标准化映射
        self.entity_normalizer = {
            "family_roles": {
                "弟弟": ["弟弟", "小弟", "弟", "兄弟"],
                "妹妹": ["妹妹", "小妹", "妹", "姐妹"],
                "哥哥": ["哥哥", "大哥", "哥", "兄弟"],
                "姐姐": ["姐姐", "大姐", "姐", "姐妹"],
                "妈妈": ["妈妈", "母亲", "老妈", "妈", "母亲大人"],
                "爸爸": ["爸爸", "父亲", "老爸", "爸", "父亲大人"],
            }
        }
        
        # 冲突检测规则
        self.conflict_rules = self._init_conflict_rules()
        
        logger.info("🔍 高级实体提取器初始化完成")
    
    def _init_chinese_patterns(self) -> Dict[str, List[str]]:
        """初始化中文模式"""
        return {
            "person": [  # 改为person
                r'我叫(.{1,4})',
                r'我的名字.*?(.{1,4})',
                r'(.{1,4})是我.*?',
                r'我是(.{1,4})',
            ],
            "family_role": [
                r'我的?(弟弟|妹妹|哥哥|姐姐|妈妈|爸爸|母亲|父亲)',
                r'(弟弟|妹妹|哥哥|姐姐|妈妈|爸爸|母亲|父亲).*?我',
            ],
            "age": [
                r'(\d+)\s*岁',
                r'年龄.*?(\d+)',
                r'今年.*?(\d+)',
            ],
            "location": [
                r'住在(.{1,10})',
                r'家在(.{1,10})',
                r'来自(.{1,10})',
                r'在(.{1,10})工作',
            ],
            "occupation": [
                r'我是(.{1,10}?(?:师|员|家|生|工))',
                r'职业.*?(.{1,10})',
                r'工作.*?(.{1,10})',
            ]
        }
    
    def _init_english_patterns(self) -> Dict[str, List[str]]:
        """初始化英文模式"""
        return {
            "person_name": [
                r'my name is (\w+)',
                r'i am (\w+)',
                r'(\w+) is my',
            ],
            "family_role": [
                r'my (brother|sister|mother|father|mom|dad)',
                r'(brother|sister|mother|father|mom|dad)',
            ],
            "age": [
                r'(\d+)\s*years?\s*old',
                r'age.*?(\d+)',
            ]
        }
    
    def _init_conflict_rules(self) -> Dict[str, Dict]:
        """初始化冲突检测规则"""
        return {
            "family_role_conflicts": {
                "弟弟": ["妹妹", "姐姐", "哥哥"],
                "妹妹": ["弟弟", "哥哥", "姐姐"],
                "哥哥": ["弟弟", "妹妹", "姐姐"],
                "姐姐": ["弟弟", "妹妹", "哥哥"],
            },
            "occupation_conflicts": {
                # 一般情况下，一个人只能有一个主要职业
                "mutual_exclusive": True
            },
            "age_conflicts": {
                # 年龄只能增长，不能减少（除非是更正错误）
                "monotonic": True,
                "max_change_per_day": 0.01  # 每天最大变化
            }
        }
    
    async def extract_entities_advanced(self, text: str, context: str = "", 
                                      language: str = "auto") -> Tuple[List[ExtractedEntity], List[ConflictInfo]]:
        """高级实体提取"""
        try:
            logger.debug(f"🔍 高级实体提取: {text}")
            
            # 1. 语言检测
            detected_lang = self._detect_language(text) if language == "auto" else language
            
            # 2. 规则提取
            rule_entities = await self._rule_based_extraction(text, detected_lang)
            
            # 3. AI增强提取
            ai_entities = await self._ai_enhanced_extraction(text, context, detected_lang)
            
            # 4. 实体融合和标准化
            merged_entities = await self._merge_and_normalize_entities(rule_entities + ai_entities)
            
            # 5. 冲突检测
            conflicts = await self._detect_conflicts(merged_entities, text)
            
            # 6. 质量过滤
            filtered_entities = self._filter_by_quality(merged_entities)
            
            logger.info(f"✅ 提取完成: {len(filtered_entities)} 个实体, {len(conflicts)} 个冲突")
            return filtered_entities, conflicts
            
        except Exception as e:
            logger.error(f"高级实体提取失败: {e}")
            return [], []
    
    def _detect_language(self, text: str) -> str:
        """检测语言"""
        # 简单的语言检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if chinese_chars > english_chars:
            return "zh"
        elif english_chars > 0:
            return "en"
        else:
            return "zh"  # 默认中文
    
    async def _rule_based_extraction(self, text: str, language: str) -> List[ExtractedEntity]:
        """基于规则的实体提取"""
        entities = []
        patterns = self.language_patterns.get(language, {})
        
        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group(1) if match.groups() else match.group(0)
                    
                    # 标准化
                    normalized = self._normalize_entity(entity_text, entity_type)
                    aliases = self._get_aliases(normalized, entity_type)
                    
                    entity = ExtractedEntity(
                        text=entity_text,
                        entity_type=EntityType(entity_type.replace("_", "_")),
                        confidence=0.9,  # 规则提取高置信度
                        context=text,
                        normalized_form=normalized,
                        aliases=aliases,
                        metadata={"extraction_method": "rule", "pattern": pattern}
                    )
                    entities.append(entity)
        
        return entities
    
    async def _ai_enhanced_extraction(self, text: str, context: str, language: str) -> List[ExtractedEntity]:
        """AI增强实体提取"""
        try:
            prompt = f"""
你是一个高级实体提取专家，请从以下文本中提取实体信息。

文本: {text}
上下文: {context}

**特别注意身份关联：**
- 如果文本是"我叫X"，提取person实体"X"，并标记为user_identity
- 如果文本是"X多少岁"且X是已知的用户姓名，建立关联

请提取以下类型的实体：
- person: 人名（特别标注是否为用户本人）
- family_role: 家庭角色
- location: 地点  
- age: 年龄
- occupation: 职业
- skill: 技能
- preference: 偏好
- physical: 身体特征
- education: 教育
- time: 时间

请严格按JSON格式输出：
{{
    "entities": [
        {{
            "text": "提取的文本",
            "type": "实体类型",
            "confidence": 0.0-1.0,
            "normalized": "标准化形式",
            "aliases": ["别名1", "别名2"],
            "is_user_identity": true/false
        }}
    ]
}}
"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            if response.choices:
                result_text = response.choices[0].message.content.strip()
                
                # 清理格式
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0].strip()
                
                try:
                    result_data = json.loads(result_text)
                    entities = []
                    
                    for entity_data in result_data.get("entities", []):
                        entity = ExtractedEntity(
                            text=entity_data["text"],
                            entity_type=EntityType(entity_data["type"]),
                            confidence=entity_data["confidence"],
                            context=text,
                            normalized_form=entity_data["normalized"],
                            aliases=entity_data.get("aliases", []),
                            metadata={"extraction_method": "ai"}
                        )
                        entities.append(entity)
                    
                    return entities
                    
                except json.JSONDecodeError:
                    logger.warning("AI实体提取JSON解析失败")
                    return []
            
            return []
            
        except Exception as e:
            logger.error(f"AI实体提取失败: {e}")
            return []
    
    def _normalize_entity(self, entity_text: str, entity_type: str) -> str:
        """实体标准化"""
        if entity_type == "family_role":
            for standard, aliases in self.entity_normalizer["family_roles"].items():
                if entity_text in aliases:
                    return standard
        
        return entity_text.strip()
    
    def _get_aliases(self, normalized_entity: str, entity_type: str) -> List[str]:
        """获取实体别名"""
        if entity_type == "family_role":
            return self.entity_normalizer["family_roles"].get(normalized_entity, [])
        
        return []
    
    async def _merge_and_normalize_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """合并和标准化实体"""
        # 按标准化形式分组
        entity_groups = {}
        
        for entity in entities:
            key = f"{entity.entity_type.value}:{entity.normalized_form}"
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        # 合并同类实体
        merged_entities = []
        for group in entity_groups.values():
            if len(group) == 1:
                merged_entities.append(group[0])
            else:
                # 选择置信度最高的
                best_entity = max(group, key=lambda x: x.confidence)
                # 合并别名
                all_aliases = set()
                for entity in group:
                    all_aliases.update(entity.aliases)
                    all_aliases.add(entity.text)
                
                best_entity.aliases = list(all_aliases)
                merged_entities.append(best_entity)
        
        return merged_entities
    
    async def _detect_conflicts(self, entities: List[ExtractedEntity], text: str) -> List[ConflictInfo]:
        """检测冲突"""
        conflicts = []
        
        # 家庭角色冲突检测
        family_roles = [e for e in entities if e.entity_type == EntityType.FAMILY_ROLE]
        if len(family_roles) > 1:
            conflict = await self._detect_family_role_conflict(family_roles, text)
            if conflict:
                conflicts.append(conflict)
        
        # 年龄冲突检测
        ages = [e for e in entities if e.entity_type == EntityType.AGE]
        if ages:
            conflict = await self._detect_age_conflict(ages[0], text)
            if conflict:
                conflicts.append(conflict)
        
        return conflicts
    
    async def _detect_family_role_conflict(self, family_roles: List[ExtractedEntity], text: str) -> Optional[ConflictInfo]:
        """检测家庭角色冲突"""
        if len(family_roles) < 2:
            return None
        
        # 检查是否存在互斥角色
        role_names = [entity.normalized_form for entity in family_roles]
        
        for i, role1 in enumerate(role_names):
            for j, role2 in enumerate(role_names[i+1:], i+1):
                if role2 in self.conflict_rules["family_role_conflicts"].get(role1, []):
                    return ConflictInfo(
                        entity="family_role",
                        conflict_type=ConflictType.CONTRADICTION,
                        old_value=role1,
                        new_value=role2,
                        confidence=0.8,
                        resolution="需要用户澄清"
                    )
        
        return None
    
    async def _detect_age_conflict(self, age_entity: ExtractedEntity, text: str) -> Optional[ConflictInfo]:
        """检测年龄冲突"""
        # 这里需要与历史数据对比，暂时返回None
        # 在实际实现中，会查询数据库中的历史年龄信息
        return None
    
    def _filter_by_quality(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """质量过滤"""
        return [entity for entity in entities if entity.confidence >= 0.5]
    
    async def resolve_conflict(self, conflict: ConflictInfo, user_choice: str = "auto") -> str:
        """解决冲突"""
        try:
            if user_choice == "auto":
                # 自动解决策略
                if conflict.conflict_type == ConflictType.CONTRADICTION:
                    return "keep_new"  # 保留新信息
                elif conflict.conflict_type == ConflictType.UPDATE:
                    return "merge"  # 合并信息
                else:
                    return "clarify"  # 需要澄清
            else:
                return user_choice
                
        except Exception as e:
            logger.error(f"冲突解决失败: {e}")
            return "keep_old"