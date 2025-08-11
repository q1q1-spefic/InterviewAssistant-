"""
OpenAI 高级向量记忆系统
使用 OpenAI Embedding API + 本地向量数据库 + GPT智能处理
"""
import asyncio
import json
import time
import sqlite3
import pickle
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
from loguru import logger
from memory_merger import MemoryMerger
from memory_importance_adjuster import MemoryImportanceAdjuster
from memory_tier_manager import MemoryTierManager, MemoryTier
from memory_graph_manager import MemoryGraphManager
from enhanced_memory_graph_manager import EnhancedMemoryGraphManager
from identity_resolver import IdentityResolver

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    logger.warning("ChromaDB未安装，将使用基础向量搜索")
    CHROMADB_AVAILABLE = False

@dataclass
class MemoryItem:
    """记忆项目"""
    id: str
    content: str
    memory_type: str
    importance: float
    timestamp: float
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryItem':
        return cls(**data)

@dataclass
class ExtractionResult:
    """信息提取结果"""
    is_important: bool
    importance: float
    memory_type: str
    structured_info: Dict[str, Any]
    summary: str
    keywords: List[str]
    should_store_in_memory: bool = True  # 新增字段，默认为True

class OpenAIMemoryManager:
    """基于OpenAI的高级记忆管理器"""
    
    def __init__(self, openai_client, data_dir: str = "data/openai_memory"):
        self.openai_client = openai_client
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置参数
        self.embedding_model = "text-embedding-3-small"
        self.max_memories = 1000
        self.similarity_threshold = 0.75
        self.importance_threshold = 0.3  # 降低阈值，保存更多信息
        
        # 内存缓存
        self.memory_cache = {}
        self.embedding_cache = {}
        
        # 初始化存储（这里会创建 self.conn）
        self._init_storage()
        
        # 在 self.conn 创建后再初始化依赖模块
        self.memory_merger = MemoryMerger(openai_client)
        self.importance_adjuster = MemoryImportanceAdjuster(self.conn)
        self.tier_manager = MemoryTierManager(self.conn)
        self.graph_manager = EnhancedMemoryGraphManager(openai_client, self.conn)
        self.identity_resolver = IdentityResolver(self.conn, openai_client)

        logger.info("🧠 OpenAI高级记忆管理器初始化完成")
    
    def _init_storage(self):
        """初始化存储系统"""
        # SQLite数据库用于元数据
        self.db_path = self.data_dir / "memories.db"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        
        # 创建表结构
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                importance REAL NOT NULL,
                timestamp REAL NOT NULL,
                summary TEXT,
                keywords TEXT,
                metadata TEXT
            )
        """)
        
        # 向量存储
        if CHROMADB_AVAILABLE:
            self._init_chromadb()
        else:
            self._init_simple_vector_storage()
        
        self.conn.commit()
        logger.info("📚 存储系统初始化完成")
    
    def _init_chromadb(self):
        """初始化ChromaDB向量数据库"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.data_dir / "chroma_db"),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # 创建或获取集合
            self.collection = self.chroma_client.get_or_create_collection(
                name="memories",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("✅ ChromaDB向量数据库初始化成功")
            
        except Exception as e:
            logger.error(f"ChromaDB初始化失败: {e}")
            self._init_simple_vector_storage()
    
    def _init_simple_vector_storage(self):
        """初始化简单向量存储"""
        self.vector_file = self.data_dir / "vectors.pkl"
        
        try:
            if self.vector_file.exists():
                with open(self.vector_file, 'rb') as f:
                    self.vector_storage = pickle.load(f)
            else:
                self.vector_storage = {"embeddings": [], "ids": []}
            
            logger.info("✅ 简单向量存储初始化成功")
            
        except Exception as e:
            logger.error(f"向量存储初始化失败: {e}")
            self.vector_storage = {"embeddings": [], "ids": []}
    
    def _is_target_for_merge(self, new_memory, existing_memory) -> bool:
        """判断是否是合并目标"""
        return (existing_memory.memory_type == new_memory.memory_type and 
                existing_memory.timestamp < new_memory.timestamp)

    async def _update_memory(self, memory_id: str, updated_data: Dict[str, Any]):
        """更新现有记忆"""
        try:
            # 更新SQLite
            self.conn.execute("""
                UPDATE memories SET 
                content = ?, importance = ?, metadata = ?
                WHERE id = ?
            """, (
                updated_data["content"],
                updated_data["importance"],
                json.dumps(updated_data["metadata"]),
                memory_id
            ))
            
            # 更新向量数据库
            if CHROMADB_AVAILABLE:
                # 生成新的embedding
                new_embedding = await self._get_embedding(updated_data["content"])
                if new_embedding:
                    self.collection.update(
                        ids=[memory_id],
                        embeddings=[new_embedding],
                        documents=[updated_data["content"]]
                    )
            
            # 清除缓存
            if memory_id in self.memory_cache:
                del self.memory_cache[memory_id]
            
            self.conn.commit()
            logger.debug(f"✅ 记忆更新成功: {memory_id}")
            
        except Exception as e:
            logger.error(f"更新记忆失败: {e}")

    async def extract_and_store(self, text: str, context: str = "") -> bool:
        """智能提取并存储记忆 - 增强身份处理"""
        try:
            logger.debug(f"🔍 分析文本: {text}")
            
            # 1. 检测身份声明
            identity_info = await self.identity_resolver.extract_identity_info(text)
            if identity_info:
                await self.identity_resolver.register_identity(
                    identity_info["real_name"], 
                    identity_info["confidence"]
                )
                logger.info(f"🆔 检测到身份声明: {identity_info['real_name']}")
            
            # 2. 原有的智能信息提取逻辑
            extraction = await self._extract_information(text, context)
            
            if not extraction.should_store_in_memory:
                logger.debug(f"📝 文本不重要，跳过存储: {text}")
                return False
            
            # 2. 生成向量嵌入
            embedding = await self._get_embedding(extraction.summary or text)
            
            # 3. 创建记忆项目
            memory_id = self._generate_memory_id(text)
            memory = MemoryItem(
                id=memory_id,
                content=text,
                memory_type=extraction.memory_type,
                importance=extraction.importance,
                timestamp=time.time(),
                embedding=embedding,
                metadata={
                    "structured_info": extraction.structured_info,
                    "summary": extraction.summary,
                    "keywords": extraction.keywords,
                    "context": context
                }
            )
            
            # 4. 智能去重和合并检测
            existing_memories = await self._search_by_type(memory.memory_type)
            merge_result = await self.memory_merger.check_and_merge_similar_memories(memory, existing_memories)

            if merge_result and merge_result.should_merge:
                # 找到要合并的目标记忆
                target_memory = None
                for existing in existing_memories:
                    if self._is_target_for_merge(memory, existing):
                        target_memory = existing
                        break
                
                if target_memory:
                    # 执行合并
                    merged_data = await self.memory_merger.merge_memories(target_memory, merge_result)
                    if merged_data:
                        # 更新现有记忆
                        await self._update_memory(target_memory.id, merged_data)
                        logger.info(f"✅ 记忆已合并: {merged_data['content']}")
                        return True

            # 5. 传统冲突检测和解决（作为备用）
            await self._handle_conflicts(memory)

            # 6. 存储新记忆
            await self._store_memory(memory)
            
            logger.info(f"✅ 记忆存储成功: {extraction.memory_type} - {extraction.summary}")
            return True
            
        except Exception as e:
            logger.error(f"记忆存储失败: {e}")
            return False
    async def force_store(self, text: str, context: str = "") -> bool:
        """强制存储记忆，跳过重要性判断"""
        try:
            logger.info(f"🔒 强制存储: {text}")
            
            # 生成向量嵌入
            embedding = await self._get_embedding(text)
            
            # 创建记忆项目（设置高重要性）
            memory_id = self._generate_memory_id(text)
            memory = MemoryItem(
                id=memory_id,
                content=text,
                memory_type="user_manual",  # 用户手动记录类型
                importance=0.95,  # 强制高重要性
                timestamp=time.time(),
                embedding=embedding,
                metadata={
                    "summary": text,
                    "keywords": text.split()[:5],
                    "context": context,
                    "force_stored": True
                }
            )
            
            # 存储记忆
            await self._store_memory(memory)
            
            logger.info(f"✅ 强制存储成功: {text}")
            return True
            
        except Exception as e:
            logger.error(f"强制存储失败: {e}")
            return False
    
    async def _extract_information(self, text: str, context: str = "") -> ExtractionResult:
        """使用GPT智能提取信息，并判断是否应加入用户长期信息库"""
        try:
            # 替换整个 prompt 变量
            prompt = f"""
    你是一个智能信息提取助手，专门识别和提取用户的重要个人信息。你的任务是判断用户输入是否包含值得长期记忆的信息。

    请分析以下内容：
    文本: {text}
    上下文: {context}

    **重要提醒：你需要对个人信息保持高度敏感，宁可多存储也不要遗漏重要信息。**

    请严格按以下JSON格式输出：
    {{
    "contains_important_info": true/false,
    "importance": 0.0-1.0,
    "should_store_in_memory": true/false,
    "memory_type": "personal_info/family/work/preference/physical/event/other",
    "structured_info": {{
        "name": "姓名（如果有）",
        "age": "年龄（如果有）",
        "family_size": "家庭人数（如果有）",
        "family_members": "家庭成员（如果有）",
        "height": "身高（如果有）",
        "weight": "体重（如果有）",
        "job": "职业（如果有）",
        "location": "地点（如果有）",
        "education": "教育背景（如果有）",
        "preferences": ["喜好列表"],
        "skills": ["技能列表"],
        "health": "健康状况（如果有）",
        "other": "其他重要信息"
    }},
    "summary": "简洁摘要",
    "keywords": ["关键词1", "关键词2"]
    }}

    **强化判断标准（更加敏感）：**
    - **家庭信息**：importance ≥ 0.95, should_store_in_memory: true
    - "我们家X口人" → family, importance: 0.95
    - "家里有X个人" → family, importance: 0.95
    - "我爸爸/妈妈是..." → family, importance: 0.95
    - "我有X个兄弟/姐妹" → family, importance: 0.95

    - **身份信息**：importance ≥ 0.95, should_store_in_memory: true
    - "我叫X" → personal_info, importance: 0.95
    - "我X岁" → personal_info, importance: 0.95
    - "我身高X" → physical, importance: 0.95
    - "我体重X" → physical, importance: 0.95

    - **居住信息**：importance ≥ 0.9, should_store_in_memory: true
    - "我住在X" → personal_info, importance: 0.9
    - "我家在X" → personal_info, importance: 0.9
    - "我来自X" → personal_info, importance: 0.9

    - **工作信息**：importance ≥ 0.9, should_store_in_memory: true
    - "我是X工程师/医生/老师" → work, importance: 0.9
    - "我在X公司工作" → work, importance: 0.9

    - **教育背景**：importance ≥ 0.85, should_store_in_memory: true
    - "我毕业于X" → personal_info, importance: 0.85
    - "我学的是X专业" → personal_info, importance: 0.85

    - **偏好习惯**：importance ≥ 0.7, should_store_in_memory: true
    - "我喜欢X" → preference, importance: 0.7
    - "我不喜欢X" → preference, importance: 0.7

    - **健康状况**：importance ≥ 0.8, should_store_in_memory: true
    - "我有X病" → personal_info, importance: 0.8
    - "我过敏X" → personal_info, importance: 0.8

    - **技能能力**：importance ≥ 0.75, should_store_in_memory: true
    - "我会X语言/技能" → personal_info, importance: 0.75

    **明确排除（should_store_in_memory: false）：**
    - 纯问候语："你好"、"再见"
    - 纯感叹词："哈哈"、"嗯嗯"
    - 重复的AI回答内容
    - 纯粹的提问："什么是X？"
    - 天气相关的临时信息

    **示例分析：**
    文本："我们家一共有八口人"
    应该输出：
    {{
    "contains_important_info": true,
    "importance": 0.95,
    "should_store_in_memory": true,
    "memory_type": "family",
    "structured_info": {{"family_size": "八口人"}},
    "summary": "家庭成员数量：8人",
    "keywords": ["家庭", "八口人", "家庭成员"]
    }}

    文本："我住在北京"
    应该输出：
    {{
    "contains_important_info": true,
    "importance": 0.9,
    "should_store_in_memory": true,
    "memory_type": "personal_info",
    "structured_info": {{"location": "北京"}},
    "summary": "居住地：北京",
    "keywords": ["居住", "北京", "地点"]
    }}

    示例1：
    文本: 我叫李雷，今年22岁，住在北京，喜欢打篮球
    输出:
    {{
    "contains_important_info": true,
    "importance": 0.92,
    "should_store_in_memory": true,
    "memory_type": "personal_info",
    ...
    }}

    示例2：
    文本: 哈哈哈你好呀
    输出:
    {{
    "contains_important_info": false,
    "importance": 0.0,
    "should_store_in_memory": false,
    "memory_type": "other",
    ...
    }}
    """

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.1
            )

            if response.choices:
                result_text = response.choices[0].message.content.strip()

                try:
                    result_data = json.loads(result_text)

                    return ExtractionResult(
                        is_important=result_data.get("contains_important_info", False),
                        importance=result_data.get("importance", 0.0),
                        should_store_in_memory=result_data.get("should_store_in_memory", False),
                        memory_type=result_data.get("memory_type", "other"),
                        structured_info=result_data.get("structured_info", {}),
                        summary=result_data.get("summary", text),
                        keywords=result_data.get("keywords", [])
                    )

                except json.JSONDecodeError:
                    logger.warning("GPT返回的JSON格式解析失败，尝试备用提取方案。")
                    return self._fallback_extraction(text)

            return self._fallback_extraction(text)

        except Exception as e:
            logger.error(f"GPT信息提取失败: {e}")
            return self._fallback_extraction(text)
    
    def _fallback_extraction(self, text: str) -> ExtractionResult:
        """优化的备用信息提取 - 增强版"""
        import re
        
        importance = 0.3
        memory_type = "other"
        structured_info = {}
        should_store = False
        
        text_lower = text.lower()
        
        # === 1. 家庭信息识别（最高优先级）===
        family_patterns = [
            (r'我们家.*?(\d+).*?口人', 'family_size', 0.95),
            (r'家里.*?(\d+).*?个人', 'family_size', 0.95),
            (r'我们家.*?(\d+).*?人', 'family_size', 0.95),
            (r'家庭.*?(\d+).*?人', 'family_size', 0.95),
        ]
        
        for pattern, field, imp in family_patterns:
            match = re.search(pattern, text)
            if match:
                importance = imp
                memory_type = "family"
                structured_info[field] = match.group(1) + "人"
                should_store = True
                break
        
        # 家庭成员职业
        if any(word in text for word in ['妈妈', '爸爸', '父母', '母亲', '父亲', '兄弟', '姐妹']):
            importance = max(importance, 0.95)
            memory_type = "family"
            should_store = True
        
        # === 2. 个人身份信息（高优先级）===
        if not should_store:
            # 姓名
            name_patterns = [
                (r'我叫(.{1,4})', 'name', 0.95),
                (r'我的名字.*?(.{1,4})', 'name', 0.95),
                (r'我是(.{1,4})', 'name', 0.9),
            ]
            
            for pattern, field, imp in name_patterns:
                match = re.search(pattern, text)
                if match:
                    importance = imp
                    memory_type = "personal_info"
                    structured_info[field] = match.group(1).strip()
                    should_store = True
                    break
        
        # === 3. 身体特征信息（高优先级）===
        if not should_store:
            # 身高
            height_patterns = [
                (r'我.*?身高.*?(\d+\.?\d*)\s*(?:米|m)', 'height', 0.95),
                (r'我.*?(\d+)\s*(?:cm|厘米|公分)', 'height', 0.95),
                (r'(\d+\.?\d*)\s*(?:米|m)', 'height', 0.9),
                (r'(\d+)\s*(?:cm|厘米|公分)', 'height', 0.9),
            ]
            
            for pattern, field, imp in height_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    importance = imp
                    memory_type = "physical"
                    height_val = match.group(1)
                    if '米' in text_lower or 'm' in text_lower:
                        structured_info[field] = f"{height_val}米"
                    else:
                        structured_info[field] = f"{height_val}cm"
                    should_store = True
                    break
            
            # 体重
            if not should_store:
                weight_patterns = [
                    (r'我.*?体重.*?(\d+\.?\d*)\s*(?:斤|kg|公斤)', 'weight', 0.95),
                    (r'我.*?(\d+\.?\d*)\s*(?:斤|kg|公斤)', 'weight', 0.9),
                ]
                
                for pattern, field, imp in weight_patterns:
                    match = re.search(pattern, text_lower)
                    if match:
                        importance = imp
                        memory_type = "physical"
                        weight_val = match.group(1)
                        if '斤' in text_lower:
                            structured_info[field] = f"{weight_val}斤"
                        else:
                            structured_info[field] = f"{weight_val}kg"
                        should_store = True
                        break
            
            # 年龄
            if not should_store:
                age_patterns = [
                    (r'我.*?(\d+)\s*岁', 'age', 0.95),
                    (r'我.*?年龄.*?(\d+)', 'age', 0.95),
                    (r'(\d+)\s*岁', 'age', 0.9),
                ]
                
                for pattern, field, imp in age_patterns:
                    match = re.search(pattern, text)
                    if match:
                        importance = imp
                        memory_type = "personal_info"
                        structured_info[field] = match.group(1) + "岁"
                        should_store = True
                        break
        
        # === 4. 居住信息===
        if not should_store:
            location_patterns = [
                (r'我住在(.{1,10})', 'location', 0.9),
                (r'我家在(.{1,10})', 'location', 0.9),
                (r'我来自(.{1,10})', 'location', 0.85),
            ]
            
            for pattern, field, imp in location_patterns:
                match = re.search(pattern, text)
                if match:
                    importance = imp
                    memory_type = "personal_info"
                    structured_info[field] = match.group(1).strip()
                    should_store = True
                    break
        
        # === 5. 工作信息===
        if not should_store:
            job_keywords = ['工作', '职业', '公司', '上班', '工程师', '医生', '老师', '程序员', '经理', '律师']
            if any(word in text for word in job_keywords):
                importance = 0.9
                memory_type = "work"
                should_store = True
        
        # === 6. 教育信息===
        if not should_store:
            education_keywords = ['毕业', '大学', '学校', '专业', '学历', '博士', '硕士', '本科']
            if any(word in text for word in education_keywords):
                importance = 0.85
                memory_type = "personal_info"
                should_store = True
        
        # === 7. 偏好信息===
        if not should_store:
            preference_keywords = ['我喜欢', '我爱', '喜好', '兴趣', '爱好', '我不喜欢', '我讨厌']
            if any(word in text for word in preference_keywords):
                importance = 0.7
                memory_type = "preference"
                should_store = True
        
        # === 8. 健康信息===
        if not should_store:
            health_keywords = ['生病', '健康', '过敏', '病史', '症状', '医院', '药物']
            if any(word in text for word in health_keywords):
                importance = 0.8
                memory_type = "personal_info"
                should_store = True
        
        # === 9. 技能信息===
        if not should_store:
            skill_keywords = ['会', '擅长', '精通', '掌握', '学过', '能力', '技能']
            if any(word in text for word in skill_keywords):
                importance = 0.75
                memory_type = "personal_info"
                should_store = True
        
        # === 10. 排除明显无意义内容===
        meaningless_patterns = [
            r'^(你好|再见|谢谢|不客气|对不起)$',
            r'^(哈哈|嗯嗯|嗯|哦|啊)$',
            r'^(什么|怎么|为什么|哪里|谁).*\?$',
            r'^(帮我|告诉我|请问).*',
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, text.strip()):
                should_store = False
                importance = 0.1
                break
        
        # === 11. 包含"我"的短句通常是重要的===
        if not should_store and '我' in text and len(text) >= 4:
            importance = max(importance, 0.6)
            memory_type = "personal_info"
            should_store = True
        
        return ExtractionResult(
            is_important=should_store,
            importance=importance,
            should_store_in_memory=should_store,  # 新增字段
            memory_type=memory_type,
            structured_info=structured_info,
            summary=text,
            keywords=text.split()[:5]
        )
    
    async def _get_embedding(self, text: str) -> List[float]:
        """获取文本向量嵌入"""
        try:
            # 检查缓存
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]
            
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            
            if response.data:
                embedding = response.data[0].embedding
                
                # 缓存结果
                self.embedding_cache[text_hash] = embedding
                
                return embedding
            
            return []
            
        except Exception as e:
            logger.error(f"获取向量嵌入失败: {e}")
            return []
    
    def _generate_memory_id(self, text: str) -> str:
        """生成记忆ID"""
        timestamp = str(int(time.time() * 1000))
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{timestamp}_{text_hash}"
    
    async def _handle_conflicts(self, memory: MemoryItem):
        """处理记忆冲突"""
        try:
            structured_info = memory.metadata.get("structured_info", {})
            
            # 检查姓名冲突
            if "name" in structured_info:
                await self._resolve_name_conflict(memory)
            
            # 检查家庭信息冲突
            if "family" in structured_info:
                await self._resolve_family_conflict(memory)
            
        except Exception as e:
            logger.debug(f"处理记忆冲突失败: {e}")  # 改为debug级别
    
    async def _resolve_family_conflict(self, memory: MemoryItem):
        """解决家庭信息冲突"""
        try:
            # 搜索现有的家庭记忆
            existing_memories = await self._search_by_type("family")
            
            new_family_info = memory.metadata["structured_info"].get("family", {})
            if not new_family_info:
                return
            
            for existing in existing_memories:
                existing_family = existing.metadata.get("structured_info", {}).get("family", {})
                
                # 检查父母职业冲突
                for parent in ["mother", "father"]:
                    new_job = new_family_info.get(parent)
                    old_job = existing_family.get(parent)
                    
                    if new_job and old_job and new_job != old_job:
                        logger.warning(f"🚨 {parent}职业冲突: 新'{new_job}' vs 旧'{old_job}'")
                        
                        # 使用GPT解决冲突
                        resolution = await self._gpt_resolve_conflict(
                            f"{parent}职业", new_job, old_job, memory, existing
                        )
                        
                        if resolution == "keep_new":
                            await self._mark_memory_outdated(existing.id)
                        elif resolution == "keep_old":
                            memory.importance *= 0.5  # 降低新记忆重要性
                
        except Exception as e:
            logger.error(f"解决家庭信息冲突失败: {e}")
    
    async def _resolve_name_conflict(self, memory: MemoryItem):
        """解决姓名冲突"""
        try:
            # 搜索现有的姓名记忆
            existing_memories = await self._search_by_type("personal_info")
            
            new_name = memory.metadata["structured_info"].get("name")
            if not new_name:
                return
            
            for existing in existing_memories:
                existing_name = existing.metadata.get("structured_info", {}).get("name")
                if existing_name and existing_name != new_name:
                    logger.warning(f"🚨 姓名冲突: 新'{new_name}' vs 旧'{existing_name}'")
                    
                    # 使用GPT解决冲突
                    resolution = await self._gpt_resolve_conflict(
                        "姓名", new_name, existing_name, memory, existing
                    )
                    
                    if resolution == "keep_new":
                        await self._mark_memory_outdated(existing.id)
                    elif resolution == "keep_old":
                        memory.importance *= 0.5  # 降低新记忆重要性
                    
        except Exception as e:
            logger.error(f"解决姓名冲突失败: {e}")
    
    async def _gpt_resolve_conflict(self, conflict_type: str, new_value: str, 
                                  old_value: str, new_memory: MemoryItem, 
                                  old_memory: MemoryItem) -> str:
        """使用GPT解决冲突"""
        try:
            prompt = f"""存在{conflict_type}冲突，需要你判断哪个信息更可信：

新信息: {new_value}
- 来源: {new_memory.content}
- 时间: {datetime.fromtimestamp(new_memory.timestamp)}
- 重要性: {new_memory.importance}

旧信息: {old_value}
- 来源: {old_memory.content}
- 时间: {datetime.fromtimestamp(old_memory.timestamp)}
- 重要性: {old_memory.importance}

请选择: "keep_new" 或 "keep_old" 或 "merge"
理由:"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            if response.choices:
                result = response.choices[0].message.content.strip().lower()
                if "keep_new" in result:
                    return "keep_new"
                elif "keep_old" in result:
                    return "keep_old"
                else:
                    return "merge"
            
            return "keep_new"  # 默认保留新信息
            
        except Exception as e:
            logger.error(f"GPT冲突解决失败: {e}")
            return "keep_new"
    
    async def _store_memory(self, memory: MemoryItem):
        """存储记忆到数据库"""
        try:
            # 存储到SQLite
            self.conn.execute("""
                INSERT OR REPLACE INTO memories 
                (id, content, memory_type, importance, timestamp, summary, keywords, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.content,
                memory.memory_type,
                memory.importance,
                memory.timestamp,
                memory.metadata.get("summary", ""),
                json.dumps(memory.metadata.get("keywords", [])),
                json.dumps(memory.metadata)
            ))
            
            # 存储向量
            if CHROMADB_AVAILABLE and memory.embedding:
                self.collection.upsert(
                    ids=[memory.id],
                    embeddings=[memory.embedding],
                    metadatas=[{
                        "memory_type": memory.memory_type,
                        "importance": memory.importance,
                        "timestamp": memory.timestamp,
                        "tier": "medium"  # 默认中期，稍后会更新
                    }],
                    documents=[memory.content]
                )
            else:
                # 简单向量存储
                self.vector_storage["ids"].append(memory.id)
                self.vector_storage["embeddings"].append(memory.embedding)
                self._save_vector_storage()
            # 分配记忆层级
            try:
                assigned_tier = await self.tier_manager.assign_memory_tier(
                    memory.id, 
                    memory.memory_type, 
                    memory.metadata.get("structured_info", {}), 
                    memory.importance
                )
                
                # 更新向量数据库中的层级信息
                if CHROMADB_AVAILABLE:
                    try:
                        self.collection.update(
                            ids=[memory.id],
                            metadatas=[{
                                "memory_type": memory.memory_type,
                                "importance": memory.importance,
                                "timestamp": memory.timestamp,
                                "tier": assigned_tier.value  # 使用 assigned_tier
                            }]
                        )
                    except Exception as e:
                        logger.debug(f"更新向量层级信息失败: {e}")
                        
            except Exception as e:
                logger.debug(f"分配记忆层级失败: {e}")
            # 缓存
            self.memory_cache[memory.id] = memory
            # 添加到记忆图谱
            try:
                context_info = memory.metadata.get("context", "")  # 从metadata获取context
                asyncio.create_task(self.graph_manager.add_memory_advanced(
                    memory.id, memory.content, memory.memory_type, memory.importance,
                    context_info
                ))
            except Exception as e:
                logger.debug(f"添加到增强图谱失败: {e}")
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"存储记忆失败: {e}")
    
    async def smart_search_and_respond(self, question: str, context: str = "") -> Tuple[str, List[str]]:
        """智能搜索并生成回答，返回(回答, 使用的记忆ID列表) - 图谱增强版"""
        
        """智能搜索并生成回答 - 增强身份解析"""
        try:
            logger.info(f"🔍 智能搜索问题: {question}")
            
            # 1. 尝试身份增强搜索
            try:
                return await self.identity_resolver.enhance_search_with_identity(question, self)
            except Exception as e:
                logger.debug(f"身份增强搜索失败，使用普通搜索: {e}")
            
            # 1. 向量搜索相关记忆
            relevant_memories = await self._vector_search(question)
            
            if not relevant_memories:
                logger.info("📭 未找到相关记忆")
                return "", []
            
            # 收集使用的记忆ID
            used_memory_ids = [memory.id for memory in relevant_memories]
            
            # 2. 尝试增强图谱搜索
            try:
                graph_response, graph_memory_ids = await self.graph_manager.enhanced_search_and_respond(question, context)
                
                if graph_response and "抱歉" not in graph_response:
                    # 图谱成功找到关联答案
                    logger.info(f"✅ 增强图谱回答: {graph_response}")
                    used_memory_ids.extend(graph_memory_ids)
                    
                    # 记录访问
                    for memory_id in set(used_memory_ids):
                        asyncio.create_task(self.importance_adjuster.track_memory_access(
                            memory_id, context_relevant=True
                        ))
                    
                    return graph_response, list(set(used_memory_ids))

            except Exception as e:
                logger.debug(f"增强图谱搜索失败，回退到普通搜索: {e}")
            
            # 3. 普通智能回答生成（备用方案）
            response = await self._generate_smart_response(question, relevant_memories, context)
            
            # 记录访问
            for memory in relevant_memories:
                asyncio.create_task(self.importance_adjuster.track_memory_access(
                    memory.id, context_relevant=True
                ))
            
            logger.info(f"✅ 普通智能回答: {response}")
            return response, used_memory_ids
            
        except Exception as e:
            logger.error(f"智能搜索回答失败: {e}")
            return "", []
    
    async def _vector_search(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """向量相似度搜索"""
        try:
            query_embedding = await self._get_embedding(query)
            if not query_embedding:
                return []
            
            if CHROMADB_AVAILABLE:
                try:
                    # 简化分层搜索，确保能找到结果
                    all_results = []
                    
                    # 1. 先尝试分层搜索
                    for tier in ["core", "medium", "short", "temporary"]:
                        try:
                            tier_results = self.collection.query(
                                query_embeddings=[query_embedding],
                                n_results=top_k//2,
                                where={"tier": tier}
                            )
                            if tier_results["ids"] and tier_results["ids"][0]:
                                all_results.extend(tier_results["ids"][0])
                                if len(all_results) >= top_k//2:  # 找到足够结果就停止
                                    break
                        except Exception as e:
                            logger.debug(f"搜索 {tier} 层失败: {e}")
                            continue
                    
                    # 2. 如果分层搜索结果不足，使用普通搜索补充
                    if len(all_results) < top_k//2:
                        logger.debug("分层搜索结果不足，补充普通搜索")
                        try:
                            fallback_results = self.collection.query(
                                query_embeddings=[query_embedding],
                                n_results=top_k
                            )
                            if fallback_results["ids"] and fallback_results["ids"][0]:
                                # 合并结果，去重
                                new_ids = [id for id in fallback_results["ids"][0] if id not in all_results]
                                all_results.extend(new_ids)
                        except Exception as e:
                            logger.debug(f"普通搜索也失败: {e}")
                    
                    memory_ids = all_results[:top_k]
                    
                except Exception as e:
                    logger.debug(f"分层搜索完全失败，使用简单搜索: {e}")
                    # 最终回退
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k
                    )
                    memory_ids = results["ids"][0] if results["ids"] and results["ids"][0] else []
            else:
                # 简单向量搜索（保持原逻辑）
                memory_ids = self._simple_vector_search(query_embedding, top_k)
            
            # 从数据库获取完整记忆
            memories = []
            for memory_id in memory_ids:
                memory = await self._get_memory_by_id(memory_id)
                if memory:
                    memories.append(memory)
                    # 记录访问（异步执行，不阻塞主流程）
                    asyncio.create_task(self.importance_adjuster.track_memory_access(
                        memory_id, context_relevant=True
                    ))
            
            logger.debug(f"🔍 向量搜索找到 {len(memories)} 条相关记忆")
            return memories
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
        
    async def get_tier_analytics(self) -> Dict[str, Any]:
        """获取分层存储分析"""
        try:
            # 基础分层统计
            tier_distribution = await self.tier_manager.get_tier_distribution()
            
            # 层级性能分析
            tier_performance = await self._analyze_tier_performance()
            
            return {
                "tier_distribution": tier_distribution,
                "tier_performance": tier_performance,
                "analysis_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"获取分层分析失败: {e}")
            return {"error": str(e)}

    async def _analyze_tier_performance(self) -> Dict[str, Any]:
        """分析各层级性能"""
        try:
            performance = {}
            
            for tier in ["core", "medium", "short", "temporary"]:
                cursor = self.conn.execute("""
                    SELECT 
                        COUNT(DISTINCT m.id) as memory_count,
                        AVG(m.importance) as avg_importance,
                        AVG(u.access_count) as avg_access,
                        COUNT(CASE WHEN u.access_count > 5 THEN 1 END) as high_access_count
                    FROM memories m
                    LEFT JOIN memory_usage u ON m.id = u.memory_id
                    WHERE m.tier = ?
                """, (tier,))
                
                result = cursor.fetchone()
                if result:
                    performance[tier] = {
                        "memory_count": result[0],
                        "avg_importance": round(result[1] or 0, 3),
                        "avg_access": round(result[2] or 0, 2),
                        "high_access_ratio": round((result[3] or 0) / max(result[0], 1), 3)
                    }
            
            return performance
            
        except Exception as e:
            logger.error(f"分析层级性能失败: {e}")
            return {}
    
    def _simple_vector_search(self, query_embedding: List[float], top_k: int) -> List[str]:
        """简单向量搜索"""
        try:
            if not self.vector_storage["embeddings"]:
                return []
            
            # 计算余弦相似度
            query_vec = np.array(query_embedding)
            similarities = []
            
            for i, embedding in enumerate(self.vector_storage["embeddings"]):
                if embedding:
                    emb_vec = np.array(embedding)
                    similarity = np.dot(query_vec, emb_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(emb_vec)
                    )
                    similarities.append((similarity, self.vector_storage["ids"][i]))
            
            # 排序并返回top_k
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            return [item[1] for item in similarities[:top_k] 
                   if item[0] >= self.similarity_threshold]
            
        except Exception as e:
            logger.error(f"简单向量搜索失败: {e}")
            return []
    
    async def _get_memory_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        """根据ID获取记忆"""
        try:
            # 先检查缓存
            if memory_id in self.memory_cache:
                return self.memory_cache[memory_id]
            
            # 从数据库查询
            cursor = self.conn.execute("""
                SELECT id, content, memory_type, importance, timestamp, metadata
                FROM memories WHERE id = ?
            """, (memory_id,))
            
            row = cursor.fetchone()
            if row:
                memory = MemoryItem(
                    id=row[0],
                    content=row[1],
                    memory_type=row[2],
                    importance=row[3],
                    timestamp=row[4],
                    metadata=json.loads(row[5]) if row[5] else {}
                )
                
                # 缓存
                self.memory_cache[memory_id] = memory
                return memory
            
            return None
            
        except Exception as e:
            logger.error(f"获取记忆失败: {e}")
            return None
    
    async def _generate_smart_response(self, question: str, memories: List[MemoryItem], 
                                     context: str = "") -> str:
        """生成智能回答"""
        try:
            # 构建记忆上下文
            memory_context = self._build_memory_context(memories)
            
            prompt = f"""基于以下个人记忆信息，回答用户问题。请生成自然、个性化的回答。

问题: {question}

个人记忆:
{memory_context}

对话上下文: {context}

要求:
1. 回答要自然、口语化
2. 基于记忆中的真实信息回答
3. 如果记忆中有相关信息，直接使用
4. 如果没有相关信息，简单说明没有相关记录
5. 1-2句话简洁回答
6. 用第一人称回答（我、我的）

请直接给出回答："""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            
            if response.choices:
                answer = response.choices[0].message.content.strip()
                return answer
            
            return ""
            
        except Exception as e:
            logger.error(f"生成智能回答失败: {e}")
            return ""
    
    def _build_memory_context(self, memories: List[MemoryItem]) -> str:
        """构建记忆上下文"""
        context_lines = []
        
        # 按重要性和时间排序
        sorted_memories = sorted(memories, 
                               key=lambda x: (x.importance, x.timestamp), 
                               reverse=True)
        
        for memory in sorted_memories[:10]:  # 最多10条记忆
            summary = memory.metadata.get("summary", memory.content)
            structured = memory.metadata.get("structured_info", {})
            
            # 优先显示结构化信息
            if structured:
                info_parts = []
                for key, value in structured.items():
                    if value:
                        info_parts.append(f"{key}: {value}")
                
                if info_parts:
                    context_lines.append(f"- {', '.join(info_parts)} (重要性: {memory.importance:.2f})")
                    continue
            
            # 否则显示摘要
            context_lines.append(f"- {summary} (重要性: {memory.importance:.2f})")
        
        return '\n'.join(context_lines) if context_lines else "暂无相关记忆"
    
    async def _search_by_type(self, memory_type: str) -> List[MemoryItem]:
        """按类型搜索记忆"""
        try:
            cursor = self.conn.execute("""
                SELECT id FROM memories WHERE memory_type = ? 
                ORDER BY importance DESC, timestamp DESC
            """, (memory_type,))
            
            memory_ids = [row[0] for row in cursor.fetchall()]
            
            memories = []
            for memory_id in memory_ids:
                memory = await self._get_memory_by_id(memory_id)
                if memory:
                    memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"按类型搜索失败: {e}")
            return []
    
    async def _mark_memory_outdated(self, memory_id: str):
        """标记记忆为过期"""
        try:
            self.conn.execute("""
                UPDATE memories SET importance = importance * 0.1
                WHERE id = ?
            """, (memory_id,))
            
            self.conn.commit()
            
            # 清除缓存
            if memory_id in self.memory_cache:
                del self.memory_cache[memory_id]
            
        except Exception as e:
            logger.error(f"标记记忆过期失败: {e}")
    
    def _save_vector_storage(self):
        """保存向量存储"""
        try:
            with open(self.vector_file, 'wb') as f:
                pickle.dump(self.vector_storage, f)
        except Exception as e:
            logger.error(f"保存向量存储失败: {e}")
    
    async def cleanup_old_memories(self, days: int = 30):
        """清理旧记忆"""
        current_time = time.time()
        try:
            cutoff_time = time.time() - (days * 24 * 3600)
            
            cursor = self.conn.execute("""
                DELETE FROM memories 
                WHERE timestamp < ? AND importance < ?
            """, (cutoff_time, 0.5))
            
            deleted_count = cursor.rowcount
            self.conn.commit()
            
            logger.info(f"🧹 清理了 {deleted_count} 条旧记忆")
            
        except Exception as e:
            logger.error(f"清理旧记忆失败: {e}")
        if current_time % 60 < 10:  # 6小时 = 21600秒
            try:
                if hasattr(self.openai_memory_manager, 'importance_adjuster'):
                    adjustments = await self.openai_memory_manager.importance_adjuster.adjust_memory_importance_by_usage()
                    if adjustments:
                        logger.info(f"📊 重要性调整: {len(adjustments)} 条记忆")
            except Exception as e:
                logger.debug(f"重要性调整失败: {e}")
        if current_time % 60 < 10:  # 12小时 = 43200秒
            try:
                if hasattr(self.openai_memory_manager, 'tier_manager'):
                    transitions = await self.openai_memory_manager.tier_manager.review_and_adjust_tiers()
                    if transitions:
                        logger.info(f"📂 层级调整: {len(transitions)} 条记忆转换")
            except Exception as e:
                logger.debug(f"层级审查失败: {e}")

    async def get_memory_analytics(self) -> Dict[str, Any]:
        """获取记忆分析数据"""
        try:
            # 基础统计
            basic_stats = await self.get_memory_stats()
            
            # 重要性趋势
            importance_trends = await self.importance_adjuster.get_importance_trends()
            
            # 热门记忆
            top_memories = await self.importance_adjuster.get_top_accessed_memories()
            
            return {
                "basic_stats": basic_stats,
                "importance_trends": importance_trends,
                "top_memories": top_memories,
                "analysis_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"获取记忆分析失败: {e}")
            return {"error": str(e)}
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        try:
            cursor = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_memories,
                    AVG(importance) as avg_importance,
                    memory_type,
                    COUNT(*) as type_count
                FROM memories 
                GROUP BY memory_type
            """)
            
            stats = {"total_memories": 0, "by_type": {}}
            
            for row in cursor.fetchall():
                if row[2]:  # memory_type
                    stats["by_type"][row[2]] = row[3]
                    if stats["total_memories"] == 0:
                        stats["total_memories"] = row[0]
                        stats["avg_importance"] = row[1]
            
            return stats
            
        except Exception as e:
            logger.error(f"获取记忆统计失败: {e}")
            return {"total_memories": 0, "by_type": {}}
    
    def close(self):
        """关闭连接"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
            
            # 保存向量存储
            if not CHROMADB_AVAILABLE:
                self._save_vector_storage()
            
            logger.info("✅ OpenAI记忆管理器已关闭")
            
        except Exception as e:
            logger.error(f"关闭记忆管理器失败: {e}")