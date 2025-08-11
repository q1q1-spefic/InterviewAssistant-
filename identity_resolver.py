"""
身份解析器
专门处理"我"="具体姓名"的关联推理
"""
import json
import sqlite3
from typing import Dict, List, Optional, Tuple
from loguru import logger

class IdentityResolver:
    """身份解析器"""
    
    def __init__(self, db_connection, openai_client):
        self.conn = db_connection
        self.openai_client = openai_client
        
        # 用户身份映射
        self.user_identities = {}  # {"user_name": "real_name"}
        
        self._init_identity_storage()
        self._load_existing_identities()
        
        logger.info("🆔 身份解析器初始化完成")
    
    def _init_identity_storage(self):
        """初始化身份存储"""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS user_identities (
                    user_session TEXT DEFAULT 'default',
                    real_name TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_time REAL NOT NULL,
                    PRIMARY KEY (user_session, real_name)
                )
            """)
            
            self.conn.commit()
            logger.debug("✅ 身份存储表初始化完成")
            
        except Exception as e:
            logger.error(f"身份存储初始化失败: {e}")
    
    def _load_existing_identities(self):
        """加载现有身份映射"""
        try:
            cursor = self.conn.execute("""
                SELECT real_name, confidence FROM user_identities 
                WHERE user_session = 'default'
                ORDER BY confidence DESC, created_time DESC
            """)
            
            for real_name, confidence in cursor.fetchall():
                self.user_identities["我"] = real_name
                logger.debug(f"加载身份映射: 我 -> {real_name} (置信度: {confidence})")
                break  # 只取置信度最高的
            
        except Exception as e:
            logger.error(f"加载身份映射失败: {e}")
    
    async def extract_identity_info(self, text: str) -> Optional[Dict]:
        """提取身份信息"""
        try:
            # 检测身份声明模式
            identity_patterns = [
                r'我叫(.{1,4})',
                r'我的名字.*?(.{1,4})',
                r'我是(.{1,4})',
            ]
            
            import re
            for pattern in identity_patterns:
                match = re.search(pattern, text)
                if match:
                    real_name = match.group(1).strip()
                    
                    # 验证是否为有效人名
                    if await self._validate_person_name(real_name):
                        return {
                            "type": "identity_declaration",
                            "real_name": real_name,
                            "confidence": 0.95,
                            "pattern": pattern
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"身份信息提取失败: {e}")
            return None
    
    async def _validate_person_name(self, name: str) -> bool:
        """验证是否为有效人名"""
        try:
            # 简单规则验证
            if len(name) < 1 or len(name) > 4:
                return False
            
            # 排除明显不是人名的词
            invalid_words = ['我', '你', '他', '她', '它', '那', '这', '的', '是', '在', '有', '没']
            if name in invalid_words:
                return False
            
            # AI验证（可选）
            if len(name) >= 2:
                prompt = f"""
判断"{name}"是否是一个中文人名。

返回JSON格式：
{{"is_person_name": true/false, "confidence": 0.0-1.0}}
"""
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.1
                )
                
                if response.choices:
                    try:
                        result = json.loads(response.choices[0].message.content)
                        return result.get("is_person_name", False)
                    except:
                        pass
            
            return True  # 默认认为是有效人名
            
        except Exception as e:
            logger.error(f"人名验证失败: {e}")
            return True
    
    async def register_identity(self, real_name: str, confidence: float = 0.95) -> bool:
        """注册用户身份"""
        try:
            import time
            
            # 存储到数据库
            self.conn.execute("""
                INSERT OR REPLACE INTO user_identities 
                (user_session, real_name, confidence, created_time)
                VALUES (?, ?, ?, ?)
            """, ("default", real_name, confidence, time.time()))
            
            # 更新内存映射
            self.user_identities["我"] = real_name
            
            self.conn.commit()
            
            logger.info(f"🆔 注册身份映射: 我 -> {real_name} (置信度: {confidence})")
            return True
            
        except Exception as e:
            logger.error(f"身份注册失败: {e}")
            return False
    
    def resolve_identity_reference(self, text: str) -> str:
        """解析身份引用"""
        try:
            # 如果包含"我"，尝试替换为具体姓名
            if "我" in text and "我" in self.user_identities:
                real_name = self.user_identities["我"]
                
                # 智能替换策略
                resolved_text = text
                
                # "我多少岁" -> "刘梅兰多少岁"
                if "我多少岁" in text:
                    resolved_text = text.replace("我多少岁", f"{real_name}多少岁")
                
                # "我的年龄" -> "刘梅兰的年龄"
                elif "我的年龄" in text:
                    resolved_text = text.replace("我的年龄", f"{real_name}的年龄")
                
                # "我几岁" -> "刘梅兰几岁"
                elif "我几岁" in text:
                    resolved_text = text.replace("我几岁", f"{real_name}几岁")
                
                logger.debug(f"身份解析: '{text}' -> '{resolved_text}'")
                return resolved_text
            
            return text
            
        except Exception as e:
            logger.error(f"身份引用解析失败: {e}")
            return text
    
    async def enhance_search_with_identity(self, question: str, memory_manager) -> Tuple[str, List[str]]:
        """使用身份增强搜索 - 修复递归问题"""
        try:
            # 1. 解析身份引用
            resolved_question = self.resolve_identity_reference(question)
            
            if resolved_question != question:
                logger.info(f"🆔 身份解析查询: '{question}' -> '{resolved_question}'")
                
                # 2. 直接调用原始搜索方法，不再调用smart_search_and_respond
                response, memory_ids = await self._direct_memory_search(resolved_question, memory_manager)
                
                if response and "抱歉" not in response:
                    return response, memory_ids
            
            # 3. 如果身份解析搜索失败，进行直接搜索
            return await self._direct_memory_search(question, memory_manager)
            
        except Exception as e:
            logger.error(f"身份增强搜索失败: {e}")
            return await self._direct_memory_search(question, memory_manager)

    async def _direct_memory_search(self, question: str, memory_manager) -> Tuple[str, List[str]]:
        """直接记忆搜索，避免递归"""
        try:
            # 1. 向量搜索相关记忆
            relevant_memories = await memory_manager._vector_search(question)
            
            if not relevant_memories:
                return "", []
            
            # 收集使用的记忆ID
            used_memory_ids = [memory.id for memory in relevant_memories]
            
            # 2. 生成智能回答
            response = await memory_manager._generate_smart_response(question, relevant_memories, "")
            
            # 3. 记录访问
            for memory in relevant_memories:
                try:
                    import asyncio
                    asyncio.create_task(memory_manager.importance_adjuster.track_memory_access(
                        memory.id, context_relevant=True
                    ))
                except:
                    pass
            
            return response, used_memory_ids
            
        except Exception as e:
            logger.error(f"直接记忆搜索失败: {e}")
            return "", []
    
    def get_user_identity(self) -> Optional[str]:
        """获取当前用户身份"""
        return self.user_identities.get("我")
    
    def get_identity_stats(self) -> Dict:
        """获取身份统计"""
        try:
            cursor = self.conn.execute("""
                SELECT COUNT(*), AVG(confidence) FROM user_identities
            """)
            
            count, avg_confidence = cursor.fetchone()
            
            return {
                "total_identities": count or 0,
                "avg_confidence": avg_confidence or 0,
                "current_identity": self.get_user_identity(),
                "identity_mappings": self.user_identities.copy()
            }
            
        except Exception as e:
            logger.error(f"获取身份统计失败: {e}")
            return {}