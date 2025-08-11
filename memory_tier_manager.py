"""
分层记忆存储管理器
根据信息类型和重要性分层存储：核心信息、中期信息、短期信息
"""
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger

class MemoryTier(Enum):
    """记忆层级"""
    CORE = "core"           # 核心信息 - 永久存储
    MEDIUM = "medium"       # 中期信息 - 6个月
    SHORT = "short"         # 短期信息 - 1个月
    TEMPORARY = "temporary" # 临时信息 - 1周

@dataclass
class TierConfig:
    """层级配置"""
    tier: MemoryTier
    retention_days: int
    importance_threshold: float
    access_boost_factor: float
    description: str

@dataclass
class TierTransition:
    """层级转换记录"""
    memory_id: str
    from_tier: MemoryTier
    to_tier: MemoryTier
    reason: str
    timestamp: float
    triggered_by: str

class MemoryTierManager:
    """分层记忆存储管理器"""
    
    def __init__(self, db_connection):
        self.conn = db_connection
        
        # 层级配置
        self.tier_configs = {
            MemoryTier.CORE: TierConfig(
                tier=MemoryTier.CORE,
                retention_days=36500,  # 100年，基本永久
                importance_threshold=0.9,
                access_boost_factor=1.2,
                description="核心身份信息(姓名、职业、家庭)"
            ),
            MemoryTier.MEDIUM: TierConfig(
                tier=MemoryTier.MEDIUM,
                retention_days=180,  # 6个月
                importance_threshold=0.7,
                access_boost_factor=1.1,
                description="重要个人信息(偏好、技能、经历)"
            ),
            MemoryTier.SHORT: TierConfig(
                tier=MemoryTier.SHORT,
                retention_days=30,   # 1个月
                importance_threshold=0.5,
                access_boost_factor=1.05,
                description="一般信息(日常对话、临时偏好)"
            ),
            MemoryTier.TEMPORARY: TierConfig(
                tier=MemoryTier.TEMPORARY,
                retention_days=7,    # 1周
                importance_threshold=0.3,
                access_boost_factor=1.0,
                description="临时信息(当天事件、短期状态)"
            )
        }
        
        # 核心信息类型定义
        self.core_info_types = {
            "name", "age", "job", "family", "education", "location", "height", "weight"
        }
        
        # 中期信息类型定义
        self.medium_info_types = {
            "preferences", "skills", "experiences", "health", "relationships"
        }
        
        self._init_tier_storage()
        logger.info("🏗️ 分层记忆存储管理器初始化完成")
    
    def _init_tier_storage(self):
        """初始化分层存储表"""
        try:
            # 为记忆表添加层级字段
            try:
                self.conn.execute("ALTER TABLE memories ADD COLUMN tier TEXT DEFAULT 'medium'")
                self.conn.execute("ALTER TABLE memories ADD COLUMN tier_assigned_time REAL DEFAULT 0")
                self.conn.execute("ALTER TABLE memories ADD COLUMN last_tier_review REAL DEFAULT 0")
            except:
                # 字段已存在
                pass
            
            # 创建层级转换历史表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS tier_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT,
                    from_tier TEXT,
                    to_tier TEXT,
                    reason TEXT,
                    triggered_by TEXT,
                    timestamp REAL
                )
            """)
            
            # 创建层级统计表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS tier_statistics (
                    date TEXT PRIMARY KEY,
                    core_count INTEGER DEFAULT 0,
                    medium_count INTEGER DEFAULT 0,
                    short_count INTEGER DEFAULT 0,
                    temporary_count INTEGER DEFAULT 0,
                    total_transitions INTEGER DEFAULT 0
                )
            """)
            
            self.conn.commit()
            logger.debug("✅ 分层存储表初始化完成")
            self.conn.execute("""
                UPDATE memories SET tier = 'medium' 
                WHERE tier IS NULL OR tier = ''
            """)

            self.conn.commit()
            logger.debug("✅ 为现有记忆分配了默认层级")
            
        except Exception as e:
            logger.error(f"初始化分层存储表失败: {e}")
    
    async def assign_memory_tier(self, memory_id: str, memory_type: str, 
                                structured_info: Dict, importance: float) -> MemoryTier:
        """为新记忆分配层级"""
        try:
            # 确定层级
            tier = self._determine_initial_tier(memory_type, structured_info, importance)
            
            # 更新数据库
            current_time = time.time()
            self.conn.execute("""
                UPDATE memories SET 
                    tier = ?, 
                    tier_assigned_time = ?,
                    last_tier_review = ?
                WHERE id = ?
            """, (tier.value, current_time, current_time, memory_id))
            
            self.conn.commit()
            
            logger.info(f"📂 记忆分层: {memory_id} → {tier.value} "
                       f"(类型: {memory_type}, 重要性: {importance:.2f})")
            
            # 记录转换（新建记忆）
            await self._record_tier_transition(
                memory_id, None, tier, "初始分配", "system"
            )
            
            return tier
            
        except Exception as e:
            logger.error(f"分配记忆层级失败: {e}")
            return MemoryTier.MEDIUM  # 默认中期
    
    def _determine_initial_tier(self, memory_type: str, structured_info: Dict, importance: float) -> MemoryTier:
        """确定初始层级 - 增强版"""
        try:
            # === 1. 强制核心信息判断（最高优先级）===
            # 1.1 基于重要性强制判断
            if importance >= 0.9:
                return MemoryTier.CORE
            
            # 1.2 基于信息类型强制判断
            core_memory_types = ["personal_info", "family", "work", "physical"]
            if memory_type in core_memory_types:
                return MemoryTier.CORE
            
            # 1.3 基于结构化信息强制判断
            core_fields = ["name", "age", "job", "family", "location", "height", "weight", "family_size", "family_members"]
            if any(field in structured_info and structured_info[field] for field in core_fields):
                return MemoryTier.CORE
            
            # 1.4 基于内容关键词强制判断
            content_str = str(structured_info)
            core_keywords = ["叫", "岁", "住在", "家", "工作", "职业", "身高", "体重", "口人"]
            if any(keyword in content_str for keyword in core_keywords):
                return MemoryTier.CORE
            
            # === 2. 中期信息判断===
            # 2.1 基于重要性判断
            if importance >= 0.7:
                return MemoryTier.MEDIUM
            
            # 2.2 基于信息类型判断
            medium_memory_types = ["preference", "education", "skill"]
            if memory_type in medium_memory_types:
                return MemoryTier.MEDIUM
            
            # 2.3 基于结构化信息判断
            medium_fields = ["preferences", "skills", "education", "health"]
            if any(field in structured_info and structured_info[field] for field in medium_fields):
                return MemoryTier.MEDIUM
            
            # 2.4 基于内容关键词判断
            medium_keywords = ["喜欢", "爱好", "技能", "学历", "毕业", "擅长"]
            if any(keyword in content_str for keyword in medium_keywords):
                return MemoryTier.MEDIUM
            
            # === 3. 临时信息判断===
            # 3.1 基于重要性判断
            if importance < 0.4:
                return MemoryTier.TEMPORARY
            
            # 3.2 基于信息类型判断
            if memory_type == "event":
                return MemoryTier.TEMPORARY
            
            # 3.3 基于时间相关内容判断
            temporal_keywords = ["今天", "刚才", "现在", "早上", "下午", "晚上", "昨天", "明天"]
            if any(keyword in content_str for keyword in temporal_keywords):
                return MemoryTier.TEMPORARY
            
            # === 4. 默认短期信息===
            # 其他情况默认为短期信息
            return MemoryTier.SHORT
            
        except Exception as e:
            logger.debug(f"确定层级失败: {e}")
            return MemoryTier.MEDIUM  # 错误时默认中期
    
    def _is_core_information(self, memory_type: str, structured_info: Dict, importance: float) -> bool:
        """判断是否是核心信息"""
        # 高重要性阈值
        if importance >= self.tier_configs[MemoryTier.CORE].importance_threshold:
            return True
        
        # 核心信息类型
        if memory_type in ["personal_info", "family", "work"]:
            return True
        
        # 检查结构化信息中的核心字段
        core_fields = {"name", "age", "job", "family", "education", "location"}
        if any(field in structured_info for field in core_fields):
            return True
        
        return False
    
    def _is_medium_term_information(self, memory_type: str, structured_info: Dict, importance: float) -> bool:
        """判断是否是中期信息"""
        # 中等重要性
        if importance >= self.tier_configs[MemoryTier.MEDIUM].importance_threshold:
            return True
        
        # 中期信息类型
        if memory_type in ["preference", "physical", "event"]:
            return True
        
        # 中期字段
        medium_fields = {"preferences", "skills", "experiences", "health", "height", "weight"}
        if any(field in structured_info for field in medium_fields):
            return True
        
        return False
    
    def _is_temporary_information(self, memory_type: str, structured_info: Dict, importance: float) -> bool:
        """判断是否是临时信息"""
        # 低重要性
        if importance < self.tier_configs[MemoryTier.TEMPORARY].importance_threshold:
            return True
        
        # 临时信息类型
        if memory_type in ["event", "other"]:
            # 检查是否是当天事件
            current_date = datetime.now().strftime("%Y-%m-%d")
            if "today" in str(structured_info) or current_date in str(structured_info):
                return True
        
        return False
    
    async def review_and_adjust_tiers(self) -> List[TierTransition]:
        """审查并调整记忆层级"""
        try:
            logger.info("🔄 开始审查记忆层级...")
            
            transitions = []
            current_time = time.time()
            
            # 获取需要审查的记忆
            cursor = self.conn.execute("""
                SELECT id, tier, importance, timestamp, tier_assigned_time, last_tier_review
                FROM memories 
                WHERE last_tier_review < ?
                ORDER BY importance DESC
            """, (current_time - 86400,))  # 24小时未审查的
            
            for row in cursor.fetchall():
                memory_id, current_tier_str, importance, created_time, assigned_time, last_review = row
                
                if not current_tier_str:
                    continue
                
                current_tier = MemoryTier(current_tier_str)
                
                # 计算新层级
                new_tier = await self._calculate_adjusted_tier(
                    memory_id, current_tier, importance, created_time, assigned_time
                )
                
                if new_tier != current_tier:
                    # 执行层级转换
                    transition = await self._execute_tier_transition(
                        memory_id, current_tier, new_tier, "定期审查", "system"
                    )
                    if transition:
                        transitions.append(transition)
                
                # 更新审查时间
                self.conn.execute("""
                    UPDATE memories SET last_tier_review = ? WHERE id = ?
                """, (current_time, memory_id))
            
            self.conn.commit()
            
            logger.info(f"✅ 层级审查完成，转换了 {len(transitions)} 条记忆")
            
            # 清理过期记忆
            await self._cleanup_expired_memories()
            
            # 更新统计
            await self._update_tier_statistics()
            
            return transitions
            
        except Exception as e:
            logger.error(f"审查记忆层级失败: {e}")
            return []
    
    async def _calculate_adjusted_tier(self, memory_id: str, current_tier: MemoryTier,
                                     importance: float, created_time: float, 
                                     assigned_time: float) -> MemoryTier:
        """计算调整后的层级"""
        try:
            current_time = time.time()
            age_days = (current_time - created_time) / 86400
            
            # 获取访问统计
            access_stats = await self._get_memory_access_stats(memory_id)
            
            # 基于访问频率调整
            if access_stats["access_count"] > 10:  # 高频访问
                # 考虑升级到更高层级
                if current_tier == MemoryTier.SHORT and importance > 0.6:
                    return MemoryTier.MEDIUM
                elif current_tier == MemoryTier.MEDIUM and importance > 0.8:
                    return MemoryTier.CORE
            
            elif access_stats["access_count"] == 0 and age_days > 30:  # 长期未访问
                # 考虑降级
                if current_tier == MemoryTier.MEDIUM:
                    return MemoryTier.SHORT
                elif current_tier == MemoryTier.SHORT:
                    return MemoryTier.TEMPORARY
            
            # 基于重要性重新评估
            if importance >= 0.9 and current_tier != MemoryTier.CORE:
                return MemoryTier.CORE
            elif importance >= 0.7 and current_tier == MemoryTier.TEMPORARY:
                return MemoryTier.MEDIUM
            elif importance < 0.3 and current_tier != MemoryTier.TEMPORARY:
                return MemoryTier.TEMPORARY
            
            # 基于时间的自然衰减
            tier_age_days = (current_time - assigned_time) / 86400
            
            if current_tier == MemoryTier.MEDIUM and tier_age_days > 90 and access_stats["access_count"] < 3:
                return MemoryTier.SHORT
            elif current_tier == MemoryTier.SHORT and tier_age_days > 30 and access_stats["access_count"] == 0:
                return MemoryTier.TEMPORARY
            
            return current_tier
            
        except Exception as e:
            logger.debug(f"计算调整层级失败: {e}")
            return current_tier
    
    async def _get_memory_access_stats(self, memory_id: str) -> Dict[str, Any]:
        """获取记忆访问统计"""
        try:
            cursor = self.conn.execute("""
                SELECT access_count, last_accessed, context_matches
                FROM memory_usage WHERE memory_id = ?
            """, (memory_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    "access_count": result[0],
                    "last_accessed": result[1],
                    "context_matches": result[2]
                }
            
            return {"access_count": 0, "last_accessed": 0, "context_matches": 0}
            
        except Exception as e:
            logger.debug(f"获取访问统计失败: {e}")
            return {"access_count": 0, "last_accessed": 0, "context_matches": 0}
    
    async def _execute_tier_transition(self, memory_id: str, from_tier: MemoryTier,
                                     to_tier: MemoryTier, reason: str, 
                                     triggered_by: str) -> Optional[TierTransition]:
        """执行层级转换"""
        try:
            current_time = time.time()
            
            # 更新记忆层级
            self.conn.execute("""
                UPDATE memories SET 
                    tier = ?, 
                    tier_assigned_time = ?,
                    last_tier_review = ?
                WHERE id = ?
            """, (to_tier.value, current_time, current_time, memory_id))
            
            # 记录转换
            transition = await self._record_tier_transition(
                memory_id, from_tier, to_tier, reason, triggered_by
            )
            
            self.conn.commit()
            
            logger.info(f"📂➡️📂 层级转换: {memory_id} {from_tier.value} → {to_tier.value} ({reason})")
            
            return transition
            
        except Exception as e:
            logger.error(f"执行层级转换失败: {e}")
            return None
    
    async def _record_tier_transition(self, memory_id: str, from_tier: Optional[MemoryTier],
                                    to_tier: MemoryTier, reason: str, 
                                    triggered_by: str) -> TierTransition:
        """记录层级转换"""
        try:
            current_time = time.time()
            from_tier_str = from_tier.value if from_tier else None
            
            self.conn.execute("""
                INSERT INTO tier_transitions 
                (memory_id, from_tier, to_tier, reason, triggered_by, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (memory_id, from_tier_str, to_tier.value, reason, triggered_by, current_time))
            
            return TierTransition(
                memory_id=memory_id,
                from_tier=from_tier,
                to_tier=to_tier,
                reason=reason,
                timestamp=current_time,
                triggered_by=triggered_by
            )
            
        except Exception as e:
            logger.error(f"记录层级转换失败: {e}")
            return None
    
    async def _cleanup_expired_memories(self):
        """清理过期记忆"""
        try:
            current_time = time.time()
            total_deleted = 0
            
            for tier, config in self.tier_configs.items():
                if tier == MemoryTier.CORE:
                    continue  # 核心信息不删除
                
                cutoff_time = current_time - (config.retention_days * 86400)
                
                # 删除过期记忆
                cursor = self.conn.execute("""
                    SELECT id FROM memories 
                    WHERE tier = ? AND tier_assigned_time < ?
                """, (tier.value, cutoff_time))
                
                expired_ids = [row[0] for row in cursor.fetchall()]
                
                if expired_ids:
                    # 删除记忆
                    placeholders = ','.join(['?'] * len(expired_ids))
                    self.conn.execute(f"""
                        DELETE FROM memories WHERE id IN ({placeholders})
                    """, expired_ids)
                    
                    # 删除相关统计
                    self.conn.execute(f"""
                        DELETE FROM memory_usage WHERE memory_id IN ({placeholders})
                    """, expired_ids)
                    
                    total_deleted += len(expired_ids)
                    
                    logger.info(f"🧹 清理 {tier.value} 层级过期记忆: {len(expired_ids)} 条")
            
            self.conn.commit()
            
            if total_deleted > 0:
                logger.info(f"✅ 总共清理过期记忆: {total_deleted} 条")
            
        except Exception as e:
            logger.error(f"清理过期记忆失败: {e}")
    
    async def _update_tier_statistics(self):
        """更新层级统计"""
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # 统计各层级记忆数量
            tier_counts = {}
            for tier in MemoryTier:
                cursor = self.conn.execute("""
                    SELECT COUNT(*) FROM memories WHERE tier = ?
                """, (tier.value,))
                tier_counts[tier.value] = cursor.fetchone()[0]
            
            # 统计当日转换次数
            today_start = time.time() - 86400
            cursor = self.conn.execute("""
                SELECT COUNT(*) FROM tier_transitions WHERE timestamp > ?
            """, (today_start,))
            transitions_count = cursor.fetchone()[0]
            
            # 插入或更新统计
            self.conn.execute("""
                INSERT OR REPLACE INTO tier_statistics 
                (date, core_count, medium_count, short_count, temporary_count, total_transitions)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                current_date,
                tier_counts.get("core", 0),
                tier_counts.get("medium", 0),
                tier_counts.get("short", 0),
                tier_counts.get("temporary", 0),
                transitions_count
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"更新层级统计失败: {e}")
    
    async def get_tier_distribution(self) -> Dict[str, Any]:
        """获取层级分布统计"""
        try:
            # 当前分布
            current_distribution = {}
            total_memories = 0
            
            for tier in MemoryTier:
                cursor = self.conn.execute("""
                    SELECT COUNT(*), AVG(importance) FROM memories WHERE tier = ?
                """, (tier.value,))
                result = cursor.fetchone()
                count = result[0]
                avg_importance = result[1] or 0
                
                current_distribution[tier.value] = {
                    "count": count,
                    "avg_importance": round(avg_importance, 3),
                    "description": self.tier_configs[tier].description
                }
                total_memories += count
            
            # 最近转换
            cursor = self.conn.execute("""
                SELECT from_tier, to_tier, COUNT(*) 
                FROM tier_transitions 
                WHERE timestamp > ?
                GROUP BY from_tier, to_tier
                ORDER BY COUNT(*) DESC
            """, (time.time() - 86400 * 7,))  # 最近7天
            
            recent_transitions = []
            for row in cursor.fetchall():
                recent_transitions.append({
                    "from": row[0],
                    "to": row[1],
                    "count": row[2]
                })
            
            return {
                "total_memories": total_memories,
                "distribution": current_distribution,
                "recent_transitions": recent_transitions,
                "tier_configs": {tier.value: {
                    "retention_days": config.retention_days,
                    "importance_threshold": config.importance_threshold,
                    "description": config.description
                } for tier, config in self.tier_configs.items()}
            }
            
        except Exception as e:
            logger.error(f"获取层级分布失败: {e}")
            return {"error": str(e)}
    
    async def force_tier_assignment(self, memory_id: str, target_tier: MemoryTier, 
                                  reason: str = "手动指定") -> bool:
        """强制指定记忆层级"""
        try:
            # 获取当前层级
            cursor = self.conn.execute("SELECT tier FROM memories WHERE id = ?", (memory_id,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"记忆不存在: {memory_id}")
                return False
            
            current_tier_str = result[0]
            if current_tier_str:
                current_tier = MemoryTier(current_tier_str)
                
                if current_tier != target_tier:
                    # 执行转换
                    transition = await self._execute_tier_transition(
                        memory_id, current_tier, target_tier, reason, "manual"
                    )
                    return transition is not None
            
            return True
            
        except Exception as e:
            logger.error(f"强制层级指定失败: {e}")
            return False