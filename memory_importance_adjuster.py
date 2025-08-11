"""
记忆重要性动态调整模块
根据使用频率、时间衰减、用户反馈等因素动态调整记忆重要性
"""
import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger

@dataclass
class UsageStats:
    """使用统计"""
    memory_id: str
    access_count: int
    last_accessed: float
    avg_access_interval: float
    context_relevance_score: float
    user_feedback_score: float

@dataclass
class ImportanceAdjustment:
    """重要性调整结果"""
    memory_id: str
    old_importance: float
    new_importance: float
    adjustment_reason: str
    adjustment_factor: float

class MemoryImportanceAdjuster:
    """记忆重要性动态调整器"""
    
    def __init__(self, db_connection):
        self.conn = db_connection
        self.usage_decay_rate = 0.95  # 时间衰减率
        self.access_boost_factor = 1.1  # 访问提升因子
        self.feedback_weight = 0.3  # 用户反馈权重
        self.context_weight = 0.2  # 上下文相关性权重
        self.time_penalty_threshold = 30  # 30天未访问开始衰减
        
        self._init_usage_tracking()
        logger.info("📊 记忆重要性动态调整器初始化完成")
    
    def _init_usage_tracking(self):
        """初始化使用统计表"""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_usage (
                    memory_id TEXT PRIMARY KEY,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL DEFAULT 0,
                    total_access_time REAL DEFAULT 0,
                    context_matches INTEGER DEFAULT 0,
                    positive_feedback INTEGER DEFAULT 0,
                    negative_feedback INTEGER DEFAULT 0,
                    created_time REAL DEFAULT 0
                )
            """)
            
            # 为现有记忆初始化使用统计
            self.conn.execute("""
                INSERT OR IGNORE INTO memory_usage (memory_id, created_time)
                SELECT id, timestamp FROM memories
            """)
            
            self.conn.commit()
            logger.debug("✅ 使用统计表初始化完成")
            
        except Exception as e:
            logger.error(f"初始化使用统计表失败: {e}")
    
    async def track_memory_access(self, memory_id: str, context_relevant: bool = False):
        """记录记忆访问"""
        try:
            current_time = time.time()
            
            # 更新访问统计
            self.conn.execute("""
                UPDATE memory_usage SET 
                    access_count = access_count + 1,
                    last_accessed = ?,
                    total_access_time = total_access_time + ?,
                    context_matches = context_matches + ?
                WHERE memory_id = ?
            """, (
                current_time,
                current_time,
                1 if context_relevant else 0,
                memory_id
            ))
            
            self.conn.commit()
            logger.debug(f"📝 记录访问: {memory_id} (上下文相关: {context_relevant})")
            
        except Exception as e:
            logger.error(f"记录记忆访问失败: {e}")
    
    async def track_user_feedback(self, memory_id: str, is_positive: bool):
        """记录用户反馈"""
        try:
            feedback_field = "positive_feedback" if is_positive else "negative_feedback"
            
            self.conn.execute(f"""
                UPDATE memory_usage SET 
                    {feedback_field} = {feedback_field} + 1
                WHERE memory_id = ?
            """, (memory_id,))
            
            self.conn.commit()
            
            feedback_type = "正面" if is_positive else "负面"
            logger.info(f"👍 记录用户反馈: {memory_id} - {feedback_type}")
            
        except Exception as e:
            logger.error(f"记录用户反馈失败: {e}")
    
    async def adjust_memory_importance_by_usage(self) -> List[ImportanceAdjustment]:
        """根据使用频率调整记忆重要性"""
        try:
            logger.info("🔄 开始动态调整记忆重要性...")
            
            # 获取所有记忆的使用统计
            usage_stats = await self._get_usage_statistics()
            
            adjustments = []
            current_time = time.time()
            
            for stats in usage_stats:
                # 计算新的重要性分数
                adjustment = await self._calculate_importance_adjustment(stats, current_time)
                
                if adjustment and abs(adjustment.adjustment_factor) > 0.05:  # 变化超过5%才调整
                    # 应用调整
                    await self._apply_importance_adjustment(adjustment)
                    adjustments.append(adjustment)
            
            logger.info(f"✅ 重要性调整完成，调整了 {len(adjustments)} 条记忆")
            
            # 记录调整历史
            await self._log_adjustment_history(adjustments)
            
            return adjustments
            
        except Exception as e:
            logger.error(f"动态调整记忆重要性失败: {e}")
            return []
    
    async def _get_usage_statistics(self) -> List[UsageStats]:
        """获取使用统计"""
        try:
            cursor = self.conn.execute("""
                SELECT 
                    u.memory_id,
                    u.access_count,
                    u.last_accessed,
                    u.total_access_time,
                    u.context_matches,
                    u.positive_feedback,
                    u.negative_feedback,
                    u.created_time,
                    m.importance
                FROM memory_usage u
                JOIN memories m ON u.memory_id = m.id
                ORDER BY u.access_count DESC
            """)
            
            stats_list = []
            
            for row in cursor.fetchall():
                memory_id, access_count, last_accessed, total_access_time, \
                context_matches, positive_feedback, negative_feedback, created_time, importance = row
                
                # 计算平均访问间隔
                if access_count > 1 and total_access_time > 0:
                    avg_interval = (last_accessed - created_time) / max(access_count - 1, 1)
                else:
                    avg_interval = float('inf')
                
                # 计算上下文相关性分数
                context_score = context_matches / max(access_count, 1) if access_count > 0 else 0
                
                # 计算用户反馈分数
                total_feedback = positive_feedback + negative_feedback
                if total_feedback > 0:
                    feedback_score = (positive_feedback - negative_feedback) / total_feedback
                else:
                    feedback_score = 0.0
                
                stats = UsageStats(
                    memory_id=memory_id,
                    access_count=access_count,
                    last_accessed=last_accessed,
                    avg_access_interval=avg_interval,
                    context_relevance_score=context_score,
                    user_feedback_score=feedback_score
                )
                
                stats_list.append(stats)
            
            return stats_list
            
        except Exception as e:
            logger.error(f"获取使用统计失败: {e}")
            return []
    
    async def _calculate_importance_adjustment(self, stats: UsageStats, current_time: float) -> Optional[ImportanceAdjustment]:
        """计算重要性调整"""
        try:
            # 获取当前重要性
            cursor = self.conn.execute("SELECT importance FROM memories WHERE id = ?", (stats.memory_id,))
            result = cursor.fetchone()
            if not result:
                return None
            
            current_importance = result[0]
            new_importance = current_importance
            adjustment_reasons = []
            
            # 1. 访问频率调整
            if stats.access_count > 5:  # 高频访问
                frequency_boost = min(0.2, stats.access_count * 0.02)
                new_importance += frequency_boost
                adjustment_reasons.append(f"高频访问({stats.access_count}次)")
            elif stats.access_count == 0:  # 从未访问
                days_since_creation = (current_time - stats.last_accessed) / 86400
                if days_since_creation > self.time_penalty_threshold:
                    time_penalty = min(0.3, (days_since_creation - self.time_penalty_threshold) * 0.01)
                    new_importance -= time_penalty
                    adjustment_reasons.append(f"长期未访问({days_since_creation:.0f}天)")
            
            # 2. 时间衰减
            if stats.last_accessed > 0:
                days_since_access = (current_time - stats.last_accessed) / 86400
                if days_since_access > self.time_penalty_threshold:
                    decay_factor = pow(self.usage_decay_rate, days_since_access - self.time_penalty_threshold)
                    time_decay = current_importance * (1 - decay_factor) * 0.5
                    new_importance -= time_decay
                    adjustment_reasons.append(f"时间衰减({days_since_access:.0f}天未访问)")
            
            # 3. 上下文相关性调整
            if stats.context_relevance_score > 0.7:
                context_boost = stats.context_relevance_score * self.context_weight
                new_importance += context_boost
                adjustment_reasons.append(f"高上下文相关性({stats.context_relevance_score:.2f})")
            elif stats.context_relevance_score < 0.3 and stats.access_count > 3:
                context_penalty = (0.3 - stats.context_relevance_score) * self.context_weight
                new_importance -= context_penalty
                adjustment_reasons.append(f"低上下文相关性({stats.context_relevance_score:.2f})")
            
            # 4. 用户反馈调整
            if abs(stats.user_feedback_score) > 0.1:
                feedback_adjustment = stats.user_feedback_score * self.feedback_weight
                new_importance += feedback_adjustment
                feedback_type = "正面" if stats.user_feedback_score > 0 else "负面"
                adjustment_reasons.append(f"{feedback_type}反馈({stats.user_feedback_score:.2f})")
            
            # 确保重要性在合理范围内
            new_importance = max(0.1, min(1.0, new_importance))
            
            # 计算调整因子
            adjustment_factor = (new_importance - current_importance) / current_importance if current_importance > 0 else 0
            
            if abs(adjustment_factor) > 0.05:  # 变化超过5%
                return ImportanceAdjustment(
                    memory_id=stats.memory_id,
                    old_importance=current_importance,
                    new_importance=new_importance,
                    adjustment_reason=" | ".join(adjustment_reasons) if adjustment_reasons else "常规调整",
                    adjustment_factor=adjustment_factor
                )
            
            return None
            
        except Exception as e:
            logger.error(f"计算重要性调整失败: {e}")
            return None
    
    async def _apply_importance_adjustment(self, adjustment: ImportanceAdjustment):
        """应用重要性调整"""
        try:
            # 更新数据库中的重要性
            self.conn.execute("""
                UPDATE memories SET importance = ? WHERE id = ?
            """, (adjustment.new_importance, adjustment.memory_id))
            
            self.conn.commit()
            
            # 记录调整日志
            direction = "提升" if adjustment.adjustment_factor > 0 else "降低"
            logger.info(f"📈 重要性{direction}: {adjustment.memory_id} "
                       f"{adjustment.old_importance:.3f} → {adjustment.new_importance:.3f} "
                       f"({adjustment.adjustment_factor:+.1%}) - {adjustment.adjustment_reason}")
            
        except Exception as e:
            logger.error(f"应用重要性调整失败: {e}")
    
    async def _log_adjustment_history(self, adjustments: List[ImportanceAdjustment]):
        """记录调整历史"""
        try:
            # 创建调整历史表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS importance_adjustment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT,
                    old_importance REAL,
                    new_importance REAL,
                    adjustment_factor REAL,
                    reason TEXT,
                    timestamp REAL
                )
            """)
            
            # 插入调整记录
            for adj in adjustments:
                self.conn.execute("""
                    INSERT INTO importance_adjustment_history 
                    (memory_id, old_importance, new_importance, adjustment_factor, reason, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    adj.memory_id,
                    adj.old_importance,
                    adj.new_importance,
                    adj.adjustment_factor,
                    adj.adjustment_reason,
                    time.time()
                ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"记录调整历史失败: {e}")
    
    async def get_importance_trends(self, days: int = 30) -> Dict[str, Any]:
        """获取重要性变化趋势"""
        try:
            cutoff_time = time.time() - (days * 86400)
            
            cursor = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_adjustments,
                    AVG(adjustment_factor) as avg_adjustment,
                    COUNT(CASE WHEN adjustment_factor > 0 THEN 1 END) as positive_adjustments,
                    COUNT(CASE WHEN adjustment_factor < 0 THEN 1 END) as negative_adjustments
                FROM importance_adjustment_history 
                WHERE timestamp > ?
            """, (cutoff_time,))
            
            result = cursor.fetchone()
            
            if result:
                return {
                    "period_days": days,
                    "total_adjustments": result[0],
                    "avg_adjustment": result[1] or 0,
                    "positive_adjustments": result[2],
                    "negative_adjustments": result[3],
                    "adjustment_ratio": result[2] / max(result[3], 1) if result[3] > 0 else float('inf')
                }
            
            return {"period_days": days, "total_adjustments": 0}
            
        except Exception as e:
            logger.error(f"获取重要性趋势失败: {e}")
            return {"error": str(e)}
    
    async def get_top_accessed_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最常访问的记忆"""
        try:
            cursor = self.conn.execute("""
                SELECT 
                    m.id,
                    m.content,
                    m.importance,
                    u.access_count,
                    u.last_accessed,
                    u.context_matches,
                    u.positive_feedback
                FROM memories m
                JOIN memory_usage u ON m.id = u.memory_id
                WHERE u.access_count > 0
                ORDER BY u.access_count DESC, m.importance DESC
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "memory_id": row[0],
                    "content": row[1][:50] + "..." if len(row[1]) > 50 else row[1],
                    "importance": row[2],
                    "access_count": row[3],
                    "last_accessed": datetime.fromtimestamp(row[4]).strftime("%Y-%m-%d %H:%M") if row[4] > 0 else "从未访问",
                    "context_matches": row[5],
                    "positive_feedback": row[6]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"获取热门记忆失败: {e}")
            return []