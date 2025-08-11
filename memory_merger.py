"""
智能记忆去重和合并模块
负责检测相似记忆并智能合并
"""
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

@dataclass
class MergeResult:
    """合并结果"""
    should_merge: bool
    merge_strategy: str  # "replace", "combine", "keep_both"
    merged_content: str
    merged_structured_info: Dict[str, Any]
    confidence: float
    reason: str

class MemoryMerger:
    """智能记忆合并器"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.similarity_threshold = 0.8
        self.merge_confidence_threshold = 0.7
        
        logger.info("🔄 智能记忆合并器初始化完成")
    
    async def check_and_merge_similar_memories(self, new_memory, existing_memories: List) -> Optional[MergeResult]:
        """检查并合并相似记忆"""
        try:
            # 找到最相似的记忆
            most_similar = await self._find_most_similar_memory(new_memory, existing_memories)
            
            if not most_similar:
                return None
            
            similar_memory, similarity_score = most_similar
            
            if similarity_score < self.similarity_threshold:
                return None
            
            logger.info(f"🔍 发现相似记忆，相似度: {similarity_score:.2f}")
            logger.info(f"   新记忆: {new_memory.content}")
            logger.info(f"   相似记忆: {similar_memory.content}")
            
            # 使用GPT判断合并策略
            merge_result = await self._gpt_merge_analysis(new_memory, similar_memory)
            
            if merge_result.should_merge and merge_result.confidence >= self.merge_confidence_threshold:
                logger.info(f"✅ 记忆合并: {merge_result.merge_strategy} (置信度: {merge_result.confidence:.2f})")
                logger.info(f"   合并理由: {merge_result.reason}")
                return merge_result
            
            return None
            
        except Exception as e:
            logger.error(f"记忆合并检查失败: {e}")
            return None
    
    async def _find_most_similar_memory(self, new_memory, existing_memories: List) -> Optional[Tuple]:
        """找到最相似的记忆"""
        try:
            if not existing_memories:
                return None
            
            max_similarity = 0.0
            most_similar_memory = None
            
            new_content = new_memory.content.lower()
            new_type = new_memory.memory_type
            new_structured = new_memory.metadata.get("structured_info", {})
            
            for existing_memory in existing_memories:
                # 只比较相同类型的记忆
                if existing_memory.memory_type != new_type:
                    continue
                
                existing_content = existing_memory.content.lower()
                existing_structured = existing_memory.metadata.get("structured_info", {})
                
                # 计算相似度
                similarity = self._calculate_similarity(
                    new_content, existing_content,
                    new_structured, existing_structured
                )
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_memory = existing_memory
            
            if most_similar_memory and max_similarity > 0.6:
                return (most_similar_memory, max_similarity)
            
            return None
            
        except Exception as e:
            logger.error(f"查找相似记忆失败: {e}")
            return None
    
    def _calculate_similarity(self, content1: str, content2: str, 
                            structured1: Dict, structured2: Dict) -> float:
        """计算记忆相似度"""
        try:
            # 文本相似度 (简单的词汇重叠)
            words1 = set(content1.split())
            words2 = set(content2.split())
            
            if not words1 or not words2:
                text_similarity = 0.0
            else:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                text_similarity = intersection / union if union > 0 else 0.0
            
            # 结构化信息相似度
            struct_similarity = self._calculate_structured_similarity(structured1, structured2)
            
            # 综合相似度 (文本60% + 结构化40%)
            total_similarity = text_similarity * 0.6 + struct_similarity * 0.4
            
            return total_similarity
            
        except Exception as e:
            logger.debug(f"计算相似度失败: {e}")
            return 0.0
    
    def _calculate_structured_similarity(self, struct1: Dict, struct2: Dict) -> float:
        """计算结构化信息相似度"""
        try:
            if not struct1 or not struct2:
                return 0.0
            
            # 检查关键字段的重叠
            key_fields = ["name", "job", "location", "height", "weight", "age"]
            
            matches = 0
            total_fields = 0
            
            for field in key_fields:
                val1 = struct1.get(field)
                val2 = struct2.get(field)
                
                if val1 or val2:
                    total_fields += 1
                    if val1 and val2 and str(val1).lower() == str(val2).lower():
                        matches += 1
            
            return matches / total_fields if total_fields > 0 else 0.0
            
        except Exception as e:
            logger.debug(f"计算结构化相似度失败: {e}")
            return 0.0
    
    async def _gpt_merge_analysis(self, new_memory, existing_memory) -> MergeResult:
        """使用GPT分析合并策略"""
        try:
            prompt = f"""你是一个智能记忆管理专家，需要判断两条相似的记忆是否应该合并，以及如何合并。

新记忆:
- 内容: {new_memory.content}
- 类型: {new_memory.memory_type}
- 重要性: {new_memory.importance}
- 结构化信息: {json.dumps(new_memory.metadata.get('structured_info', {}), ensure_ascii=False)}

已存在记忆:
- 内容: {existing_memory.content}
- 类型: {existing_memory.memory_type}
- 重要性: {existing_memory.importance}
- 结构化信息: {json.dumps(existing_memory.metadata.get('structured_info', {}), ensure_ascii=False)}

请分析并输出JSON格式结果：

{{
    "should_merge": true/false,
    "merge_strategy": "replace/combine/keep_both",
    "merged_content": "合并后的内容",
    "merged_structured_info": {{"合并后的结构化信息"}},
    "confidence": 0.0-1.0,
    "reason": "合并理由"
}}

合并策略说明：
- replace: 新信息更准确/详细，替换旧信息
- combine: 两条信息互补，合并为更完整的信息
- keep_both: 信息不同但都重要，保留两条

判断标准：
1. 如果是同一信息的不同表述 → should_merge: true, strategy: replace/combine
2. 如果新信息更详细准确 → should_merge: true, strategy: replace
3. 如果信息互补 → should_merge: true, strategy: combine
4. 如果信息矛盾且无法判断 → should_merge: false, strategy: keep_both

示例：
- "我住在北京" + "我家在北京朝阳区" → combine: "我住在北京朝阳区"
- "我身高180cm" + "我有一米八" → replace: "我身高180cm"
- "我妈妈是医生" + "我妈妈是老师" → keep_both (需要用户确认)"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            if response.choices:
                result_text = response.choices[0].message.content.strip()
                try:
                    result_data = json.loads(result_text)
                    
                    return MergeResult(
                        should_merge=result_data.get("should_merge", False),
                        merge_strategy=result_data.get("merge_strategy", "keep_both"),
                        merged_content=result_data.get("merged_content", new_memory.content),
                        merged_structured_info=result_data.get("merged_structured_info", {}),
                        confidence=result_data.get("confidence", 0.0),
                        reason=result_data.get("reason", "GPT分析结果")
                    )
                    
                except json.JSONDecodeError:
                    logger.warning("GPT合并分析JSON解析失败")
            
            # 备用策略
            return self._fallback_merge_analysis(new_memory, existing_memory)
            
        except Exception as e:
            logger.error(f"GPT合并分析失败: {e}")
            return self._fallback_merge_analysis(new_memory, existing_memory)
    
    def _fallback_merge_analysis(self, new_memory, existing_memory) -> MergeResult:
        """备用合并分析"""
        # 简单规则：如果新记忆更长更详细，替换旧记忆
        if len(new_memory.content) > len(existing_memory.content) * 1.2:
            return MergeResult(
                should_merge=True,
                merge_strategy="replace",
                merged_content=new_memory.content,
                merged_structured_info=new_memory.metadata.get("structured_info", {}),
                confidence=0.8,
                reason="新记忆更详细"
            )
        
        # 否则保留两条
        return MergeResult(
            should_merge=False,
            merge_strategy="keep_both",
            merged_content="",
            merged_structured_info={},
            confidence=0.5,
            reason="无法确定合并策略"
        )
    
    async def merge_memories(self, target_memory, merge_result: MergeResult) -> Dict[str, Any]:
        """执行记忆合并"""
        try:
            if merge_result.merge_strategy == "replace":
                # 替换策略：使用新内容，保持更高的重要性
                merged_memory = {
                    "content": merge_result.merged_content,
                    "importance": max(target_memory.importance, target_memory.importance * 1.1),
                    "metadata": {
                        **target_memory.metadata,
                        "structured_info": merge_result.merged_structured_info,
                        "merge_history": {
                            "strategy": "replace",
                            "timestamp": time.time(),
                            "reason": merge_result.reason,
                            "original_content": target_memory.content
                        }
                    }
                }
                
            elif merge_result.merge_strategy == "combine":
                # 合并策略：组合信息
                merged_memory = {
                    "content": merge_result.merged_content,
                    "importance": max(target_memory.importance, target_memory.importance * 1.2),
                    "metadata": {
                        **target_memory.metadata,
                        "structured_info": {
                            **target_memory.metadata.get("structured_info", {}),
                            **merge_result.merged_structured_info
                        },
                        "merge_history": {
                            "strategy": "combine",
                            "timestamp": time.time(),
                            "reason": merge_result.reason,
                            "combined_from": [target_memory.content]
                        }
                    }
                }
            
            else:  # keep_both
                return None
            
            logger.info(f"✅ 记忆合并完成: {merge_result.merge_strategy}")
            return merged_memory
            
        except Exception as e:
            logger.error(f"执行记忆合并失败: {e}")
            return None