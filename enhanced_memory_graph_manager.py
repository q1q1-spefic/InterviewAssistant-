"""
增强版记忆图谱管理器
集成高级实体提取、优化图谱引擎和冲突解决
"""
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

from advanced_entity_extractor import AdvancedEntityExtractor, ExtractedEntity, ConflictInfo
from optimized_graph_engine import OptimizedGraphEngine, GraphNode, GraphEdge, SearchResult

class EnhancedMemoryGraphManager:
    """增强版记忆图谱管理器"""
    
    def __init__(self, openai_client, db_connection):
        self.openai_client = openai_client
        self.conn = db_connection
        
        # 初始化子系统
        self.entity_extractor = AdvancedEntityExtractor(openai_client)
        self.graph_engine = OptimizedGraphEngine(db_connection)
        
        # 配置参数
        self.auto_conflict_resolution = True
        self.max_search_results = 20
        self.search_timeout = 5.0
        
        # 性能监控
        self.performance_metrics = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "total_searches": 0,
            "successful_searches": 0,
            "avg_extraction_time": 0.0,
            "avg_search_time": 0.0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0
        }
        
        logger.info("🚀 增强版记忆图谱管理器初始化完成")
    
    async def add_memory_advanced(self, memory_id: str, content: str, 
                                memory_type: str, importance: float,
                                context: str = "") -> bool:
        """高级记忆添加"""
        try:
            start_time = time.time()
            self.performance_metrics["total_extractions"] += 1
            
            logger.debug(f"🧠 高级记忆添加: {content}")
            
            # 1. 高级实体提取
            entities, conflicts = await self.entity_extractor.extract_entities_advanced(
                content, context
            )
            
            if conflicts:
                self.performance_metrics["conflicts_detected"] += len(conflicts)
                logger.warning(f"⚠️ 检测到 {len(conflicts)} 个冲突")
                
                # 处理冲突
                if self.auto_conflict_resolution:
                    for conflict in conflicts:
                        resolution = await self.entity_extractor.resolve_conflict(conflict)
                        logger.info(f"🔧 冲突解决: {conflict.entity} -> {resolution}")
                        self.performance_metrics["conflicts_resolved"] += 1
            
            # 2. 创建图节点
            node_properties = {
                "memory_type": memory_type,
                "context": context,
                "entities": [entity.text for entity in entities],
                "entity_types": [entity.entity_type.value for entity in entities]
            }
            
            graph_node = GraphNode(
                id=memory_id,
                content=content,
                node_type=memory_type,
                importance=importance,
                properties=node_properties,
                created_time=time.time(),
                last_accessed=time.time(),
                access_count=0
            )
            
            # 3. 添加到图引擎
            entity_texts = [entity.normalized_form for entity in entities]
            success = await self.graph_engine.add_node(graph_node, entity_texts)
            
            if success:
                # 4. 构建关联边
                await self._build_advanced_relations(graph_node, entities)
                
                self.performance_metrics["successful_extractions"] += 1
                
                extraction_time = time.time() - start_time
                self._update_extraction_time(extraction_time)
                
                logger.info(f"✅ 高级记忆添加成功: {len(entities)} 个实体, 耗时 {extraction_time:.3f}s")
                return True
            else:
                logger.error(f"❌ 图节点添加失败: {memory_id}")
                return False
            
        except Exception as e:
            logger.error(f"高级记忆添加失败: {e}")
            return False
    
    async def _build_advanced_relations(self, node: GraphNode, entities: List[ExtractedEntity]):
        """构建高级关联关系"""
        try:
            # 1. 实体关联
            for entity in entities:
                # 查找具有相同实体的其他节点
                similar_nodes = await self.graph_engine.search_nodes_by_entities(
                    [entity.normalized_form] + entity.aliases, max_results=10
                )
                
                for similar_node in similar_nodes:
                    if similar_node.id != node.id:
                        # 创建实体关联边
                        edge = GraphEdge(
                            id=f"{node.id}_{similar_node.id}_entity_{entity.normalized_form}",
                            source_id=node.id,
                            target_id=similar_node.id,
                            edge_type=f"entity_{entity.entity_type.value}",
                            weight=1.0,
                            confidence=entity.confidence,
                            properties={
                                "entity": entity.normalized_form,
                                "entity_type": entity.entity_type.value,
                                "relation_reason": f"共同实体: {entity.normalized_form}"
                            },
                            created_time=time.time()
                        )
                        
                        await self.graph_engine.add_edge(edge)
            
            # 2. 语义关联
            await self._build_semantic_relations(node)
            
            # 3. 重要性关联
            await self._build_importance_relations(node)
            
        except Exception as e:
            logger.error(f"构建高级关联失败: {e}")
    
    async def _build_semantic_relations(self, node: GraphNode):
        """构建语义关联"""
        try:
            # 查找高重要性节点进行语义关联
            high_importance_nodes = await self.graph_engine.search_by_importance(
                min_importance=0.8, max_results=10
            )
            
            for other_node in high_importance_nodes:
                if other_node.id != node.id:
                    # 使用AI分析语义关联
                    relation = await self._analyze_semantic_similarity(node, other_node)
                    
                    if relation and relation.confidence > 0.7:
                        edge = GraphEdge(
                            id=f"{node.id}_{other_node.id}_semantic",
                            source_id=node.id,
                            target_id=other_node.id,
                            edge_type="semantic",
                            weight=relation.confidence,
                            confidence=relation.confidence,
                            properties={
                                "relation_type": "semantic_similarity",
                                "description": relation.description
                            },
                            created_time=time.time()
                        )
                        
                        await self.graph_engine.add_edge(edge)
            
        except Exception as e:
            logger.error(f"构建语义关联失败: {e}")
    
    async def _analyze_semantic_similarity(self, node1: GraphNode, node2: GraphNode) -> Optional[Any]:
        """分析语义相似性"""
        try:
            prompt = f"""
分析以下两个记忆片段的语义关联：

记忆1: {node1.content}
记忆2: {node2.content}

请判断它们是否有语义关联，如果有，给出关联类型和置信度。

输出JSON格式：
{{
    "has_relation": true/false,
    "confidence": 0.0-1.0,
    "relation_type": "family/work/location/preference/other",
    "description": "关联描述"
}}
"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1
            )
            
            if response.choices:
                import json
                try:
                    result = json.loads(response.choices[0].message.content)
                    if result.get("has_relation", False):
                        from dataclasses import dataclass
                        
                        @dataclass
                        class SemanticRelation:
                            confidence: float
                            description: str
                        
                        return SemanticRelation(
                            confidence=result.get("confidence", 0.5),
                            description=result.get("description", "语义关联")
                        )
                except json.JSONDecodeError:
                    logger.debug("语义分析JSON解析失败")
            
            return None
            
        except Exception as e:
            logger.error(f"语义相似性分析失败: {e}")
            return None
    
    async def _build_importance_relations(self, node: GraphNode):
        """构建重要性关联"""
        try:
            # 如果是高重要性节点，与其他高重要性节点建立关联
            if node.importance >= 0.8:
                similar_importance_nodes = await self.graph_engine.search_by_importance(
                    min_importance=node.importance - 0.1, max_results=5
                )
                
                for other_node in similar_importance_nodes:
                    if (other_node.id != node.id and 
                        other_node.node_type == node.node_type):
                        
                        edge = GraphEdge(
                            id=f"{node.id}_{other_node.id}_importance",
                            source_id=node.id,
                            target_id=other_node.id,
                            edge_type="importance_cluster",
                            weight=min(node.importance, other_node.importance),
                            confidence=0.8,
                            properties={
                                "relation_type": "similar_importance",
                                "importance_level": "high"
                            },
                            created_time=time.time()
                        )
                        
                        await self.graph_engine.add_edge(edge)
            
        except Exception as e:
            logger.error(f"构建重要性关联失败: {e}")
    
    async def enhanced_search_and_respond(self, question: str, context: str = "") -> Tuple[str, List[str]]:
        """增强搜索并回答"""
        try:
            start_time = time.time()
            self.performance_metrics["total_searches"] += 1
            
            logger.info(f"🔍 增强搜索: {question}")
            
            # 1. 从问题中提取实体
            question_entities, _ = await self.entity_extractor.extract_entities_advanced(question, context)
            entity_texts = [entity.normalized_form for entity in question_entities]
            
            logger.debug(f"问题实体: {entity_texts}")
            
            # 2. 多策略搜索
            search_strategies = ["hybrid", "entity_first", "importance_first"]
            best_result = None
            best_confidence = 0.0
            
            for strategy in search_strategies:
                try:
                    result = await asyncio.wait_for(
                        self.graph_engine.advanced_search(question, entity_texts, strategy),
                        timeout=self.search_timeout
                    )
                    
                    if result.confidence > best_confidence:
                        best_result = result
                        best_confidence = result.confidence
                        
                except asyncio.TimeoutError:
                    logger.warning(f"搜索策略 {strategy} 超时")
                    continue
            
            if not best_result or not best_result.nodes:
                logger.info("📭 未找到相关记忆")
                return "抱歉，我没有找到相关信息。", []
            
            # 3. 更新访问统计
            for node in best_result.nodes:
                await self.graph_engine.update_node_access(node.id)
            
            # 4. 生成智能回答
            response = await self._generate_enhanced_response(
                question, best_result, context
            )
            
            used_memory_ids = [node.id for node in best_result.nodes]
            
            search_time = time.time() - start_time
            self._update_search_time(search_time)
            self.performance_metrics["successful_searches"] += 1
            
            logger.info(f"✅ 增强搜索完成: {len(used_memory_ids)} 条记忆, 耗时 {search_time:.3f}s")
            return response, used_memory_ids
            
        except Exception as e:
            logger.error(f"增强搜索失败: {e}")
            return "抱歉，搜索过程中出现了错误。", []
    
    async def _generate_enhanced_response(self, question: str, search_result: SearchResult, context: str) -> str:
        """生成增强回答"""
        try:
            # 构建丰富的上下文
            memory_context = []
            reasoning_context = []
            
            # 按重要性排序节点
            sorted_nodes = sorted(search_result.nodes, key=lambda x: x.importance, reverse=True)
            
            for node in sorted_nodes[:10]:  # 最多10个节点
                memory_context.append(f"- {node.content} (重要性: {node.importance:.2f})")
            
            # 推理路径
            for step in search_result.reasoning_path:
                reasoning_context.append(f"→ {step}")
            
            # 关联信息
            if search_result.edges:
                relation_info = []
                for edge in search_result.edges[:5]:  # 最多5个关联
                    relation_info.append(f"  {edge.edge_type}: {edge.properties.get('description', '关联')}")
                reasoning_context.extend(relation_info)
            
            memory_text = "\n".join(memory_context)
            reasoning_text = "\n".join(reasoning_context)
            
            prompt = f"""基于以下记忆图谱信息和推理路径，回答用户问题。

问题: {question}
上下文: {context}

记忆信息:
{memory_text}

推理路径:
{reasoning_text}

要求:
1. 基于记忆信息进行推理回答
2. 利用推理路径中的关联信息
3. 如果能推理出答案，说明推理过程
4. 如果信息不足，明确指出缺少什么
5. 回答要自然、口语化
6. 1-2句话简洁回答

请直接给出回答："""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            
            if response.choices:
                return response.choices[0].message.content.strip()
            
            return "抱歉，无法生成回答。"
            
        except Exception as e:
            logger.error(f"生成增强回答失败: {e}")
            return "抱歉，回答生成过程中出现了错误。"
    
    async def batch_add_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量添加记忆"""
        try:
            logger.info(f"📦 批量添加 {len(memories)} 条记忆")
            
            start_time = time.time()
            successes = 0
            failures = 0
            errors = []
            
            # 并发处理（限制并发数）
            semaphore = asyncio.Semaphore(5)  # 最多5个并发
            
            async def process_memory(memory_data):
                async with semaphore:
                    try:
                        success = await self.add_memory_advanced(
                            memory_data["id"],
                            memory_data["content"],
                            memory_data.get("memory_type", "general"),
                            memory_data.get("importance", 0.5),
                            memory_data.get("context", "")
                        )
                        return success, None
                    except Exception as e:
                        return False, str(e)
            
            # 执行批量处理
            tasks = [process_memory(memory) for memory in memories]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, tuple):
                    success, error = result
                    if success:
                        successes += 1
                    else:
                        failures += 1
                        if error:
                            errors.append(f"记忆{i}: {error}")
                else:
                    failures += 1
                    errors.append(f"记忆{i}: {str(result)}")
            
            total_time = time.time() - start_time
            
            logger.info(f"📦 批量添加完成: {successes}成功 / {failures}失败, 耗时 {total_time:.1f}s")
            
            return {
                "total": len(memories),
                "successes": successes,
                "failures": failures,
                "success_rate": successes / len(memories) * 100,
                "total_time": total_time,
                "throughput": len(memories) / total_time,
                "errors": errors[:10]  # 最多显示10个错误
            }
            
        except Exception as e:
            logger.error(f"批量添加记忆失败: {e}")
            return {"error": str(e)}
    
    async def analyze_memory_network(self) -> Dict[str, Any]:
        """分析记忆网络"""
        try:
            logger.info("📊 分析记忆网络结构")
            
            # 获取图引擎性能统计
            graph_stats = self.graph_engine.get_performance_stats()
            
            # 分析网络密度
            if graph_stats["total_nodes"] > 0 and graph_stats["total_edges"] > 0:
                # 网络密度 = 实际边数 / 最大可能边数
                max_edges = graph_stats["total_nodes"] * (graph_stats["total_nodes"] - 1) / 2
                network_density = graph_stats["total_edges"] / max_edges if max_edges > 0 else 0
            else:
                network_density = 0
            
            # 分析实体分布
            entity_stats = await self._analyze_entity_distribution()
            
            # 分析重要性分布
            importance_stats = await self._analyze_importance_distribution()
            
            # 分析连接性
            connectivity_stats = await self._analyze_connectivity()
            
            analysis = {
                "network_overview": {
                    "total_nodes": graph_stats["total_nodes"],
                    "total_edges": graph_stats["total_edges"],
                    "network_density": network_density,
                    "avg_search_time": graph_stats["avg_search_time"],
                    "cache_hit_rate": graph_stats.get("cache_hit_rate", 0)
                },
                "entity_distribution": entity_stats,
                "importance_distribution": importance_stats,
                "connectivity_analysis": connectivity_stats,
                "performance_metrics": self.performance_metrics,
                "recommendations": self._generate_network_recommendations(graph_stats, network_density)
            }
            
            logger.info(f"📊 网络分析完成: {graph_stats['total_nodes']} 节点, 密度 {network_density:.3f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"记忆网络分析失败: {e}")
            return {"error": str(e)}
    
    async def _analyze_entity_distribution(self) -> Dict[str, Any]:
        """分析实体分布"""
        try:
            cursor = self.conn.execute("""
                SELECT entity, COUNT(*) as count 
                FROM entity_node_index 
                GROUP BY entity 
                ORDER BY count DESC 
                LIMIT 20
            """)
            
            entity_counts = dict(cursor.fetchall())
            
            # 统计实体类型
            cursor = self.conn.execute("""
                SELECT 
                    json_extract(properties, '$.entity_types') as types,
                    COUNT(*) as count
                FROM graph_nodes 
                WHERE json_extract(properties, '$.entity_types') IS NOT NULL
                GROUP BY types
            """)
            
            type_distribution = {}
            for types_json, count in cursor.fetchall():
                if types_json:
                    import json
                    try:
                        types = json.loads(types_json)
                        for entity_type in types:
                            type_distribution[entity_type] = type_distribution.get(entity_type, 0) + count
                    except:
                        pass
            
            return {
                "top_entities": entity_counts,
                "entity_type_distribution": type_distribution,
                "total_unique_entities": len(entity_counts),
                "avg_entity_frequency": sum(entity_counts.values()) / len(entity_counts) if entity_counts else 0
            }
            
        except Exception as e:
            logger.error(f"实体分布分析失败: {e}")
            return {}
    
    async def _analyze_importance_distribution(self) -> Dict[str, Any]:
        """分析重要性分布"""
        try:
            cursor = self.conn.execute("""
                SELECT 
                    CASE 
                        WHEN importance >= 0.9 THEN 'very_high'
                        WHEN importance >= 0.7 THEN 'high'
                        WHEN importance >= 0.5 THEN 'medium'
                        WHEN importance >= 0.3 THEN 'low'
                        ELSE 'very_low'
                    END as importance_level,
                    COUNT(*) as count,
                    AVG(importance) as avg_importance
                FROM graph_nodes
                GROUP BY importance_level
                ORDER BY avg_importance DESC
            """)
            
            distribution = {}
            for level, count, avg_imp in cursor.fetchall():
                distribution[level] = {
                    "count": count,
                    "avg_importance": round(avg_imp, 3)
                }
            
            # 总体统计
            cursor = self.conn.execute("SELECT AVG(importance), MIN(importance), MAX(importance) FROM graph_nodes")
            overall_avg, min_imp, max_imp = cursor.fetchone()
            
            return {
                "distribution": distribution,
                "overall_avg": round(overall_avg or 0, 3),
                "min_importance": round(min_imp or 0, 3),
                "max_importance": round(max_imp or 0, 3)
            }
            
        except Exception as e:
            logger.error(f"重要性分布分析失败: {e}")
            return {}
    
    async def _analyze_connectivity(self) -> Dict[str, Any]:
        """分析连接性"""
        try:
            # 计算平均度数
            cursor = self.conn.execute("""
                SELECT node_id, COUNT(*) as degree
                FROM (
                    SELECT source_id as node_id FROM graph_edges
                    UNION ALL
                    SELECT target_id as node_id FROM graph_edges
                ) 
                GROUP BY node_id
            """)
            
            degrees = [row[1] for row in cursor.fetchall()]
            
            if degrees:
                avg_degree = sum(degrees) / len(degrees)
                max_degree = max(degrees)
                min_degree = min(degrees)
            else:
                avg_degree = max_degree = min_degree = 0
            
            # 孤立节点数量
            cursor = self.conn.execute("""
                SELECT COUNT(*) FROM graph_nodes 
                WHERE id NOT IN (
                    SELECT DISTINCT source_id FROM graph_edges
                    UNION
                    SELECT DISTINCT target_id FROM graph_edges
                )
            """)
            isolated_nodes = cursor.fetchone()[0]
            
            # 边类型分布
            cursor = self.conn.execute("""
                SELECT edge_type, COUNT(*) as count 
                FROM graph_edges 
                GROUP BY edge_type 
                ORDER BY count DESC
            """)
            edge_type_distribution = dict(cursor.fetchall())
            
            return {
                "avg_degree": round(avg_degree, 2),
                "max_degree": max_degree,
                "min_degree": min_degree,
                "isolated_nodes": isolated_nodes,
                "edge_type_distribution": edge_type_distribution,
                "connectivity_ratio": (len(degrees) / self.graph_engine.performance_stats["total_nodes"]) if self.graph_engine.performance_stats["total_nodes"] > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"连接性分析失败: {e}")
            return {}
    
    def _generate_network_recommendations(self, graph_stats: Dict, network_density: float) -> List[str]:
        """生成网络优化建议"""
        recommendations = []
        
        # 密度建议
        if network_density < 0.1:
            recommendations.append("网络密度较低，建议增强实体关联算法")
        elif network_density > 0.8:
            recommendations.append("网络密度过高，可能存在冗余关联，建议优化关联质量")
        
        # 性能建议
        if graph_stats.get("avg_search_time", 0) > 1.0:
            recommendations.append("搜索性能较慢，建议优化索引和缓存策略")
        
        # 缓存建议
        if graph_stats.get("cache_hit_rate", 0) < 0.5:
            recommendations.append("缓存命中率较低，建议调整缓存策略和大小")
        
        # 规模建议
        if graph_stats["total_nodes"] > 10000:
            recommendations.append("节点数量较大，建议启用分布式处理和数据分片")
        
        if not recommendations:
            recommendations.append("网络结构良好，继续保持当前配置")
        
        return recommendations
    
    def _update_extraction_time(self, time_taken: float):
        """更新提取时间统计"""
        total_time = (self.performance_metrics["avg_extraction_time"] * 
                     (self.performance_metrics["successful_extractions"] - 1))
        self.performance_metrics["avg_extraction_time"] = (
            (total_time + time_taken) / self.performance_metrics["successful_extractions"]
        )
    
    def _update_search_time(self, time_taken: float):
        """更新搜索时间统计"""
        total_time = (self.performance_metrics["avg_search_time"] * 
                     (self.performance_metrics["successful_searches"] - 1))
        self.performance_metrics["avg_search_time"] = (
            (total_time + time_taken) / self.performance_metrics["successful_searches"]
        )
    
    async def optimize_network(self) -> Dict[str, Any]:
        """优化网络结构"""
        try:
            logger.info("🔧 开始网络优化...")
            
            start_time = time.time()
            optimizations = []
            
            # 1. 清理低价值节点
            cleaned_count = await self.graph_engine.cleanup_low_value_nodes(
                min_importance=0.2, min_access_count=1, days_threshold=30
            )
            
            if cleaned_count > 0:
                optimizations.append(f"清理了 {cleaned_count} 个低价值节点")
            
            # 2. 重建索引
            self.graph_engine._build_indexes()
            optimizations.append("重建了内存索引")
            
            # 3. 清理缓存
            self.graph_engine.node_cache.clear()
            self.graph_engine.edge_cache.clear()
            self.graph_engine.search_cache.clear()
            optimizations.append("清理了所有缓存")
            
            optimization_time = time.time() - start_time
            
            logger.info(f"🔧 网络优化完成: 耗时 {optimization_time:.1f}s")
            
            return {
                "optimization_time": optimization_time,
                "optimizations_applied": optimizations,
                "nodes_cleaned": cleaned_count
            }
            
        except Exception as e:
            logger.error(f"网络优化失败: {e}")
            return {"error": str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            graph_stats = self.graph_engine.get_performance_stats()
            
            return {
                "system_health": "healthy" if graph_stats.get("avg_search_time", 0) < 2.0 else "degraded",
                "performance_metrics": self.performance_metrics,
                "graph_statistics": graph_stats,
                "extractor_status": "active",
                "auto_conflict_resolution": self.auto_conflict_resolution,
                "search_timeout": self.search_timeout,
                "max_search_results": self.max_search_results
            }
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {"error": str(e)}
    
    def close(self):
        """关闭管理器"""
        try:
            if hasattr(self.graph_engine, 'close'):
                self.graph_engine.close()
            
            logger.info("🚀 增强版记忆图谱管理器已关闭")
            
        except Exception as e:
            logger.error(f"关闭管理器失败: {e}")