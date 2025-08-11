"""
优化图谱引擎
支持大规模数据、快速搜索、智能缓存和分布式处理
"""
import asyncio
import json
import time
import sqlite3
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import pickle
import lru
from loguru import logger

@dataclass
class GraphNode:
    """图节点"""
    id: str
    content: str
    node_type: str
    importance: float
    properties: Dict[str, Any]
    created_time: float
    last_accessed: float
    access_count: int

@dataclass
class GraphEdge:
    """图边"""
    id: str
    source_id: str
    target_id: str
    edge_type: str
    weight: float
    confidence: float
    properties: Dict[str, Any]
    created_time: float

@dataclass
class SearchResult:
    """搜索结果"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    reasoning_path: List[str]
    confidence: float
    search_time: float

class OptimizedGraphEngine:
    """优化图谱引擎"""
    
    def __init__(self, db_connection, cache_size: int = 10000):
        self.conn = db_connection
        self.cache_size = cache_size
        
        # 高性能缓存
        self.node_cache = lru.LRU(cache_size)
        self.edge_cache = lru.LRU(cache_size)
        self.search_cache = lru.LRU(cache_size // 10)
        
        # 索引结构
        self.entity_index = defaultdict(set)  # entity -> node_ids
        self.type_index = defaultdict(set)    # type -> node_ids
        self.importance_index = []            # [(importance, node_id)]
        
        # 性能监控
        self.performance_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "avg_search_time": 0.0,
            "total_nodes": 0,
            "total_edges": 0
        }
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self._init_optimized_storage()
        self._build_indexes()
        
        logger.info("⚡ 优化图谱引擎初始化完成")
    
    def _init_optimized_storage(self):
        """初始化优化存储"""
        try:
            # 优化的节点表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    importance REAL NOT NULL,
                    properties TEXT,
                    created_time REAL NOT NULL,
                    last_accessed REAL DEFAULT 0,
                    access_count INTEGER DEFAULT 0
                )
            """)
            
            # 优化的边表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    confidence REAL NOT NULL,
                    properties TEXT,
                    created_time REAL NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES graph_nodes (id),
                    FOREIGN KEY (target_id) REFERENCES graph_nodes (id)
                )
            """)
            
            # 高性能索引表
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_node_index (
                    entity TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    PRIMARY KEY (entity, node_id)
                )
            """)
            
            # 创建索引
            try:
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type_importance ON graph_nodes (node_type, importance DESC)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_importance ON graph_nodes (importance DESC)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges (source_id)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges (target_id)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON graph_edges (edge_type)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_index ON entity_node_index (entity)")
            except:
                pass  # 索引可能已存在
            
            self.conn.commit()
            logger.debug("✅ 优化存储初始化完成")
            
        except Exception as e:
            logger.error(f"优化存储初始化失败: {e}")
    
    def _build_indexes(self):
        """构建内存索引"""
        try:
            start_time = time.time()
            
            # 构建实体索引
            cursor = self.conn.execute("SELECT entity, node_id FROM entity_node_index")
            for entity, node_id in cursor.fetchall():
                self.entity_index[entity].add(node_id)
            
            # 构建类型索引
            cursor = self.conn.execute("SELECT node_type, id FROM graph_nodes")
            for node_type, node_id in cursor.fetchall():
                self.type_index[node_type].add(node_id)
            
            # 构建重要性索引
            cursor = self.conn.execute("SELECT importance, id FROM graph_nodes ORDER BY importance DESC")
            self.importance_index = list(cursor.fetchall())
            
            # 更新统计
            cursor = self.conn.execute("SELECT COUNT(*) FROM graph_nodes")
            self.performance_stats["total_nodes"] = cursor.fetchone()[0]
            
            cursor = self.conn.execute("SELECT COUNT(*) FROM graph_edges")
            self.performance_stats["total_edges"] = cursor.fetchone()[0]
            
            build_time = time.time() - start_time
            logger.info(f"📊 索引构建完成: {self.performance_stats['total_nodes']} 节点, "
                       f"{self.performance_stats['total_edges']} 边, 耗时 {build_time:.3f}s")
            
        except Exception as e:
            logger.error(f"索引构建失败: {e}")
    
    async def add_node(self, node: GraphNode, entities: List[str] = None) -> bool:
        """添加节点"""
        try:
            # 存储节点
            self.conn.execute("""
                INSERT OR REPLACE INTO graph_nodes 
                (id, content, node_type, importance, properties, created_time, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node.id, node.content, node.node_type, node.importance,
                json.dumps(node.properties), node.created_time, 
                node.last_accessed, node.access_count
            ))
            
            # 更新实体索引
            if entities:
                for entity in entities:
                    self.conn.execute("""
                        INSERT OR REPLACE INTO entity_node_index (entity, node_id, confidence)
                        VALUES (?, ?, ?)
                    """, (entity, node.id, 1.0))
                    
                    # 更新内存索引
                    self.entity_index[entity].add(node.id)
            
            # 更新类型索引
            self.type_index[node.node_type].add(node.id)
            
            # 更新重要性索引
            self.importance_index.append((node.importance, node.id))
            self.importance_index.sort(reverse=True)
            
            # 缓存节点
            self.node_cache[node.id] = node
            
            self.conn.commit()
            self.performance_stats["total_nodes"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"添加节点失败: {e}")
            return False
    
    async def add_edge(self, edge: GraphEdge) -> bool:
        """添加边"""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO graph_edges 
                (id, source_id, target_id, edge_type, weight, confidence, properties, created_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                edge.id, edge.source_id, edge.target_id, edge.edge_type,
                edge.weight, edge.confidence, json.dumps(edge.properties), edge.created_time
            ))
            
            # 缓存边
            edge_key = f"{edge.source_id}:{edge.target_id}"
            if edge_key not in self.edge_cache:
                self.edge_cache[edge_key] = []
            self.edge_cache[edge_key].append(edge)
            
            self.conn.commit()
            self.performance_stats["total_edges"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"添加边失败: {e}")
            return False
    
    async def search_nodes_by_entities(self, entities: List[str], max_results: int = 50) -> List[GraphNode]:
        """通过实体搜索节点"""
        try:
            start_time = time.time()
            
            # 使用内存索引快速查找
            candidate_node_ids = set()
            for entity in entities:
                if entity in self.entity_index:
                    candidate_node_ids.update(self.entity_index[entity])
            
            if not candidate_node_ids:
                return []
            
            # 限制结果数量
            candidate_node_ids = list(candidate_node_ids)[:max_results]
            
            # 批量获取节点
            nodes = await self._get_nodes_batch(candidate_node_ids)
            
            # 按重要性排序
            nodes.sort(key=lambda x: x.importance, reverse=True)
            
            search_time = time.time() - start_time
            self._update_performance_stats(search_time, len(nodes))
            
            return nodes[:max_results]
            
        except Exception as e:
            logger.error(f"实体搜索失败: {e}")
            return []
    
    async def search_by_importance(self, min_importance: float = 0.7, max_results: int = 20) -> List[GraphNode]:
        """按重要性搜索"""
        try:
            # 使用重要性索引
            candidate_ids = [node_id for importance, node_id in self.importance_index 
                           if importance >= min_importance][:max_results]
            
            nodes = await self._get_nodes_batch(candidate_ids)
            return nodes
            
        except Exception as e:
            logger.error(f"重要性搜索失败: {e}")
            return []
    
    async def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> Optional[List[str]]:
        """查找两个节点间的路径"""
        try:
            # 使用BFS查找最短路径
            queue = deque([(source_id, [source_id])])
            visited = {source_id}
            
            while queue:
                current_id, path = queue.popleft()
                
                if len(path) > max_depth:
                    continue
                
                if current_id == target_id:
                    return path
                
                # 获取邻居节点
                neighbors = await self._get_neighbors(current_id)
                
                for neighbor_id in neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, path + [neighbor_id]))
            
            return None  # 未找到路径
            
        except Exception as e:
            logger.error(f"路径查找失败: {e}")
            return None
    
    async def _get_neighbors(self, node_id: str) -> List[str]:
        """获取邻居节点"""
        try:
            # 检查缓存
            cache_key = f"neighbors:{node_id}"
            if cache_key in self.search_cache:
                return self.search_cache[cache_key]
            
            cursor = self.conn.execute("""
                SELECT target_id FROM graph_edges WHERE source_id = ?
                UNION
                SELECT source_id FROM graph_edges WHERE target_id = ?
            """, (node_id, node_id))
            
            neighbors = [row[0] for row in cursor.fetchall()]
            
            # 缓存结果
            self.search_cache[cache_key] = neighbors
            
            return neighbors
            
        except Exception as e:
            logger.error(f"获取邻居失败: {e}")
            return []
    
    async def _get_nodes_batch(self, node_ids: List[str]) -> List[GraphNode]:
        """批量获取节点"""
        try:
            nodes = []
            uncached_ids = []
            
            # 先从缓存获取
            for node_id in node_ids:
                if node_id in self.node_cache:
                    nodes.append(self.node_cache[node_id])
                    self.performance_stats["cache_hits"] += 1
                else:
                    uncached_ids.append(node_id)
            
            # 批量查询未缓存的节点
            if uncached_ids:
                placeholders = ','.join(['?'] * len(uncached_ids))
                cursor = self.conn.execute(f"""
                    SELECT id, content, node_type, importance, properties, 
                           created_time, last_accessed, access_count
                    FROM graph_nodes WHERE id IN ({placeholders})
                """, uncached_ids)
                
                for row in cursor.fetchall():
                    node = GraphNode(
                        id=row[0],
                        content=row[1],
                        node_type=row[2],
                        importance=row[3],
                        properties=json.loads(row[4]) if row[4] else {},
                        created_time=row[5],
                        last_accessed=row[6],
                        access_count=row[7]
                    )
                    nodes.append(node)
                    
                    # 缓存节点
                    self.node_cache[node.id] = node
            
            return nodes
            
        except Exception as e:
            logger.error(f"批量获取节点失败: {e}")
            return []
    
    async def advanced_search(self, query: str, entities: List[str], 
                            search_strategy: str = "hybrid") -> SearchResult:
        """高级搜索"""
        try:
            start_time = time.time()
            
            # 生成缓存键
            cache_key = hashlib.md5(f"{query}:{':'.join(entities)}:{search_strategy}".encode()).hexdigest()
            
            # 检查搜索缓存
            if cache_key in self.search_cache:
                logger.debug("🎯 搜索缓存命中")
                return self.search_cache[cache_key]
            
            if search_strategy == "entity_first":
                result = await self._entity_first_search(query, entities)
            elif search_strategy == "importance_first":
                result = await self._importance_first_search(query, entities)
            elif search_strategy == "path_based":
                result = await self._path_based_search(query, entities)
            else:  # hybrid
                result = await self._hybrid_search(query, entities)
            
            result.search_time = time.time() - start_time
            
            # 缓存结果
            self.search_cache[cache_key] = result
            
            self._update_performance_stats(result.search_time, len(result.nodes))
            
            return result
            
        except Exception as e:
            logger.error(f"高级搜索失败: {e}")
            return SearchResult([], [], [], 0.0, 0.0)
    
    async def _entity_first_search(self, query: str, entities: List[str]) -> SearchResult:
        """实体优先搜索"""
        primary_nodes = await self.search_nodes_by_entities(entities, max_results=20)
        
        # 扩展搜索
        expanded_nodes = []
        reasoning_path = [f"找到 {len(primary_nodes)} 个直接匹配的节点"]
        
        for node in primary_nodes[:5]:  # 只从前5个节点扩展
            neighbors = await self._get_neighbors(node.id)
            neighbor_nodes = await self._get_nodes_batch(neighbors[:3])  # 每个节点最多3个邻居
            expanded_nodes.extend(neighbor_nodes)
            
            if neighbor_nodes:
                reasoning_path.append(f"从节点 '{node.content}' 扩展找到 {len(neighbor_nodes)} 个相关节点")
        
        all_nodes = primary_nodes + expanded_nodes
        all_edges = await self._get_edges_between_nodes([node.id for node in all_nodes])
        
        # 计算置信度
        confidence = min(1.0, len(primary_nodes) * 0.2 + len(expanded_nodes) * 0.1)
        
        return SearchResult(all_nodes, all_edges, reasoning_path, confidence, 0.0)
    
    async def _importance_first_search(self, query: str, entities: List[str]) -> SearchResult:
        """重要性优先搜索"""
        high_importance_nodes = await self.search_by_importance(min_importance=0.8, max_results=10)
        
        # 过滤与查询相关的节点
        relevant_nodes = []
        reasoning_path = ["按重要性搜索高价值节点"]
        
        for node in high_importance_nodes:
            # 简单的相关性检查
            if any(entity.lower() in node.content.lower() for entity in entities):
                relevant_nodes.append(node)
                reasoning_path.append(f"高重要性节点: '{node.content}' (重要性: {node.importance:.2f})")
        
        edges = await self._get_edges_between_nodes([node.id for node in relevant_nodes])
        confidence = len(relevant_nodes) * 0.15
        
        return SearchResult(relevant_nodes, edges, reasoning_path, confidence, 0.0)
    
    async def _path_based_search(self, query: str, entities: List[str]) -> SearchResult:
        """路径基础搜索"""
        entity_nodes = await self.search_nodes_by_entities(entities, max_results=10)
        
        if len(entity_nodes) < 2:
            return SearchResult(entity_nodes, [], ["实体节点不足，无法进行路径搜索"], 0.1, 0.0)
        
        # 查找节点间路径
        paths = []
        reasoning_path = ["查找实体节点间的连接路径"]
        
        for i, node1 in enumerate(entity_nodes[:3]):
            for node2 in entity_nodes[i+1:i+3]:  # 限制路径数量
                path = await self.find_path(node1.id, node2.id, max_depth=3)
                if path:
                    paths.append(path)
                    reasoning_path.append(f"找到路径: {node1.content} → {node2.content}")
        
        # 收集路径上的所有节点
        path_node_ids = set()
        for path in paths:
            path_node_ids.update(path)
        
        path_nodes = await self._get_nodes_batch(list(path_node_ids))
        path_edges = await self._get_edges_between_nodes(list(path_node_ids))
        
        confidence = min(1.0, len(paths) * 0.3)
        
        return SearchResult(path_nodes, path_edges, reasoning_path, confidence, 0.0)
    
    async def _hybrid_search(self, query: str, entities: List[str]) -> SearchResult:
        """混合搜索策略"""
        # 并行执行多种搜索策略
        tasks = [
            self._entity_first_search(query, entities),
            self._importance_first_search(query, entities),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并结果
        all_nodes = []
        all_edges = []
        reasoning_path = ["使用混合搜索策略"]
        total_confidence = 0.0
        
        for i, result in enumerate(results):
            if isinstance(result, SearchResult):
                all_nodes.extend(result.nodes)
                all_edges.extend(result.edges)
                reasoning_path.extend([f"策略{i+1}: " + line for line in result.reasoning_path])
                total_confidence += result.confidence
        
        # 去重
        seen_node_ids = set()
        unique_nodes = []
        for node in all_nodes:
            if node.id not in seen_node_ids:
                unique_nodes.append(node)
                seen_node_ids.add(node.id)
        
        # 按重要性排序
        unique_nodes.sort(key=lambda x: x.importance, reverse=True)
        
        return SearchResult(unique_nodes[:20], all_edges, reasoning_path, 
                          min(1.0, total_confidence), 0.0)
    
    async def _get_edges_between_nodes(self, node_ids: List[str]) -> List[GraphEdge]:
        """获取节点间的边"""
        try:
            if not node_ids:
                return []
            
            placeholders = ','.join(['?'] * len(node_ids))
            cursor = self.conn.execute(f"""
                SELECT id, source_id, target_id, edge_type, weight, confidence, properties, created_time
                FROM graph_edges 
                WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})
            """, node_ids + node_ids)
            
            edges = []
            for row in cursor.fetchall():
                edge = GraphEdge(
                    id=row[0],
                    source_id=row[1],
                    target_id=row[2],
                    edge_type=row[3],
                    weight=row[4],
                    confidence=row[5],
                    properties=json.loads(row[6]) if row[6] else {},
                    created_time=row[7]
                )
                edges.append(edge)
            
            return edges
            
        except Exception as e:
            logger.error(f"获取边失败: {e}")
            return []
    
    def _update_performance_stats(self, search_time: float, result_count: int):
        """更新性能统计"""
        self.performance_stats["total_searches"] += 1
        total_time = self.performance_stats["avg_search_time"] * (self.performance_stats["total_searches"] - 1)
        self.performance_stats["avg_search_time"] = (total_time + search_time) / self.performance_stats["total_searches"]
    
    async def update_node_access(self, node_id: str):
        """更新节点访问信息"""
        try:
            current_time = time.time()
            self.conn.execute("""
                UPDATE graph_nodes 
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?
            """, (current_time, node_id))
            
            # 更新缓存
            if node_id in self.node_cache:
                node = self.node_cache[node_id]
                node.last_accessed = current_time
                node.access_count += 1
            
        except Exception as e:
            logger.error(f"更新节点访问失败: {e}")
    
    async def cleanup_low_value_nodes(self, min_importance: float = 0.3, 
                                    min_access_count: int = 1, days_threshold: int = 30):
        """清理低价值节点"""
        try:
            cutoff_time = time.time() - (days_threshold * 24 * 3600)
            
            cursor = self.conn.execute("""
                SELECT id FROM graph_nodes 
                WHERE importance < ? AND access_count < ? AND created_time < ?
            """, (min_importance, min_access_count, cutoff_time))
            
            low_value_ids = [row[0] for row in cursor.fetchall()]
            
            if low_value_ids:
                # 删除节点和相关边
                placeholders = ','.join(['?'] * len(low_value_ids))
                
                self.conn.execute(f"DELETE FROM graph_edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})", 
                                low_value_ids + low_value_ids)
                
                self.conn.execute(f"DELETE FROM graph_nodes WHERE id IN ({placeholders})", low_value_ids)
                
                self.conn.execute(f"DELETE FROM entity_node_index WHERE node_id IN ({placeholders})", low_value_ids)
                
                # 清理缓存
                for node_id in low_value_ids:
                    if node_id in self.node_cache:
                        del self.node_cache[node_id]
                
                self.conn.commit()
                
                # 重建索引
                self._build_indexes()
                
                logger.info(f"🧹 清理了 {len(low_value_ids)} 个低价值节点")
                
                return len(low_value_ids)
            
            return 0
            
        except Exception as e:
            logger.error(f"清理低价值节点失败: {e}")
            return 0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        cache_hit_rate = (self.performance_stats["cache_hits"] / 
                         max(self.performance_stats["total_searches"], 1))
        
        return {
            **self.performance_stats,
            "cache_hit_rate": cache_hit_rate,
            "cache_sizes": {
                "node_cache": len(self.node_cache),
                "edge_cache": len(self.edge_cache),
                "search_cache": len(self.search_cache)
            },
            "index_sizes": {
                "entity_index": len(self.entity_index),
                "type_index": len(self.type_index),
                "importance_index": len(self.importance_index)
            }
        }
    
    def close(self):
        """关闭引擎"""
        try:
            self.executor.shutdown(wait=True)
            logger.info("⚡ 优化图谱引擎已关闭")
        except Exception as e:
            logger.error(f"关闭图谱引擎失败: {e}")