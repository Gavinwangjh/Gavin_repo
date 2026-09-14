from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import uuid
import json
import time
import hashlib
import json
from collections import OrderedDict
from typing import Dict, Any
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility
)
from loguru import logger

from src.utils.helpers import get_config, PerformanceTimer
from src.document_loader.loader import DocumentChunk


@dataclass
class SearchHit:
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    chunk_index: int


@dataclass
class InsertResult:
    ids: List[str]
    insert_count: int
    success: bool
    error: Optional[str] = None


class MilvusVectorStore:
    """Milvus向量存储管理器（支持KB管理字段）"""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        index_type: Optional[str] = None,
        metric_type: Optional[str] = None,
        nlist: Optional[int] = None,
        # 第六阶段优化新增参数
        enable_search_cache: bool = True,
        search_cache_size: int = 1000,
        search_cache_ttl: int = 300,  # 5分钟
        nprobe: int = 16
    ):
        self.host = host or get_config("milvus.host", "localhost")
        self.port = port or get_config("milvus.port", 19530)
        self.collection_name = collection_name or get_config("milvus.collection_name", "document_collection")
        self.dimension = dimension or get_config("embedding.dimension", 1024)
        self.index_type = index_type or get_config("milvus.index_type", "IVF_FLAT")
        self.metric_type = metric_type or get_config("milvus.metric_type", "IP")
        self.nlist = nlist or get_config("milvus.nlist", 1024)

        self.collection: Optional[Collection] = None
        self.is_connected = False

        logger.info(f"Milvus向量存储初始化: {self.host}:{self.port}")

        # ===== 第六阶段优化新增 =====
        self.enable_search_cache = enable_search_cache
        self.search_cache_size = search_cache_size
        self.search_cache_ttl = search_cache_ttl
        self.nprobe = nprobe
        
        # 搜索缓存（LRU）
        self.search_cache = OrderedDict()
        
        # 性能统计
        self.performance_stats = {
            "search_count": 0,
            "search_cache_hits": 0,
            "search_cache_misses": 0,
            "avg_search_time": 0.0,
            "insert_count": 0,
            "avg_insert_time": 0.0,
            "total_queries_processed": 0
        }
        
        # 索引参数优化
        self.search_params = {
            "metric_type": self.metric_type,
            "params": {"nprobe": self.nprobe},
        }
        
        logger.info(f"Milvus优化配置: 搜索缓存={enable_search_cache}, 缓存大小={search_cache_size}, nprobe={nprobe}")

    def connect(self) -> bool:
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
            self.is_connected = True
            logger.info(f"Milvus连接成功: {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Milvus连接失败: {str(e)}")
            self.is_connected = False
            return False

    def disconnect(self):
        try:
            connections.disconnect("default")
            self.is_connected = False
            logger.info("Milvus连接已断开")
        except Exception as e:
            logger.warning(f"Milvus断开连接失败: {str(e)}")

    def health_check(self) -> Dict[str, Any]:
        """Return a lightweight Milvus connection status for API health checks."""
        if not self.is_connected:
            return {
                "status": "disconnected",
                "host": self.host,
                "port": self.port,
                "collection_name": self.collection_name,
            }

        try:
            collection_exists = utility.has_collection(self.collection_name)
            return {
                "status": "ok" if collection_exists else "collection_missing",
                "host": self.host,
                "port": self.port,
                "collection_name": self.collection_name,
                "collection_exists": collection_exists,
            }
        except Exception as e:
            return {
                "status": "error",
                "host": self.host,
                "port": self.port,
                "collection_name": self.collection_name,
                "error": str(e),
            }

    def drop_collection(self) -> bool:
        if not self.is_connected:
            if not self.connect():
                return False
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"已删除集合: {self.collection_name}")
            self.collection = None
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {str(e)}")
            return False

    def create_collection(self, drop_existing: bool = False) -> bool:
        if not self.is_connected:
            if not self.connect():
                return False

        try:
            if utility.has_collection(self.collection_name):
                if drop_existing:
                    utility.drop_collection(self.collection_name)
                    logger.info(f"已删除现有集合: {self.collection_name}")
                else:
                    logger.info(f"集合已存在: {self.collection_name}")
                    self.collection = Collection(self.collection_name)
                    return True

            # ✅ KB 管理字段：chunk_hash / doc_version / updated_at / is_deleted
            # is_deleted：优先用 BOOL；若某些环境不兼容可改 INT8（0/1）
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),

                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),

                FieldSchema(name="chunk_hash", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="doc_version", dtype=DataType.INT64),
                FieldSchema(name="updated_at", dtype=DataType.INT64),
                FieldSchema(name="is_deleted", dtype=DataType.BOOL),

                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
            ]

            schema = CollectionSchema(
                fields=fields,
                description=f"RAG document collection with {self.dimension}D embeddings + KB fields"
            )

            self.collection = Collection(name=self.collection_name, schema=schema)
            logger.info(f"集合创建成功: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"创建集合失败: {str(e)}")
            return False

    def create_index(self) -> bool:
        if not self.collection:
            logger.error("集合未初始化")
            return False

        try:
            indexes = self.collection.indexes
            if indexes:
                logger.info(f"索引已存在: {[idx.field_name for idx in indexes]}")
                return True

            index_params = {
                "metric_type": self.metric_type,
                "index_type": self.index_type,
                "params": {"nlist": self.nlist},
            }

            self.collection.create_index(field_name="embedding", index_params=index_params)
            logger.info(f"索引创建成功: {self.index_type}, metric={self.metric_type}")
            return True

        except Exception as e:
            logger.error(f"创建索引失败: {str(e)}")
            return False

    def load_collection(self) -> bool:
        if not self.collection:
            logger.error("集合未初始化")
            return False

        try:
            self.collection.load()
            logger.info(f"集合加载成功: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"加载集合失败: {str(e)}")
            return False

    def initialize(self, force_recreate: bool = False) -> bool:
        try:
            if not self.connect():
                return False
            if not self.create_collection(drop_existing=force_recreate):
                return False
            if not self.create_index():
                return False
            if not self.load_collection():
                return False
            logger.info("Milvus向量存储初始化完成")
            return True
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            return False

    # -------------------------
    # 写入 / 删除
    # -------------------------

    def insert_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        doc_version: int = 1,
        updated_at: Optional[int] = None,
        is_deleted: bool = False
    ) -> InsertResult:
        """插入文档块和对应的嵌入向量（带KB字段）"""
        if not self.collection:
            return InsertResult([], 0, False, "集合未初始化")

        if len(chunks) != len(embeddings):
            return InsertResult([], 0, False, "文档块数量与嵌入向量数量不匹配")

        updated_at = int(updated_at or time.time())

        try:
            with PerformanceTimer(f"插入 {len(chunks)} 个文档块"):
                ids, contents, doc_ids, chunk_indexes, metadatas = [], [], [], [], []
                chunk_hashes, doc_versions, updated_ats, deleted_flags = [], [], [], []

                for chunk in chunks:
                    cid = chunk.chunk_id or str(uuid.uuid4())
                    ids.append(cid)
                    contents.append(chunk.content)
                    doc_ids.append(chunk.doc_id)
                    chunk_indexes.append(int(chunk.chunk_index))

                    md = chunk.metadata or {}
                    metadatas.append(json.dumps(md, ensure_ascii=False))

                    chunk_hashes.append(str(md.get("chunk_hash", ""))[:64])
                    doc_versions.append(int(md.get("doc_version", doc_version)))
                    updated_ats.append(int(md.get("updated_at", updated_at)))
                    deleted_flags.append(bool(md.get("is_deleted", is_deleted)))

                entities = [
                    ids,
                    embeddings,
                    contents,
                    doc_ids,
                    chunk_indexes,
                    chunk_hashes,
                    doc_versions,
                    updated_ats,
                    deleted_flags,
                    metadatas
                ]

                self.collection.insert(entities)
                self.collection.flush()

                logger.info(f"文档插入成功: {len(ids)} 个块")
                return InsertResult(ids=ids, insert_count=len(ids), success=True)

        except Exception as e:
            logger.error(f"插入文档失败: {str(e)}")
            return InsertResult([], 0, False, str(e))

    def delete_by_doc_id(self, doc_id: str) -> bool:
        """物理删除 doc_id 下所有块（overwrite用）"""
        if not self.collection:
            logger.error("集合未初始化")
            return False
        try:
            expr = f'doc_id == "{doc_id}"'
            self.collection.delete(expr)
            self.collection.flush()
            logger.info(f"删除文档成功: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            return False

    def delete_by_ids(self, ids: List[str]) -> bool:
        """物理删除指定chunk ids"""
        if not self.collection:
            logger.error("集合未初始化")
            return False
        if not ids:
            return True

        try:
            # Milvus expr 里 in 的字符串写法
            quoted = ", ".join([f'"{i}"' for i in ids])
            expr = f"id in [{quoted}]"
            self.collection.delete(expr)
            self.collection.flush()
            logger.info(f"删除文档块成功: {len(ids)} 个")
            return True
        except Exception as e:
            logger.error(f"删除文档块失败: {str(e)}")
            return False

    # -------------------------
    # 查询 / 检索
    # -------------------------

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[SearchHit]:
        if not self.collection:
            logger.error("集合未初始化")
            return []

        try:
            search_params = get_config("milvus.search_params", {"nprobe": 16})
            if output_fields is None:
                # 兼容新字段
                output_fields = ["content", "doc_id", "chunk_index", "metadata", "chunk_hash", "doc_version", "updated_at", "is_deleted"]

            # 默认过滤掉 deleted
            expr = filter_expr
            if expr:
                expr = f"({expr}) && (is_deleted == false)"
            else:
                expr = "is_deleted == false"

            with PerformanceTimer("向量搜索"):
                results = self.collection.search(
                    data=[query_embedding],
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k,
                    expr=expr,
                    output_fields=output_fields
                )

            hits: List[SearchHit] = []
            if results and len(results) > 0:
                for hit in results[0]:
                    try:
                        md = json.loads(hit.get("metadata") or "{}")
                        hits.append(SearchHit(
                            id=str(hit.id),
                            score=float(hit.score),
                            content=hit.get("content") or "",
                            metadata=md,
                            doc_id=hit.get("doc_id") or "",
                            chunk_index=int(hit.get("chunk_index") or 0)
                        ))
                    except Exception as e:
                        logger.warning(f"解析搜索结果失败: {str(e)}")

            logger.info(f"搜索完成: 返回 {len(hits)} 个结果")
            return hits

        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            return []

    def get_chunks_by_doc_id(self, doc_id: str, include_deleted: bool = False) -> List[Dict[str, Any]]:
        """用于增量更新：取出 doc_id 的 chunk_hash/id"""
        if not self.collection:
            logger.error("集合未初始化")
            return []

        try:
            expr = f'doc_id == "{doc_id}"'
            if not include_deleted:
                expr = f'({expr}) && (is_deleted == false)'

            rows = self.collection.query(
                expr=expr,
                output_fields=["id", "chunk_hash", "chunk_index", "doc_version", "updated_at", "is_deleted", "metadata"],
                limit=16384
            )

            out = []
            for r in rows or []:
                md = {}
                try:
                    md = json.loads(r.get("metadata") or "{}")
                except Exception:
                    md = {}
                out.append({
                    "id": r.get("id"),
                    "chunk_hash": r.get("chunk_hash") or md.get("chunk_hash", ""),
                    "chunk_index": int(r.get("chunk_index") or 0),
                    "doc_version": int(r.get("doc_version") or md.get("doc_version", 1)),
                    "updated_at": int(r.get("updated_at") or md.get("updated_at", 0)),
                    "is_deleted": bool(r.get("is_deleted")),
                    "metadata": md
                })
            return out

        except Exception as e:
            logger.error(f"get_chunks_by_doc_id 失败: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        if not self.collection:
            return {"error": "集合未初始化"}
        try:
            entity_count = self.collection.num_entities
            return {
                "collection_name": self.collection_name,
                "entity_count": entity_count,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric_type": self.metric_type
            }
        except Exception as e:
            logger.error(f"获取集合统计失败: {str(e)}")
            return {"error": str(e)}
    def search_optimized(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> List[SearchHit]:
        """优化版向量搜索（带缓存）"""
        start_time = time.time()
        self.performance_stats["search_count"] += 1
        
        if not self.collection:
            logger.error("集合未初始化")
            return []
        
        # 生成缓存键
        cache_key = None
        if use_cache and self.enable_search_cache:
            cache_key = self._generate_cache_key(query_embedding, top_k, filter_expr, output_fields)
            
            # 检查缓存
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self.performance_stats["search_cache_hits"] += 1
                logger.debug(f"搜索缓存命中: {cache_key[:20]}...")
                return cached_result
        
        # 缓存未命中
        self.performance_stats["search_cache_misses"] += 1
        
        try:
            # 优化搜索参数
            search_params = {
                "metric_type": self.metric_type,
                "params": {"nprobe": self.nprobe},
            }
            
            if output_fields is None:
                output_fields = ["content", "doc_id", "chunk_index", "metadata", "chunk_hash", "doc_version"]
            
            # 构建过滤表达式
            expr = filter_expr or ""
            if expr and "is_deleted" not in expr:
                expr = f"({expr}) && (is_deleted == false)"
            elif not expr:
                expr = "is_deleted == false"
            
            # 执行搜索
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k * 2,  # 获取更多结果供后续筛选
                expr=expr,
                output_fields=output_fields
            )
            
            # 处理结果
            hits = self._process_search_results(results)
            
            # 截断到top_k
            hits = hits[:top_k]
            
            # 计算搜索时间
            search_time = time.time() - start_time
            self.performance_stats["avg_search_time"] = (
                self.performance_stats["avg_search_time"] * (self.performance_stats["search_count"] - 1) + search_time
            ) / self.performance_stats["search_count"]
            
            # 缓存结果
            if use_cache and self.enable_search_cache and cache_key and hits:
                self._add_to_cache(cache_key, hits)
            
            logger.info(f"向量搜索完成: 返回 {len(hits)} 个结果，耗时 {search_time:.3f}s")
            return hits
            
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            return []
    
    def _generate_cache_key(self, query_embedding: List[float], top_k: int, 
                          filter_expr: Optional[str], output_fields: Optional[List[str]]) -> str:
        """生成缓存键"""
        import hashlib
        
        # 将查询向量转换为字符串表示
        vector_str = ",".join(f"{v:.6f}" for v in query_embedding[:10])  # 只取前10个维度
        
        # 构建缓存数据
        cache_data = {
            "vector": vector_str,
            "top_k": top_k,
            "filter_expr": filter_expr or "",
            "output_fields": ",".join(output_fields) if output_fields else "",
            "nprobe": self.nprobe
        }
        
        # 生成哈希
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[SearchHit]]:
        """从缓存获取结果"""
        if cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            
            # 检查是否过期
            if time.time() - cache_entry["timestamp"] < self.search_cache_ttl:
                # 移动到最近使用的位置
                self.search_cache.move_to_end(cache_key)
                return cache_entry["hits"]
            else:
                # 移除过期缓存
                del self.search_cache[cache_key]
        
        return None
    
    def _add_to_cache(self, cache_key: str, hits: List[SearchHit]):
        """添加结果到缓存"""
        # 如果缓存已满，移除最久未使用的
        if len(self.search_cache) >= self.search_cache_size:
            self.search_cache.popitem(last=False)
        
        # 添加新缓存
        self.search_cache[cache_key] = {
            "hits": hits,
            "timestamp": time.time()
        }
    
    def _process_search_results(self, results) -> List[SearchHit]:
        """处理搜索结果"""
        hits: List[SearchHit] = []
        
        if results and len(results) > 0:
            for hit in results[0]:
                try:
                    metadata_str = hit.get("metadata") or "{}"
                    metadata = json.loads(metadata_str)
                    
                    hits.append(SearchHit(
                        id=str(hit.id),
                        score=float(hit.score),
                        content=hit.get("content") or "",
                        metadata=metadata,
                        doc_id=hit.get("doc_id") or "",
                        chunk_index=int(hit.get("chunk_index") or 0)
                    ))
                except Exception as e:
                    logger.warning(f"解析搜索结果失败: {str(e)}")
        
        return hits
    
    def _insert_batch(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        doc_version: int,
        updated_at: int
    ) -> InsertResult:
        """Insert one prepared batch without flushing the whole collection."""
        try:
            ids, contents, doc_ids, chunk_indexes, metadatas = [], [], [], [], []
            chunk_hashes, doc_versions, updated_ats, deleted_flags = [], [], [], []

            for chunk in chunks:
                cid = chunk.chunk_id or str(uuid.uuid4())
                ids.append(cid)
                contents.append(chunk.content)
                doc_ids.append(chunk.doc_id)
                chunk_indexes.append(int(chunk.chunk_index))

                md = chunk.metadata or {}
                metadatas.append(json.dumps(md, ensure_ascii=False))
                chunk_hashes.append(str(md.get("chunk_hash", ""))[:64])
                doc_versions.append(int(md.get("doc_version", doc_version)))
                updated_ats.append(int(md.get("updated_at", updated_at)))
                deleted_flags.append(bool(md.get("is_deleted", False)))

            self.collection.insert([
                ids,
                embeddings,
                contents,
                doc_ids,
                chunk_indexes,
                chunk_hashes,
                doc_versions,
                updated_ats,
                deleted_flags,
                metadatas,
            ])
            return InsertResult(ids=ids, insert_count=len(ids), success=True)
        except Exception as e:
            logger.error(f"批量插入文档失败: {str(e)}")
            return InsertResult([], 0, False, str(e))

    def insert_documents_optimized(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        doc_version: int = 1,
        batch_size: int = 100,
        use_transaction: bool = True
    ) -> InsertResult:
        """优化版文档插入（分批处理）"""
        start_time = time.time()
        
        if not self.collection:
            return InsertResult([], 0, False, "集合未初始化")
        
        if len(chunks) != len(embeddings):
            return InsertResult([], 0, False, "文档块数量与嵌入向量数量不匹配")
        
        try:
            all_ids = []
            updated_at = int(time.time())
            
            # 分批插入
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                batch_result = self._insert_batch(
                    batch_chunks, batch_embeddings, doc_version, updated_at
                )
                
                if not batch_result.success:
                    return batch_result
                
                all_ids.extend(batch_result.ids)
            
            # 刷新并更新索引（如果需要）
            if use_transaction:
                self.collection.flush()
            
            # 更新性能统计
            insert_time = time.time() - start_time
            self.performance_stats["insert_count"] += 1
            self.performance_stats["avg_insert_time"] = (
                self.performance_stats["avg_insert_time"] * (self.performance_stats["insert_count"] - 1) + insert_time
            ) / self.performance_stats["insert_count"]
            
            logger.info(f"文档插入优化完成: {len(all_ids)} 个块，耗时 {insert_time:.3f}s")
            return InsertResult(ids=all_ids, insert_count=len(all_ids), success=True)
            
        except Exception as e:
            logger.error(f"优化插入文档失败: {str(e)}")
            return InsertResult([], 0, False, str(e))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        cache_hit_rate = 0
        if self.performance_stats["search_count"] > 0:
            cache_hit_rate = self.performance_stats["search_cache_hits"] / self.performance_stats["search_count"]
        
        collection_stats = self.get_collection_stats()
        
        return {
            **self.performance_stats,
            "search_cache_hit_rate": cache_hit_rate,
            "search_cache_size": len(self.search_cache),
            "max_search_cache_size": self.search_cache_size,
            "collection_stats": collection_stats,
            "optimization_enabled": {
                "search_cache": self.enable_search_cache,
                "nprobe": self.nprobe,
                "search_cache_ttl": self.search_cache_ttl
            }
        }
    
    def clear_search_cache(self):
        """清空搜索缓存"""
        self.search_cache.clear()
        logger.info("搜索缓存已清空")
    
    def optimize_collection(self):
        """优化集合性能"""
        if not self.collection:
            return False
        
        try:
            # 1. 重建索引（如果需要）
            self.collection.release()
            self.collection.load()
            
            # 2. 压缩集合
            self.collection.compact()
            
            # 3. 清空缓存
            self.clear_search_cache()
            
            logger.info("集合优化完成")
            return True
            
        except Exception as e:
            logger.error(f"集合优化失败: {str(e)}")
            return False
