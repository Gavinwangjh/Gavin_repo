from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import json
import time
import hashlib
from collections import defaultdict
from enum import Enum
from functools import lru_cache
# 在文件顶部添加导入
from src.rerank.reranker import HybridReranker, PolicyDocumentReranker, SimpleScoreReranker
from loguru import logger

from src.vector_store.milvus_store import MilvusVectorStore, SearchHit
from src.embedding.embedder import EmbeddingManager
from src.utils.helpers import get_config, PerformanceTimer


class SearchMethod(str, Enum):
    """检索方法枚举"""
    HYBRID = "hybrid"
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID_RRF = "hybrid_rrf"


@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    query: str
    hits: List[SearchHit]
    dense_hits: List[SearchHit]
    sparse_hits: List[SearchHit]
    total_hits: int
    retrieval_time: float
    method: SearchMethod
    reranked: bool = False


@dataclass
class HybridRetrieverConfig:
    """混合检索器配置"""
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    similarity_threshold: float = 0.1
    initial_top_k: int = 50  # 初始召回数量，扩大召回
    final_top_k: int = 5     # 最终返回数量
    rrf_k: int = 60          # RRF参数
    use_rrf: bool = True     # 是否使用RRF融合
    use_structured_boost: bool = True  # 是否使用结构化boost
    boost_multiplier: float = 1.5      # 结构化boost倍数


class EnhancedKeywordExtractor:
    """增强版关键词提取器，支持结构化信息提取"""
    
    def __init__(self):
        jieba.setLogLevel('WARNING')
        
        # 法律条文相关关键词增强
        self.law_keywords = {
            '条', '款', '项', '目', '章', '节', '编', '部分',
            '第一', '第二', '第三', '第四', '第五', '第六', '第七', '第八', '第九', '第十',
            '（一）', '（二）', '（三）', '（四）', '（五）', '（六）', '（七）', '（八）', '（九）', '（十）'
        }
        
        self.policy_keywords = {
            "适用范围", "申报条件", "申请条件", "支持对象", "扶持对象", "补贴对象",
            "扶持标准", "补助标准", "奖励标准", "资助标准", "认定标准",
            "申报材料", "申请材料", "办理流程", "申报流程", "审批流程",
            "责任部门", "主管部门", "牵头部门", "职责分工", "监督管理",
            "实施期限", "有效期", "截止时间", "附则", "解释权",
            "处罚", "违规", "考核", "验收", "公示", "备案",
        }

        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
            '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
            '会', '着', '没有', '看', '好', '自己', '这', '那', '它', '他', '她'
        }
        
        logger.info("增强版关键词提取器初始化完成")
    
    def extract_keywords(self, text: str, top_k: int = 15) -> List[str]:
        """提取关键词，增强法律条文识别"""
        try:
            # 1. 提取TF-IDF关键词
            keywords = jieba.analyse.extract_tags(
                text,
                topK=top_k * 2,  # 多提取一些用于后续过滤
                withWeight=False,
                allowPOS=('n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd', 'm')
            )
            
            # 2. 提取法律条文编号
            article_patterns = [
                r'第[一二三四五六七八九十\d]+条',
                r'第[一二三四五六七八九十\d]+款',
                r'第[一二三四五六七八九十\d]+项',
                r'（[一二三四五六七八九十]+）',
                r'\d+\.\d+',  # 1.1, 2.3 等编号
            ]
            
            article_numbers = []
            for pattern in article_patterns:
                matches = re.findall(pattern, text)
                article_numbers.extend(matches)
            
            # 3. 过滤和组合
            filtered_keywords = [
                kw for kw in keywords 
                if kw not in self.stop_words and len(kw) > 1
            ]
            
            # 优先保留法律条文关键词
            law_related = [kw for kw in filtered_keywords if kw in self.law_keywords]
            policy_related = [kw for kw in self.policy_keywords if kw in text]
            other_keywords = [
                kw for kw in filtered_keywords
                if kw not in self.law_keywords and kw not in self.policy_keywords
            ]
            
            # 组合结果：法律条文编号 + 法律关键词 + 其他关键词
            final_keywords = article_numbers + law_related + policy_related + other_keywords[:top_k]
            
            return final_keywords[:top_k]
            
        except Exception as e:
            logger.warning(f"关键词提取失败: {str(e)}")
            return []

    def extract_entities(self, text: str) -> List[str]:
        """Backward-compatible lightweight entity extraction."""
        if not text:
            return []

        patterns = [
            r"\d+(?:\.\d+)?%?",
            r"[A-Za-z][A-Za-z0-9_+\-.]{1,}",
            r"第[一二三四五六七八九十百千万\d]+[条章节款项]",
            r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*",
        ]

        entities: List[str] = []
        seen: Set[str] = set()
        for pattern in patterns:
            for match in re.findall(pattern, text):
                value = match.rstrip("%")
                if len(value) > 1 and value not in seen:
                    seen.add(value)
                    entities.append(value)

        return entities
    
    def extract_article_number(self, text: str) -> Optional[str]:
        """提取文本中的条文编号"""
        patterns = [
            r'第[一二三四五六七八九十\d]+条',
            r'第[一二三四五六七八九十\d]+款',
            r'第[一二三四五六七八九十\d]+项',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        return None


class EnhancedSparseRetriever:
    """增强版稀疏检索器，支持结构化信息boost"""
    
    def __init__(self, config: Optional[HybridRetrieverConfig] = None):
        self.config = config or HybridRetrieverConfig()
        self.keyword_extractor = EnhancedKeywordExtractor()
        self.vectorizer = None
        self.document_vectors = None
        self.document_metadata = []
        
        # 缓存
        self.keyword_cache = {}
        
        logger.info("增强版稀疏检索器初始化完成")
    
    def build_index(self, documents: List[SearchHit]):
        """构建增强版TF-IDF索引"""
        try:
            with PerformanceTimer("构建增强版TF-IDF索引"):
                # 准备文档文本（增强版：包含结构化信息）
                enhanced_texts = []
                for hit in documents:
                    # 将结构化信息也加入文本中
                    metadata_text = self._extract_metadata_text(hit.metadata)
                    enhanced_text = f"{hit.content} {metadata_text}"
                    enhanced_texts.append(enhanced_text)
                
                # 自定义分词器（增强版）
                def enhanced_tokenizer(text):
                    # 先提取法律条文编号
                    article_numbers = re.findall(r'第[一二三四五六七八九十\d]+[条款项]', text)
                    # 再分词
                    tokens = list(jieba.cut(text, cut_all=False))
                    # 合并结果，确保条文编号不被切分
                    all_tokens = article_numbers + tokens
                    return all_tokens
                
                # 构建TF-IDF向量化器（参数优化）
                self.vectorizer = TfidfVectorizer(
                    tokenizer=enhanced_tokenizer,
                    stop_words=list(self.keyword_extractor.stop_words),
                    max_features=self.config.initial_top_k * 10,  # 根据召回量调整
                    ngram_range=(1, 3),  # 扩展到3-gram，捕捉更长的法律术语
                    min_df=2,  # 提高min_df，减少噪声
                    max_df=0.85,  # 降低max_df，保留更多特色词
                    sublinear_tf=True  # 使用次线性TF缩放
                )
                
                # 训练并转换文档
                self.document_vectors = self.vectorizer.fit_transform(enhanced_texts)
                
                # 存储文档元数据
                self.document_metadata = documents
                
                logger.info(f"增强版TF-IDF索引构建完成: {len(documents)} 个文档, {self.document_vectors.shape[1]} 个特征")
        
        except Exception as e:
            logger.error(f"构建增强版TF-IDF索引失败: {str(e)}")
            raise
    
    def _extract_metadata_text(self, metadata: Dict[str, Any]) -> str:
        """从metadata中提取文本信息"""
        text_parts = []
        
        # 提取可能的条文编号
        if 'article_number' in metadata:
            text_parts.append(str(metadata['article_number']))
        if 'title' in metadata:
            text_parts.append(str(metadata['title']))
        if 'section_path' in metadata:
            text_parts.append(str(metadata['section_path']))
        
        return " ".join(text_parts)
    
    def search(self, query: str, top_k: int = 10, boost_metadata: Optional[Dict[str, Any]] = None) -> List[SearchHit]:
        """增强版TF-IDF检索，支持结构化boost"""
        if self.vectorizer is None or self.document_vectors is None:
            logger.warning("TF-IDF索引未构建，返回空结果")
            return []
        
        try:
            with PerformanceTimer("增强版TF-IDF检索"):
                # 增强查询：加入提取的法律条文编号
                article_number = self.keyword_extractor.extract_article_number(query)
                enhanced_query = query
                if article_number:
                    enhanced_query = f"{query} {article_number}"
                
                # 将查询转换为向量
                query_vector = self.vectorizer.transform([enhanced_query])
                
                # 计算相似度
                similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
                
                # 应用结构化boost
                if self.config.use_structured_boost and boost_metadata:
                    similarities = self._apply_structured_boost(
                        similarities, boost_metadata
                    )
                
                # 获取top-k结果
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                # 构建结果
                results = []
                for idx in top_indices:
                    if similarities[idx] <= 0:  # 过滤掉负分
                        continue
                    
                    hit = self.document_metadata[idx]
                    result_hit = SearchHit(
                        id=hit.id,
                        score=float(similarities[idx]),
                        content=hit.content,
                        metadata=hit.metadata,
                        doc_id=hit.doc_id,
                        chunk_index=hit.chunk_index
                    )
                    results.append(result_hit)
                
                logger.debug(f"增强版TF-IDF检索完成: 返回 {len(results)} 个结果")
                return results
        
        except Exception as e:
            logger.error(f"增强版TF-IDF检索失败: {str(e)}")
            return []
    
    def _apply_structured_boost(self, scores: np.ndarray, boost_metadata: Dict[str, Any]) -> np.ndarray:
        """应用结构化boost"""
        boosted_scores = scores.copy()
        
        for idx, hit in enumerate(self.document_metadata):
            boost_factor = 1.0
            
            # 1. 条文编号匹配boost
            if 'article_number' in hit.metadata and 'target_article' in boost_metadata:
                if hit.metadata['article_number'] == boost_metadata['target_article']:
                    boost_factor *= self.config.boost_multiplier
            
            # 2. 文档类型匹配boost
            if 'doc_type' in hit.metadata and 'target_doc_type' in boost_metadata:
                if hit.metadata['doc_type'] == boost_metadata['target_doc_type']:
                    boost_factor *= 1.2
            
            # 3. 标题匹配boost
            if 'title' in hit.metadata and 'query_keywords' in boost_metadata:
                title = hit.metadata['title']
                keywords = boost_metadata['query_keywords']
                if any(keyword in title for keyword in keywords):
                    boost_factor *= 1.1
            
            boosted_scores[idx] *= boost_factor
        
        return boosted_scores
    
    @lru_cache(maxsize=1000)
    def cached_keyword_search(self, query: str, top_k: int = 10) -> List[SearchHit]:
        """带缓存的关键词搜索"""
        return self.keyword_search(query, top_k)
    
    def keyword_search(self, query: str, documents: List[SearchHit], top_k: int = 10) -> List[SearchHit]:
        """增强版关键词检索"""
        try:
            # 提取查询关键词（增强版）
            query_keywords = self.keyword_extractor.extract_keywords(query, top_k=20)
            if not query_keywords:
                return []
            
            # 提取查询中的条文编号
            query_article = self.keyword_extractor.extract_article_number(query)
            
            matches = []
            
            for hit in documents:
                # 检查metadata中的条文编号
                hit_article = hit.metadata.get('article_number') if hit.metadata else None
                
                # 如果查询中有条文编号，且文档有匹配的条文编号，直接高权重
                if query_article and hit_article and query_article in hit_article:
                    result_hit = SearchHit(
                        id=hit.id,
                        score=2.0,  # 高权重
                        content=hit.content,
                        metadata=hit.metadata,
                        doc_id=hit.doc_id,
                        chunk_index=hit.chunk_index
                    )
                    matches.append(result_hit)
                    continue
                
                # 普通关键词匹配
                doc_keywords = self.keyword_extractor.extract_keywords(hit.content, top_k=30)
                
                matched_keywords = set(query_keywords) & set(doc_keywords)
                
                if matched_keywords:
                    # 增强版分数计算：考虑词频和位置
                    score = 0.0
                    for keyword in matched_keywords:
                        # 查询权重
                        try:
                            query_rank = query_keywords.index(keyword)
                            query_weight = 1.0 / (query_rank + 1)
                        except ValueError:
                            query_weight = 0.5
                        
                        # 文档权重
                        try:
                            doc_rank = doc_keywords.index(keyword)
                            doc_weight = 1.0 / (doc_rank + 1)
                        except ValueError:
                            doc_weight = 0.5
                        
                        # 词性权重：名词和动词权重更高
                        pos_weight = 1.2 if len(keyword) > 2 else 1.0
                        
                        score += query_weight * doc_weight * pos_weight
                    
                    # 归一化
                    score = score / max(len(query_keywords), 1)
                    
                    # 如果有关键词匹配但分数低，给予基础分
                    if score < 0.1:
                        score = 0.1 + len(matched_keywords) * 0.05
                    
                    result_hit = SearchHit(
                        id=hit.id,
                        score=score,
                        content=hit.content,
                        metadata=hit.metadata,
                        doc_id=hit.doc_id,
                        chunk_index=hit.chunk_index
                    )
                    matches.append(result_hit)
            
            # 按分数排序
            matches.sort(key=lambda x: x.score, reverse=True)
            return matches[:top_k]
        
        except Exception as e:
            logger.error(f"增强版关键词检索失败: {str(e)}")
            return []


class HybridRetriever:
    """增强版混合检索器 - 实现第四阶段优化"""
    
    def __init__(
        self,
        vector_store: MilvusVectorStore,
        embedding_manager: EmbeddingManager,
        config: Optional[HybridRetrieverConfig] = None,
        **kwargs
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.config = config or HybridRetrieverConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.sparse_retriever = EnhancedSparseRetriever(self.config)
        self.is_sparse_index_built = False
        
        # 新增：重排序器 🔥
        self.reranker = HybridReranker(
            rerankers=[PolicyDocumentReranker(), SimpleScoreReranker()],
            weights=[0.7, 0.3]
        )
        self.enable_rerank = True  # 重排序开关
        
        # 缓存
        self.query_cache = {}
        
        logger.info(f"增强版混合检索器初始化: 初始召回={self.config.initial_top_k}, RRF={self.config.use_rrf}, Rerank={self.enable_rerank}")
    
    def _fallback_sparse_documents_from_search(self, limit: int = 10000) -> List[SearchHit]:
        """Fetch candidate documents through vector search when collection.query is unavailable."""
        try:
            dimension = int(getattr(self.embedding_manager, "dimension", 0) or get_config("embedding.dimension", 1024))
            if dimension <= 0:
                dimension = 1024

            hits = self.vector_store.search([0.0] * dimension, top_k=limit)
            if not hits:
                return []

            return [hit for hit in hits if isinstance(hit, SearchHit)]
        except Exception as e:
            logger.warning(f"稀疏索引回退获取文档失败: {str(e)}")
            return []

    def build_sparse_index(self, force_rebuild: bool = False):
        """构建增强版稀疏检索索引"""
        if self.is_sparse_index_built and not force_rebuild:
            logger.info("稀疏检索索引已存在，跳过构建")
            return
        
        try:
            # 使用更简单的方式获取所有文档
            logger.info("获取所有文档构建稀疏索引...")
            
            if not self.vector_store.collection:
                logger.error("向量存储集合未初始化")
                return
            
            all_hits = []
            
            try:
                # 方法1：直接获取所有数据（不进行ID过滤）
                logger.info("尝试获取所有文档...")
                
                # 首先获取总数量
                try:
                    total_count = self.vector_store.collection.num_entities
                    logger.info(f"集合总文档数: {total_count}")
                    
                    if total_count == 0:
                        logger.warning("集合为空，无法构建稀疏索引")
                        return
                    
                    # 如果数据量不大，一次性获取
                    if total_count <= 10000:
                        batch_results = self.vector_store.collection.query(
                            expr="is_deleted == false",  # 只获取未删除的文档
                            output_fields=["id", "content", "doc_id", "chunk_index", "metadata"],
                            limit=total_count
                        )
                    else:
                        # 数据量太大，分批处理（使用分页）
                        logger.warning(f"数据量较大({total_count})，分批处理...")
                        batch_size = 5000
                        offset = 0
                        batch_results = []
                        
                        while True:
                            # 使用limit和offset参数（如果支持）
                            try:
                                batch = self.vector_store.collection.query(
                                    expr="is_deleted == false",
                                    output_fields=["id", "content", "doc_id", "chunk_index", "metadata"],
                                    limit=batch_size,
                                    offset=offset
                                )
                                
                                if not batch:
                                    break
                                
                                batch_results.extend(batch)
                                offset += len(batch)
                                logger.info(f"已获取 {offset} 个文档...")
                                
                                if len(batch) < batch_size:
                                    break
                                    
                            except Exception as batch_e:
                                logger.error(f"分批获取失败: {str(batch_e)}")
                                break
                    
                except Exception as count_e:
                    logger.warning(f"无法获取文档总数: {str(count_e)}")
                    # 回退到简单方式
                    batch_results = self.vector_store.collection.query(
                        expr="is_deleted == false",
                        output_fields=["id", "content", "doc_id", "chunk_index", "metadata"],
                        limit=10000  # 限制数量
                    )
                
                if not batch_results:
                    logger.warning("没有获取到任何文档")
                    return
                
                # 转换为SearchHit格式
                for result in batch_results:
                    try:
                        metadata = json.loads(result.get('metadata', '{}'))
                        
                        hit = SearchHit(
                            id=str(result.get('id', '')),
                            score=1.0,
                            content=result.get('content', ''),
                            metadata=metadata,
                            doc_id=result.get('doc_id', ''),
                            chunk_index=int(result.get('chunk_index', 0))
                        )
                        all_hits.append(hit)
                    except Exception as parse_e:
                        logger.warning(f"解析文档结果失败: {str(parse_e)}")
                        continue
                
                logger.info(f"已获取 {len(all_hits)} 个文档用于构建稀疏索引")
                
            except Exception as query_e:
                logger.error(f"查询文档失败: {str(query_e)}")
                all_hits = self._fallback_sparse_documents_from_search()
            
            if not all_hits:
                logger.warning("没有有效文档用于构建稀疏索引")
                return
            
            # 构建增强版TF-IDF索引
            self.sparse_retriever.build_index(all_hits)
            self.is_sparse_index_built = True
            
            logger.info(f"增强版稀疏检索索引构建完成，包含 {len(all_hits)} 个文档")
        
        except Exception as e:
            logger.error(f"构建稀疏检索索引失败: {str(e)}")
        
    def dense_search(self, query: str, top_k: int = None, use_filter: bool = True) -> List[SearchHit]:
        """增强版稠密向量检索 - 扩大召回"""
        top_k = top_k or self.config.initial_top_k
        
        try:
            with PerformanceTimer("增强版稠密向量检索"):
                # 生成查询嵌入
                query_embedding = self.embedding_manager.embed_query(query)
                
                # 扩大召回，不预先过滤
                hits = self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k * 3,  # 扩大召回
                    filter_expr=None  # 先不进行过滤
                )
                
                # 应用相似度阈值（可选）
                if use_filter:
                    filtered_hits = [
                        hit for hit in hits 
                        if hit.score >= self.config.similarity_threshold
                    ]
                else:
                    filtered_hits = hits
                
                # 结构化boost
                if self.config.use_structured_boost:
                    filtered_hits = self._apply_dense_boost(query, filtered_hits)
                
                logger.debug(f"增强版稠密检索完成: {len(filtered_hits)} 个结果")
                return filtered_hits[:top_k]  # 返回指定数量
        
        except Exception as e:
            logger.error(f"增强版稠密向量检索失败: {str(e)}")
            return []
    
    def _apply_dense_boost(self, query: str, hits: List[SearchHit]) -> List[SearchHit]:
        """对稠密检索结果应用结构化boost"""
        # 提取查询中的条文编号
        article_number = self.sparse_retriever.keyword_extractor.extract_article_number(query)
        
        if not article_number:
            return hits
        
        boosted_hits = []
        for hit in hits:
            boosted_score = hit.score
            
            # 检查metadata中的条文编号
            hit_article = hit.metadata.get('article_number') if hit.metadata else None
            
            # 如果匹配条文编号，给予boost
            if hit_article and article_number in hit_article:
                boosted_score *= self.config.boost_multiplier
            
            # 创建新的SearchHit
            boosted_hit = SearchHit(
                id=hit.id,
                score=boosted_score,
                content=hit.content,
                metadata=hit.metadata,
                doc_id=hit.doc_id,
                chunk_index=hit.chunk_index
            )
            boosted_hits.append(boosted_hit)
        
        # 重新排序
        boosted_hits.sort(key=lambda x: x.score, reverse=True)
        return boosted_hits
    
    def sparse_search(self, query: str, top_k: int = None) -> List[SearchHit]:
        """增强版稀疏检索"""
        top_k = top_k or self.config.initial_top_k
        
        try:
            with PerformanceTimer("增强版稀疏检索"):
                if not self.is_sparse_index_built:
                    logger.warning("稀疏索引未构建，尝试构建...")
                    self.build_sparse_index()
                
                if not self.is_sparse_index_built:
                    logger.warning("稀疏索引构建失败，返回空结果")
                    return []
                
                # 提取结构化信息用于boost
                boost_metadata = self._extract_boost_metadata(query)
                
                # 使用增强版TF-IDF检索
                hits = self.sparse_retriever.search(
                    query, 
                    top_k=top_k,
                    boost_metadata=boost_metadata
                )
                
                logger.debug(f"增强版稀疏检索完成: 返回 {len(hits)} 个结果")
                return hits
        
        except Exception as e:
            logger.error(f"增强版稀疏检索失败: {str(e)}")
            return []
    
    def _extract_boost_metadata(self, query: str) -> Dict[str, Any]:
        """从查询中提取结构化信息用于boost"""
        boost_metadata = {}
        
        # 提取条文编号
        article_number = self.sparse_retriever.keyword_extractor.extract_article_number(query)
        if article_number:
            boost_metadata['target_article'] = article_number
        
        # 提取关键词
        keywords = self.sparse_retriever.keyword_extractor.extract_keywords(query, top_k=10)
        if keywords:
            boost_metadata['query_keywords'] = keywords
        
        # 可能的文档类型
        law_keywords = {'法', '条例', '规定', '办法', '细则'}
        if any(kw in query for kw in law_keywords):
            boost_metadata['target_doc_type'] = 'law'
        
        return boost_metadata
    
    def hybrid_search_rrf(self, query: str, top_k: int = None) -> RetrievalResult:
        """使用RRF的混合检索"""
        top_k = top_k or self.config.final_top_k
        start_time = time.time()
        
        try:
            logger.info(f"开始RRF混合检索: query='{query}', top_k={top_k}")
            
            # 扩大召回
            dense_hits = self.dense_search(query, top_k=self.config.initial_top_k, use_filter=False)
            sparse_hits = self.sparse_search(query, top_k=self.config.initial_top_k)
            
            # 使用RRF融合
            fused_hits = self._fuse_results_rrf(dense_hits, sparse_hits, top_k)
            
            # 新增：重排序 🔥
            if self.enable_rerank and self.reranker and fused_hits:
                rerank_start = time.time()
                fused_hits = self.reranker.rerank(query, fused_hits, top_k)
                rerank_time = time.time() - rerank_start
                logger.info(f"重排序完成: 输入={len(fused_hits)}个, 耗时={rerank_time:.3f}s")
            
            retrieval_time = time.time() - start_time
            
            result = RetrievalResult(
                query=query,
                hits=fused_hits,
                dense_hits=dense_hits[:top_k],
                sparse_hits=sparse_hits[:top_k],
                total_hits=len(fused_hits),
                retrieval_time=retrieval_time,
                method=SearchMethod.HYBRID_RRF,
                reranked=self.enable_rerank  # 新增：标记是否经过重排序
            )
            
            logger.info(f"RRF混合检索完成: 稠密={len(dense_hits)}, 稀疏={len(sparse_hits)}, 融合={len(fused_hits)}, 重排序={self.enable_rerank}, 耗时={retrieval_time:.2f}s")
            return result
        
        except Exception as e:
            logger.error(f"RRF混合检索失败: {str(e)}")
            return RetrievalResult(
                query=query,
                hits=[],
                dense_hits=[],
                sparse_hits=[],
                total_hits=0,
                retrieval_time=time.time() - start_time,
                method=SearchMethod.HYBRID_RRF
            )
        
    def _fuse_results_rrf(self, dense_hits: List[SearchHit], sparse_hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        """使用RRF (Reciprocal Rank Fusion) 算法融合结果"""
        try:
            # 使用字典记录每个文档的排名
            rank_map = defaultdict(lambda: {'dense_rank': float('inf'), 'sparse_rank': float('inf'), 'hit': None})
            
            # 记录稠密检索排名
            for rank, hit in enumerate(dense_hits):
                if hit.id not in rank_map:
                    rank_map[hit.id]['hit'] = hit
                rank_map[hit.id]['dense_rank'] = rank + 1
            
            # 记录稀疏检索排名
            for rank, hit in enumerate(sparse_hits):
                if hit.id not in rank_map:
                    rank_map[hit.id]['hit'] = hit
                rank_map[hit.id]['sparse_rank'] = rank + 1
            
            # 计算RRF分数
            fused_results = []
            for doc_id, data in rank_map.items():
                rrf_score = 0.0
                
                # 稠密检索RRF分数
                if data['dense_rank'] != float('inf'):
                    rrf_score += 1.0 / (self.config.rrf_k + data['dense_rank'])
                
                # 稀疏检索RRF分数
                if data['sparse_rank'] != float('inf'):
                    rrf_score += 1.0 / (self.config.rrf_k + data['sparse_rank'])
                
                # 创建融合后的结果
                fused_hit = SearchHit(
                    id=data['hit'].id,
                    score=rrf_score,
                    content=data['hit'].content,
                    metadata=data['hit'].metadata,
                    doc_id=data['hit'].doc_id,
                    chunk_index=data['hit'].chunk_index
                )
                fused_results.append(fused_hit)
            
            # 按RRF分数排序
            fused_results.sort(key=lambda x: x.score, reverse=True)
            
            return fused_results[:top_k]
        
        except Exception as e:
            logger.error(f"RRF结果融合失败: {str(e)}")
            # 降级处理：返回稠密检索结果
            return dense_hits[:top_k]
    
    def hybrid_search_weighted(self, query: str, top_k: int = None) -> RetrievalResult:
        """使用加权分数的混合检索（原方法）"""
        top_k = top_k or self.config.final_top_k
        start_time = time.time()
        
        try:
            logger.info(f"开始加权混合检索: query='{query}', top_k={top_k}")
            
            # 扩大召回
            dense_hits = self.dense_search(query, top_k=self.config.initial_top_k)
            sparse_hits = self.sparse_search(query, top_k=self.config.initial_top_k)
            
            # 使用加权融合
            fused_hits = self._fuse_results_weighted(dense_hits, sparse_hits, top_k)
            
            retrieval_time = time.time() - start_time
            
            result = RetrievalResult(
                query=query,
                hits=fused_hits,
                dense_hits=dense_hits[:top_k],
                sparse_hits=sparse_hits[:top_k],
                total_hits=len(fused_hits),
                retrieval_time=retrieval_time,
                method=SearchMethod.HYBRID
            )
            
            logger.info(f"加权混合检索完成: 稠密={len(dense_hits)}, 稀疏={len(sparse_hits)}, 融合={len(fused_hits)}, 耗时={retrieval_time:.2f}s")
            return result
        
        except Exception as e:
            logger.error(f"加权混合检索失败: {str(e)}")
            return RetrievalResult(
                query=query,
                hits=[],
                dense_hits=[],
                sparse_hits=[],
                total_hits=0,
                retrieval_time=time.time() - start_time,
                method=SearchMethod.HYBRID
            )
    
    def _fuse_results_weighted(self, dense_hits: List[SearchHit], sparse_hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        """使用加权分数融合结果（兼容原方法）"""
        try:
            # 归一化分数
            dense_scores = [hit.score for hit in dense_hits]
            sparse_scores = [hit.score for hit in sparse_hits]
            
            if dense_scores:
                dense_max = max(dense_scores)
                dense_min = min(dense_scores)
                if dense_max > dense_min:
                    dense_scores_normalized = [(s - dense_min) / (dense_max - dense_min) for s in dense_scores]
                else:
                    dense_scores_normalized = [1.0] * len(dense_scores)
            else:
                dense_scores_normalized = []
            
            if sparse_scores:
                sparse_max = max(sparse_scores)
                sparse_min = min(sparse_scores)
                if sparse_max > sparse_min:
                    sparse_scores_normalized = [(s - sparse_min) / (sparse_max - sparse_min) for s in sparse_scores]
                else:
                    sparse_scores_normalized = [1.0] * len(sparse_scores)
            else:
                sparse_scores_normalized = []
            
            # 创建分数映射
            score_map = {}
            
            # 处理稠密检索结果
            for idx, hit in enumerate(dense_hits):
                if hit.id not in score_map:
                    score_map[hit.id] = {
                        'hit': hit,
                        'dense_score': 0.0,
                        'sparse_score': 0.0
                    }
                
                if dense_scores_normalized:
                    score_map[hit.id]['dense_score'] = dense_scores_normalized[idx]
                else:
                    score_map[hit.id]['dense_score'] = hit.score
            
            # 处理稀疏检索结果
            for idx, hit in enumerate(sparse_hits):
                if hit.id not in score_map:
                    score_map[hit.id] = {
                        'hit': hit,
                        'dense_score': 0.0,
                        'sparse_score': 0.0
                    }
                
                if sparse_scores_normalized:
                    score_map[hit.id]['sparse_score'] = sparse_scores_normalized[idx]
                else:
                    score_map[hit.id]['sparse_score'] = hit.score
            
            # 计算融合分数
            fused_results = []
            for item_id, data in score_map.items():
                fused_score = (
                    self.config.dense_weight * data['dense_score'] + 
                    self.config.sparse_weight * data['sparse_score']
                )
                
                fused_hit = SearchHit(
                    id=data['hit'].id,
                    score=fused_score,
                    content=data['hit'].content,
                    metadata=data['hit'].metadata,
                    doc_id=data['hit'].doc_id,
                    chunk_index=data['hit'].chunk_index
                )
                fused_results.append(fused_hit)
            
            # 按融合分数排序
            fused_results.sort(key=lambda x: x.score, reverse=True)
            
            return fused_results[:top_k]
        
        except Exception as e:
            logger.error(f"加权结果融合失败: {str(e)}")
            return dense_hits[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = None) -> RetrievalResult:
        """Backward-compatible weighted hybrid search."""
        return self.hybrid_search_weighted(query, top_k)

    def _fuse_results(self, dense_hits: List[SearchHit], sparse_hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        """Backward-compatible result fusion helper."""
        return self._fuse_results_weighted(dense_hits, sparse_hits, top_k)

    def search(self, query: str, top_k: int = None, method: Union[str, SearchMethod] = "hybrid_rrf") -> RetrievalResult:
        """统一检索接口"""
        top_k = top_k or self.config.final_top_k
        
        # 缓存检查
        cache_key = f"{query}_{top_k}_{method}"
        if cache_key in self.query_cache:
            logger.debug(f"从缓存返回检索结果: {cache_key}")
            return self.query_cache[cache_key]
        
        # 执行检索
        if isinstance(method, str):
            method = SearchMethod(method)
        
        if method == SearchMethod.DENSE:
            result = self._dense_only_search(query, top_k)
        elif method == SearchMethod.SPARSE:
            result = self._sparse_only_search(query, top_k)
        elif method == SearchMethod.HYBRID_RRF:
            result = self.hybrid_search_rrf(query, top_k)
        else:  # HYBRID
            result = self.hybrid_search(query, top_k)
        
        # 缓存结果
        self.query_cache[cache_key] = result
        if len(self.query_cache) > 100:  # 限制缓存大小
            self.query_cache.pop(next(iter(self.query_cache)))
        
        return result
    
    def _dense_only_search(self, query: str, top_k: int) -> RetrievalResult:
        """仅稠密检索"""
        start_time = time.time()
        hits = self.dense_search(query, top_k)
        
        return RetrievalResult(
            query=query,
            hits=hits,
            dense_hits=hits,
            sparse_hits=[],
            total_hits=len(hits),
            retrieval_time=time.time() - start_time,
            method=SearchMethod.DENSE
        )
    
    def _sparse_only_search(self, query: str, top_k: int) -> RetrievalResult:
        """仅稀疏检索"""
        start_time = time.time()
        hits = self.sparse_search(query, top_k)
        
        return RetrievalResult(
            query=query,
            hits=hits,
            dense_hits=[],
            sparse_hits=hits,
            total_hits=len(hits),
            retrieval_time=time.time() - start_time,
            method=SearchMethod.SPARSE
        )
    
    @property
    def dense_weight(self) -> float:
        return self.config.dense_weight

    @property
    def sparse_weight(self) -> float:
        return self.config.sparse_weight

    @property
    def similarity_threshold(self) -> float:
        return self.config.similarity_threshold

    def update_weights(self, dense_weight: float, sparse_weight: float):
        """Backward-compatible weight update."""
        total = dense_weight + sparse_weight
        if total <= 0:
            raise ValueError("dense_weight and sparse_weight must sum to a positive value")
        self.config.dense_weight = dense_weight / total
        self.config.sparse_weight = sparse_weight / total
        self.clear_cache()

    def update_config(self, config: HybridRetrieverConfig):
        """更新检索器配置"""
        self.config = config
        logger.info(f"检索器配置已更新: {config}")
    
    def clear_cache(self):
        """清空缓存"""
        self.query_cache.clear()
        logger.info("检索缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        return {
            "dense_weight": self.config.dense_weight,
            "sparse_weight": self.config.sparse_weight,
            "similarity_threshold": self.config.similarity_threshold,
            "sparse_index_built": self.is_sparse_index_built,
            "embedding_dimension": self.embedding_manager.dimension,
            "config": {
                "dense_weight": self.config.dense_weight,
                "sparse_weight": self.config.sparse_weight,
                "initial_top_k": self.config.initial_top_k,
                "final_top_k": self.config.final_top_k,
                "use_rrf": self.config.use_rrf,
                "rrf_k": self.config.rrf_k,
                "use_structured_boost": self.config.use_structured_boost,
                "boost_multiplier": self.config.boost_multiplier
            },
            "status": {
                "sparse_index_built": self.is_sparse_index_built,
                "cache_size": len(self.query_cache),
                "embedding_dimension": self.embedding_manager.dimension
            }
        }


# Backward-compatible names used by older tests and scripts.
KeywordExtractor = EnhancedKeywordExtractor
SparseRetriever = EnhancedSparseRetriever
