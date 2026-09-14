import pytest
from unittest.mock import Mock, patch
import numpy as np

from src.retrieval.retriever import HybridRetriever, SparseRetriever, KeywordExtractor, RetrievalResult
from src.vector_store.milvus_store import SearchHit
from src.embedding.embedder import EmbeddingManager


class TestKeywordExtractor:
    """关键词提取器测试"""
    
    def setup_method(self):
        self.extractor = KeywordExtractor()
    
    def test_extract_keywords(self):
        """测试关键词提取"""
        text = "人工智能和机器学习是现代科技发展的重要方向。深度学习技术在图像识别和自然语言处理领域取得了重大突破。"
        keywords = self.extractor.extract_keywords(text, top_k=5)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        assert all(len(kw) > 1 for kw in keywords)  # 关键词长度大于1
        
        # 应该包含一些重要词汇
        important_words = ["人工智能", "机器学习", "深度学习", "图像识别", "自然语言处理"]
        extracted_important = [kw for kw in keywords if kw in important_words]
        assert len(extracted_important) > 0
    
    def test_extract_keywords_empty_text(self):
        """测试空文本关键词提取"""
        keywords = self.extractor.extract_keywords("", top_k=5)
        assert keywords == []
    
    def test_extract_keywords_stop_words_filtered(self):
        """测试停用词过滤"""
        text = "这是一个的了在有就不人都一上也很到说要去你会着没有"
        keywords = self.extractor.extract_keywords(text, top_k=10)
        
        # 停用词应该被过滤掉
        assert len(keywords) == 0 or all(kw not in self.extractor.stop_words for kw in keywords)
    
    def test_extract_entities(self):
        """测试实体提取"""
        text = "这个算法的准确率是95.5%，处理了1000个样本，使用了Python和TensorFlow。"
        entities = self.extractor.extract_entities(text)
        
        assert isinstance(entities, list)
        # 应该提取到数字和英文单词
        assert "95.5" in entities or "1000" in entities
        assert "Python" in entities or "TensorFlow" in entities


class TestSparseRetriever:
    """稀疏检索器测试"""
    
    def setup_method(self):
        self.retriever = SparseRetriever()
        
        # 创建测试文档
        self.test_hits = [
            SearchHit(
                id="doc1",
                score=0.9,
                content="人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的机器。",
                metadata={"filename": "ai_intro.txt"},
                doc_id="doc1",
                chunk_index=0
            ),
            SearchHit(
                id="doc2", 
                score=0.8,
                content="机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习。",
                metadata={"filename": "ml_basics.txt"},
                doc_id="doc2",
                chunk_index=0
            ),
            SearchHit(
                id="doc3",
                score=0.7,
                content="深度学习是机器学习的一个分支，它模仿人脑神经网络的工作方式。",
                metadata={"filename": "dl_overview.txt"},
                doc_id="doc3",
                chunk_index=0
            )
        ]
    
    def test_build_index(self):
        """测试构建TF-IDF索引"""
        self.retriever.build_index(self.test_hits)
        
        assert self.retriever.vectorizer is not None
        assert self.retriever.document_vectors is not None
        assert len(self.retriever.document_metadata) == len(self.test_hits)
    
    def test_search_with_index(self):
        """测试基于TF-IDF的检索"""
        self.retriever.build_index(self.test_hits)
        
        query = "什么是人工智能"
        results = self.retriever.search(query, top_k=2)
        
        assert isinstance(results, list)
        assert len(results) <= 2
        assert all(isinstance(hit, SearchHit) for hit in results)
        
        # 分数应该是递减的
        if len(results) > 1:
            assert results[0].score >= results[1].score
    
    def test_search_without_index(self):
        """测试未构建索引时的检索"""
        query = "测试查询"
        results = self.retriever.search(query)
        
        assert results == []
    
    def test_keyword_search(self):
        """测试关键词检索"""
        query = "人工智能机器学习"
        results = self.retriever.keyword_search(query, self.test_hits, top_k=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        assert all(isinstance(hit, SearchHit) for hit in results)
        
        # 包含更多匹配关键词的文档应该得分更高
        if len(results) > 1:
            assert results[0].score >= results[1].score
    
    def test_keyword_search_no_matches(self):
        """测试无匹配关键词的检索"""
        query = "完全不相关的内容xyz"
        results = self.retriever.keyword_search(query, self.test_hits)
        
        # 应该返回空结果或很少的结果
        assert len(results) == 0 or all(hit.score < 0.1 for hit in results)


class TestHybridRetriever:
    """混合检索器测试"""
    
    def setup_method(self):
        # Mock依赖组件
        self.mock_vector_store = Mock()
        self.mock_embedding_manager = Mock()
        self.mock_embedding_manager.dimension = 1024
        
        self.retriever = HybridRetriever(
            vector_store=self.mock_vector_store,
            embedding_manager=self.mock_embedding_manager,
            dense_weight=0.7,
            sparse_weight=0.3
        )
        
        # 设置测试数据
        self.test_dense_hits = [
            SearchHit("dense1", 0.95, "AI相关内容1", {}, "doc1", 0),
            SearchHit("dense2", 0.85, "AI相关内容2", {}, "doc2", 0),
            SearchHit("dense3", 0.75, "AI相关内容3", {}, "doc3", 0)
        ]
        
        self.test_sparse_hits = [
            SearchHit("sparse1", 0.9, "ML相关内容1", {}, "doc4", 0),
            SearchHit("dense1", 0.8, "重复文档", {}, "doc1", 0),  # 与稠密检索重复
            SearchHit("sparse2", 0.7, "ML相关内容2", {}, "doc5", 0)
        ]
    
    def test_dense_search(self):
        """测试稠密向量检索"""
        # Mock嵌入和向量搜索
        self.mock_embedding_manager.embed_query.return_value = [0.1] * 1024
        self.mock_vector_store.search.return_value = self.test_dense_hits
        
        query = "人工智能的发展"
        results = self.retriever.dense_search(query, top_k=3)
        
        assert len(results) <= 3
        assert all(isinstance(hit, SearchHit) for hit in results)
        
        # 验证调用了正确的方法
        self.mock_embedding_manager.embed_query.assert_called_once_with(query)
        self.mock_vector_store.search.assert_called_once()
    
    def test_sparse_search_without_index(self):
        """测试稀疏检索（未构建索引）"""
        query = "机器学习算法"
        results = self.retriever.sparse_search(query, top_k=3)
        
        # 未构建索引时应该返回空结果
        assert results == []
    
    def test_hybrid_search(self):
        """测试混合检索"""
        # Mock稠密检索
        self.mock_embedding_manager.embed_query.return_value = [0.1] * 1024
        self.mock_vector_store.search.return_value = self.test_dense_hits
        
        # Mock稀疏检索（构建简单索引）
        with patch.object(self.retriever.sparse_retriever, 'search') as mock_sparse_search:
            mock_sparse_search.return_value = self.test_sparse_hits
            self.retriever.is_sparse_index_built = True
            
            query = "人工智能和机器学习"
            result = self.retriever.hybrid_search(query, top_k=5)
            
            assert isinstance(result, RetrievalResult)
            assert result.query == query
            assert result.method == "hybrid"
            assert len(result.hits) <= 5
            assert len(result.dense_hits) > 0
            assert len(result.sparse_hits) > 0
    
    def test_fuse_results(self):
        """测试结果融合"""
        fused_results = self.retriever._fuse_results(
            self.test_dense_hits,
            self.test_sparse_hits,
            top_k=5
        )
        
        assert isinstance(fused_results, list)
        assert len(fused_results) <= 5
        assert all(isinstance(hit, SearchHit) for hit in fused_results)
        
        # 融合结果应该按分数排序
        if len(fused_results) > 1:
            scores = [hit.score for hit in fused_results]
            assert scores == sorted(scores, reverse=True)
        
        # 重复文档应该被去重（通过ID识别）
        ids = [hit.id for hit in fused_results]
        assert len(ids) == len(set(ids))  # 无重复ID
    
    def test_search_different_methods(self):
        """测试不同检索方法"""
        self.mock_embedding_manager.embed_query.return_value = [0.1] * 1024
        self.mock_vector_store.search.return_value = self.test_dense_hits
        
        query = "测试查询"
        
        # 测试稠密检索
        dense_result = self.retriever.search(query, method="dense")
        assert dense_result.method == "dense"
        assert len(dense_result.sparse_hits) == 0
        
        # 测试稀疏检索
        sparse_result = self.retriever.search(query, method="sparse")
        assert sparse_result.method == "sparse"
        assert len(sparse_result.dense_hits) == 0
        
        # 测试混合检索
        with patch.object(self.retriever, 'hybrid_search') as mock_hybrid:
            mock_hybrid.return_value = RetrievalResult(
                query=query, hits=[], dense_hits=[], sparse_hits=[],
                total_hits=0, retrieval_time=0.0, method="hybrid"
            )
            
            hybrid_result = self.retriever.search(query, method="hybrid")
            assert hybrid_result.method == "hybrid"
            mock_hybrid.assert_called_once()
    
    def test_update_weights(self):
        """测试更新检索权重"""
        new_dense_weight = 0.8
        new_sparse_weight = 0.2
        
        self.retriever.update_weights(new_dense_weight, new_sparse_weight)
        
        assert self.retriever.dense_weight == new_dense_weight
        assert self.retriever.sparse_weight == new_sparse_weight
    
    def test_get_stats(self):
        """测试获取统计信息"""
        stats = self.retriever.get_stats()
        
        assert isinstance(stats, dict)
        assert "dense_weight" in stats
        assert "sparse_weight" in stats
        assert "similarity_threshold" in stats
        assert "sparse_index_built" in stats
        assert "embedding_dimension" in stats
        
        assert stats["dense_weight"] == self.retriever.dense_weight
        assert stats["sparse_weight"] == self.retriever.sparse_weight
        assert stats["embedding_dimension"] == 1024
    
    def test_build_sparse_index(self):
        """测试构建稀疏索引"""
        # Mock获取所有文档
        self.mock_vector_store.search.return_value = self.test_dense_hits
        
        with patch.object(self.retriever.sparse_retriever, 'build_index') as mock_build:
            self.retriever.build_sparse_index()
            
            # 应该调用了稀疏检索器的build_index方法
            mock_build.assert_called_once()
            assert self.retriever.is_sparse_index_built is True
    
    def test_build_sparse_index_no_documents(self):
        """测试无文档时构建稀疏索引"""
        # Mock返回空文档列表
        self.mock_vector_store.search.return_value = []
        
        self.retriever.build_sparse_index()
        
        # 无文档时不应该设置为已构建
        assert self.retriever.is_sparse_index_built is False


class TestRetrievalIntegration:
    """检索功能集成测试"""
    
    @pytest.mark.skip(reason="需要真实的Milvus和嵌入服务")
    def test_end_to_end_retrieval(self):
        """端到端检索测试（需要真实服务）"""
        # 这个测试需要真实的Milvus和嵌入服务
        from src.vector_store.milvus_store import MilvusVectorStore
        from src.embedding.embedder import EmbeddingManager
        
        vector_store = MilvusVectorStore()
        embedding_manager = EmbeddingManager()
        retriever = HybridRetriever(vector_store, embedding_manager)
        
        # 初始化
        assert vector_store.initialize()
        
        # 构建索引
        retriever.build_sparse_index()
        
        # 执行查询
        query = "人工智能的应用"
        result = retriever.hybrid_search(query, top_k=5)
        
        assert isinstance(result, RetrievalResult)
        assert result.query == query
        assert result.total_hits >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])