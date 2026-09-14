import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import tempfile
import io

# 导入API应用
from src.api.app import app, app_state


class TestAPI:
    """API接口测试"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.client = TestClient(app)
        
        # Mock所有组件
        self.mock_components = {
            "document_loader": Mock(),
            "embedding_manager": Mock(),
            "vector_store": Mock(),
            "retriever": Mock(),
            "generator": Mock()
        }
        
        # 设置mock返回值
        self.mock_components["vector_store"].initialize.return_value = True
        self.mock_components["vector_store"].health_check.return_value = {
            "connected": True,
            "collection_exists": True,
            "entity_count": 100
        }
        self.mock_components["vector_store"].get_collection_stats.return_value = {
            "collection_name": "test_collection",
            "entity_count": 100,
            "dimension": 1024
        }
        
        # 将mock组件注入应用状态
        app_state.update(self.mock_components)
        app_state["initialized"] = True
    
    def test_root_endpoint(self):
        """测试根路径"""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "LLM RAG API"
    
    def test_health_check_healthy(self):
        """测试健康检查（正常状态）"""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert "timestamp" in data
        assert "milvus" in data["components"]
    
    def test_health_check_unhealthy(self):
        """测试健康检查（异常状态）"""
        app_state["initialized"] = False
        
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
    
    def test_upload_document_success(self):
        """测试成功上传文档"""
        # Mock文档处理流程
        from src.document_loader.loader import DocumentChunk
        from src.embedding.embedder import EmbeddingResult
        from src.vector_store.milvus_store import InsertResult
        
        mock_chunks = [
            DocumentChunk(
                content="测试内容1",
                metadata={"chunk_index": 0},
                chunk_id="chunk1",
                doc_id="doc1",
                chunk_index=0
            ),
            DocumentChunk(
                content="测试内容2",
                metadata={"chunk_index": 1},
                chunk_id="chunk2",
                doc_id="doc1",
                chunk_index=1
            )
        ]
        
        mock_embedding_result = EmbeddingResult(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            texts=["测试内容1", "测试内容2"],
            model="test_model",
            dimension=2
        )
        
        mock_insert_result = InsertResult(
            ids=["chunk1", "chunk2"],
            insert_count=2,
            success=True
        )
        
        self.mock_components["document_loader"].load_and_chunk_document.return_value = mock_chunks
        self.mock_components["embedding_manager"].embed_documents.return_value = mock_embedding_result
        self.mock_components["vector_store"].insert_documents.return_value = mock_insert_result
        
        # 创建测试文件
        test_content = b"This is a test document content."
        test_file = ("test.txt", io.BytesIO(test_content), "text/plain")
        
        response = self.client.post(
            "/documents/upload",
            files={"file": test_file}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["chunk_count"] == 2
        assert "document" in data
        assert data["document"]["filename"] == "test.txt"
    
    def test_upload_document_unsupported_format(self):
        """测试上传不支持的文件格式"""
        test_content = b"This is a test file."
        test_file = ("test.xyz", io.BytesIO(test_content), "application/octet-stream")
        
        response = self.client.post(
            "/documents/upload",
            files={"file": test_file}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "不支持的文件格式" in data["detail"]
    
    def test_upload_document_too_large(self):
        """测试上传过大文件"""
        # 创建超大文件（假设限制为10MB）
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        test_file = ("large.txt", io.BytesIO(large_content), "text/plain")
        
        response = self.client.post(
            "/documents/upload",
            files={"file": test_file}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "文件大小超限" in data["detail"]
    
    def test_query_documents_success(self):
        """测试成功查询文档"""
        from src.retrieval.retriever import RetrievalResult
        from src.generation.generator import GenerationResult
        from src.vector_store.milvus_store import SearchHit
        
        # Mock检索结果
        mock_hits = [
            SearchHit(
                id="hit1",
                score=0.9,
                content="相关内容1",
                metadata={"filename": "test1.txt"},
                doc_id="doc1",
                chunk_index=0
            )
        ]
        
        mock_retrieval_result = RetrievalResult(
            query="测试问题",
            hits=mock_hits,
            dense_hits=mock_hits,
            sparse_hits=[],
            total_hits=1,
            retrieval_time=0.1,
            method="hybrid"
        )
        
        mock_generation_result = GenerationResult(
            question="测试问题",
            answer="这是AI生成的回答",
            sources=mock_hits,
            retrieval_result=mock_retrieval_result,
            generation_time=0.2,
            total_time=0.3,
            model="test_model"
        )
        
        self.mock_components["retriever"].search.return_value = mock_retrieval_result
        self.mock_components["generator"].generate_answer.return_value = mock_generation_result
        
        # 发送查询请求
        query_data = {
            "question": "测试问题",
            "top_k": 5,
            "method": "hybrid"
        }
        
        response = self.client.post("/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "测试问题"
        assert data["answer"] == "这是AI生成的回答"
        assert len(data["sources"]) == 1
        assert data["method"] == "hybrid"
    
    def test_query_documents_invalid_request(self):
        """测试无效查询请求"""
        # 缺少必要字段
        invalid_data = {
            "top_k": 5,
            "method": "hybrid"
            # 缺少question字段
        }
        
        response = self.client.post("/query", json=invalid_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_query_documents_invalid_method(self):
        """测试无效检索方法"""
        query_data = {
            "question": "测试问题",
            "top_k": 5,
            "method": "invalid_method"
        }
        
        response = self.client.post("/query", json=query_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_chat_success(self):
        """测试成功聊天"""
        self.mock_components["generator"].chat.return_value = "聊天回答"
        
        chat_data = {
            "question": "你好",
            "history": [
                {"role": "user", "content": "之前的问题"},
                {"role": "assistant", "content": "之前的回答"}
            ]
        }
        
        response = self.client.post("/chat", json=chat_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "聊天回答"
    
    def test_delete_document_success(self):
        """测试成功删除文档"""
        self.mock_components["vector_store"].delete_by_doc_id.return_value = True
        
        response = self.client.delete("/documents/test_doc_id")
        
        assert response.status_code == 200
        data = response.json()
        assert "删除成功" in data["message"]
    
    def test_delete_document_not_found(self):
        """测试删除不存在的文档"""
        self.mock_components["vector_store"].delete_by_doc_id.return_value = False
        
        response = self.client.delete("/documents/nonexistent_doc")
        
        assert response.status_code == 404
        data = response.json()
        assert "不存在或删除失败" in data["detail"]
    
    def test_get_stats_success(self):
        """测试获取统计信息"""
        self.mock_components["retriever"].get_stats.return_value = {
            "dense_weight": 0.7,
            "sparse_weight": 0.3
        }
        self.mock_components["generator"].get_stats.return_value = {
            "model": "test_model",
            "max_tokens": 1000
        }
        
        response = self.client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "milvus_stats" in data
        assert "retriever_stats" in data
        assert "generator_stats" in data
    
    def test_rebuild_index_success(self):
        """测试重建索引"""
        response = self.client.post("/index/rebuild")
        
        assert response.status_code == 200
        data = response.json()
        assert "索引重建任务已启动" in data["message"]
    
    def test_query_stream_endpoint(self):
        """测试流式查询端点"""
        from src.retrieval.retriever import RetrievalResult
        from src.generation.generator import GenerationResult
        
        # Mock检索和生成结果
        mock_retrieval_result = RetrievalResult(
            query="测试问题",
            hits=[],
            dense_hits=[],
            sparse_hits=[],
            total_hits=0,
            retrieval_time=0.1,
            method="hybrid"
        )
        
        mock_generation_result = GenerationResult(
            question="测试问题",
            answer="流式 回答 测试",
            sources=[],
            generation_time=0.2,
            total_time=0.3,
            model="test_model"
        )
        
        self.mock_components["retriever"].search.return_value = mock_retrieval_result
        self.mock_components["generator"].generate_answer.return_value = mock_generation_result
        
        query_data = {
            "question": "测试问题",
            "top_k": 5,
            "method": "hybrid",
            "stream": True
        }
        
        response = self.client.post("/query/stream", json=query_data)
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        # 检查流式响应内容
        content = response.text
        assert "data:" in content
        assert "[DONE]" in content


class TestAPIErrorHandling:
    """API错误处理测试"""
    
    def setup_method(self):
        self.client = TestClient(app)
        
        # 设置未初始化状态
        app_state.clear()
        app_state["initialized"] = False
    
    def test_upload_without_initialization(self):
        """测试未初始化时上传文档"""
        test_content = b"Test content"
        test_file = ("test.txt", io.BytesIO(test_content), "text/plain")
        
        response = self.client.post(
            "/documents/upload",
            files={"file": test_file}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "系统未就绪" in data["detail"]
    
    def test_query_without_initialization(self):
        """测试未初始化时查询"""
        query_data = {
            "question": "测试问题",
            "top_k": 5,
            "method": "hybrid"
        }
        
        response = self.client.post("/query", json=query_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "系统未就绪" in data["detail"]
    
    def test_chat_without_initialization(self):
        """测试未初始化时聊天"""
        chat_data = {
            "question": "你好"
        }
        
        response = self.client.post("/chat", json=chat_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "系统未就绪" in data["detail"]


class TestAPIValidation:
    """API输入验证测试"""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_query_validation_empty_question(self):
        """测试空问题验证"""
        query_data = {
            "question": "",  # 空问题
            "top_k": 5,
            "method": "hybrid"
        }
        
        response = self.client.post("/query", json=query_data)
        assert response.status_code == 422
    
    def test_query_validation_invalid_top_k(self):
        """测试无效top_k验证"""
        query_data = {
            "question": "测试问题",
            "top_k": 0,  # 无效值
            "method": "hybrid"
        }
        
        response = self.client.post("/query", json=query_data)
        assert response.status_code == 422
    
    def test_query_validation_long_question(self):
        """测试过长问题验证"""
        query_data = {
            "question": "x" * 1001,  # 超过最大长度
            "top_k": 5,
            "method": "hybrid"
        }
        
        response = self.client.post("/query", json=query_data)
        assert response.status_code == 422
    
    def test_chat_validation_empty_question(self):
        """测试聊天空问题验证"""
        chat_data = {
            "question": ""  # 空问题
        }
        
        response = self.client.post("/chat", json=chat_data)
        assert response.status_code == 422
    
    def test_chat_validation_invalid_history(self):
        """测试聊天无效历史验证"""
        chat_data = {
            "question": "测试问题",
            "history": "invalid_history"  # 应该是列表
        }
        
        response = self.client.post("/chat", json=chat_data)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])