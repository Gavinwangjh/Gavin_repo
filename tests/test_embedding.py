import pytest
from unittest.mock import Mock, patch
import numpy as np

from src.embedding.embedder import EmbeddingManager, SiliconFlowEmbedder, EmbeddingResult, TextProcessor


class TestTextProcessor:
    """文本处理器测试"""
    
    def test_clean_text(self):
        """测试文本清理"""
        dirty_text = "这是一个   有很多空格\n\n和换行的\t文档。\x00\x1f"
        cleaned = TextProcessor.clean_text(dirty_text)
        
        assert "   " not in cleaned
        assert "\n\n" not in cleaned
        assert "\t" not in cleaned
        assert "\x00" not in cleaned
        assert "\x1f" not in cleaned
        assert "这是一个 有很多空格 和换行的 文档。" == cleaned
    
    def test_split_long_text(self):
        """测试长文本分割"""
        long_text = "这是一个很长的文本。" * 1000  # 创建很长的文本
        chunks = TextProcessor.split_long_text(long_text, max_length=100)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks[:-1])  # 除了最后一个，都不超过限制
        
        # 重新组合应该包含原始内容的大部分
        combined = "".join(chunks)
        assert len(combined) >= len(long_text) * 0.9
    
    def test_split_short_text(self):
        """测试短文本不分割"""
        short_text = "这是一个短文本。"
        chunks = TextProcessor.split_long_text(short_text, max_length=100)
        
        assert len(chunks) == 1
        assert chunks[0] == short_text


class TestSiliconFlowEmbedder:
    """SiliconFlow嵌入器测试"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        # 使用mock API密钥
        with patch('src.embedding.embedder.get_config') as mock_config:
            mock_config.side_effect = lambda key, default=None: {
                "api_keys.siliconflow_api_key": "test_api_key",
                "embedding.dimension": 1024
            }.get(key, default)
            
            self.embedder = SiliconFlowEmbedder(
                api_key="test_api_key",
                model="test_model",
                batch_size=2
            )
    
    @patch('httpx.Client')
    def test_embed_texts_success(self, mock_client):
        """测试成功嵌入文本"""
        # Mock HTTP响应
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ],
            "usage": {"total_tokens": 10}
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_client_instance
        
        # 测试嵌入
        texts = ["文本1", "文本2"]
        result = self.embedder.embed_texts(texts)
        
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 2
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
        assert result.embeddings[1] == [0.4, 0.5, 0.6]
        assert result.texts == texts
        assert result.token_count == 10
    
    @patch('httpx.Client')
    def test_embed_texts_batch_processing(self, mock_client):
        """测试批处理"""
        # Mock HTTP响应
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]}
            ],
            "usage": {"total_tokens": 5}
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_client_instance
        
        # 测试大于batch_size的文本列表
        texts = ["文本1", "文本2", "文本3", "文本4"]  # batch_size=2，应该分成2批
        result = self.embedder.embed_texts(texts)
        
        # 应该调用2次API
        assert mock_client_instance.post.call_count == 2
        assert len(result.embeddings) == 4
        assert result.token_count == 10  # 2批，每批5个token
    
    @patch('httpx.Client')
    def test_embed_texts_http_error(self, mock_client):
        """测试HTTP错误处理"""
        import httpx
        
        mock_client_instance = Mock()
        mock_client_instance.post.side_effect = httpx.HTTPStatusError(
            "API Error", request=Mock(), response=Mock(status_code=400, text="Bad Request")
        )
        mock_client.return_value.__enter__.return_value = mock_client_instance
        
        with pytest.raises(Exception, match="嵌入API调用失败"):
            self.embedder.embed_texts(["测试文本"])
    
    def test_embed_empty_list(self):
        """测试空文本列表"""
        result = self.embedder.embed_texts([])
        
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 0
        assert len(result.texts) == 0
    
    @patch('httpx.Client')
    def test_embed_single_text(self, mock_client):
        """测试单文本嵌入"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "usage": {"total_tokens": 5}
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_client_instance
        
        embedding = self.embedder.embed_single_text("测试文本")
        
        assert embedding == [0.1, 0.2, 0.3]


class TestEmbeddingManager:
    """嵌入管理器测试"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        with patch('src.embedding.embedder.SiliconFlowEmbedder') as mock_embedder_class:
            mock_embedder = Mock()
            mock_embedder.dimension = 1024
            mock_embedder.embed_texts.return_value = EmbeddingResult(
                embeddings=[[0.1, 0.2], [0.3, 0.4]],
                texts=["文本1", "文本2"],
                model="test_model",
                dimension=2
            )
            mock_embedder.embed_single_text.return_value = [0.1, 0.2]
            mock_embedder_class.return_value = mock_embedder
            
            self.manager = EmbeddingManager(provider="siliconflow")
            self.mock_embedder = mock_embedder
    
    def test_embed_documents(self):
        """测试文档嵌入"""
        documents = ["这是文档1。", "这是文档2。"]
        result = self.manager.embed_documents(documents)
        
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 2
        
        # 验证调用了底层嵌入器
        self.mock_embedder.embed_texts.assert_called_once()
    
    def test_embed_query(self):
        """测试查询嵌入"""
        query = "这是一个查询"
        embedding = self.manager.embed_query(query)
        
        assert embedding == [0.1, 0.2]
        self.mock_embedder.embed_single_text.assert_called_once()
    
    def test_embed_documents_with_long_text(self):
        """测试包含长文本的文档嵌入"""
        long_doc = "这是一个很长的文档。" * 1000
        documents = [long_doc, "短文档"]
        
        # 设置mock返回更多嵌入（因为长文档会被分割）
        self.mock_embedder.embed_texts.return_value = EmbeddingResult(
            embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],  # 3个嵌入
            texts=["chunk1", "chunk2", "短文档"],
            model="test_model",
            dimension=2
        )
        
        result = self.manager.embed_documents(documents)
        
        # 长文档被分割，所以应该有更多嵌入
        assert len(result.embeddings) >= 2
    
    def test_compute_similarity(self):
        """测试相似度计算"""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        emb3 = [1.0, 0.0, 0.0]
        
        # 垂直向量相似度应该为0
        sim1 = self.manager.compute_similarity(emb1, emb2)
        assert abs(sim1 - 0.0) < 1e-6
        
        # 相同向量相似度应该为1
        sim2 = self.manager.compute_similarity(emb1, emb3)
        assert abs(sim2 - 1.0) < 1e-6
    
    def test_compute_similarity_zero_vectors(self):
        """测试零向量相似度计算"""
        emb1 = [0.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        
        sim = self.manager.compute_similarity(emb1, emb2)
        assert sim == 0.0
    
    def test_dimension_property(self):
        """测试维度属性"""
        assert self.manager.dimension == 1024
    
    def test_unsupported_provider(self):
        """测试不支持的提供商"""
        with pytest.raises(ValueError, match="不支持的嵌入提供商"):
            EmbeddingManager(provider="unsupported_provider")


class TestEmbeddingIntegration:
    """嵌入功能集成测试"""
    
    @pytest.mark.skip(reason="需要真实API密钥")
    def test_real_api_integration(self):
        """集成测试（需要真实API密钥）"""
        # 这个测试需要真实的API密钥，通常在CI环境中跳过
        manager = EmbeddingManager(provider="siliconflow")
        
        test_texts = ["这是一个测试文本", "另一个测试文本"]
        result = manager.embed_documents(test_texts)
        
        assert len(result.embeddings) == len(test_texts)
        assert all(len(emb) == manager.dimension for emb in result.embeddings)
        
        # 测试查询嵌入
        query_emb = manager.embed_query("测试查询")
        assert len(query_emb) == manager.dimension
        
        # 测试相似度计算
        sim = manager.compute_similarity(result.embeddings[0], query_emb)
        assert 0 <= sim <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])