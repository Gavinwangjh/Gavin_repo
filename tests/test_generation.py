import pytest
from unittest.mock import Mock, patch
import time

from src.generation.generator import (
    RAGGenerator, KimiLLMClient, PromptTemplate, 
    GenerationResult, ChatMessage
)
from src.retrieval.retriever import RetrievalResult
from src.vector_store.milvus_store import SearchHit


class TestChatMessage:
    """聊天消息测试"""
    
    def test_chat_message_creation(self):
        """测试聊天消息创建"""
        msg = ChatMessage(role="user", content="Hello")
        
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestPromptTemplate:
    """提示词模板测试"""
    
    def test_build_rag_prompt_with_context(self):
        """测试构建带上下文的RAG提示词"""
        question = "什么是人工智能？"
        context_chunks = [
            SearchHit(
                id="doc1",
                score=0.9,
                content="人工智能是计算机科学的一个分支。",
                metadata={"filename": "ai_intro.txt"},
                doc_id="doc1",
                chunk_index=0
            ),
            SearchHit(
                id="doc2",
                score=0.8,
                content="AI技术在多个领域都有应用。",
                metadata={"filename": "ai_applications.txt"},
                doc_id="doc2",
                chunk_index=0
            )
        ]
        
        messages = PromptTemplate.build_rag_prompt(question, context_chunks)
        
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        
        # 用户消息应该包含问题和上下文
        user_content = messages[1].content
        assert question in user_content
        assert "人工智能是计算机科学的一个分支" in user_content
        assert "AI技术在多个领域都有应用" in user_content
        assert "ai_intro.txt" in user_content
        assert "ai_applications.txt" in user_content
    
    def test_build_rag_prompt_without_context(self):
        """测试构建无上下文的RAG提示词"""
        question = "什么是人工智能？"
        context_chunks = []
        
        messages = PromptTemplate.build_rag_prompt(question, context_chunks)
        
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        
        # 应该使用无上下文模板
        user_content = messages[1].content
        assert question in user_content
        assert "没有找到与您的问题相关的文档内容" in user_content
    
    def test_build_chat_prompt(self):
        """测试构建聊天提示词"""
        question = "你好"
        chat_history = [
            ChatMessage(role="user", content="之前的问题"),
            ChatMessage(role="assistant", content="之前的回答")
        ]
        
        messages = PromptTemplate.build_chat_prompt(question, chat_history)
        
        assert len(messages) == 4  # system + history + current question
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[1].content == "之前的问题"
        assert messages[2].role == "assistant"
        assert messages[2].content == "之前的回答"
        assert messages[3].role == "user"
        assert messages[3].content == question
    
    def test_build_chat_prompt_no_history(self):
        """测试构建无历史的聊天提示词"""
        question = "你好"
        
        messages = PromptTemplate.build_chat_prompt(question, None)
        
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[1].content == question


class TestKimiLLMClient:
    """Kimi LLM客户端测试"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        with patch('src.generation.generator.get_config') as mock_config:
            mock_config.side_effect = lambda key, default=None: {
                "api_keys.kimi_api_key": "test_api_key",
                "llm.model": "test_model",
                "llm.max_tokens": 1000,
                "llm.temperature": 0.7,
                "llm.max_retries": 3
            }.get(key, default)
            
            self.client = KimiLLMClient(api_key="test_api_key")
    
    @patch('httpx.Client')
    def test_generate_success(self, mock_client):
        """测试成功生成文本"""
        # Mock HTTP响应
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "这是AI生成的回答。"
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_client_instance
        
        messages = [ChatMessage(role="user", content="测试问题")]
        result = self.client.generate(messages)
        
        assert result == "这是AI生成的回答。"
        
        # 验证API调用
        mock_client_instance.post.assert_called_once()
        call_args = mock_client_instance.post.call_args
        assert "chat/completions" in call_args[0][0]
        assert call_args[1]["json"]["model"] == "kimi-k3"
    
    @patch('httpx.Client')
    def test_generate_http_error(self, mock_client):
        """测试HTTP错误处理"""
        import httpx
        
        mock_client_instance = Mock()
        mock_client_instance.post.side_effect = httpx.HTTPStatusError(
            "API Error", request=Mock(), response=Mock(status_code=401, text="Unauthorized")
        )
        mock_client.return_value.__enter__.return_value = mock_client_instance
        
        messages = [ChatMessage(role="user", content="测试问题")]
        
        with pytest.raises(Exception, match="LLM API调用失败"):
            self.client.generate(messages)
    
    @patch('httpx.Client')
    def test_generate_invalid_response(self, mock_client):
        """测试无效响应处理"""
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Invalid request"}
        mock_response.raise_for_status.return_value = None
        
        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_client_instance
        
        messages = [ChatMessage(role="user", content="测试问题")]
        
        with pytest.raises(ValueError, match="API返回格式异常"):
            self.client.generate(messages)
    
    def test_invalid_api_key(self):
        """测试无效API密钥"""
        with patch.dict("os.environ", {}, clear=True), patch('src.generation.generator.get_config') as mock_config:
            mock_config.return_value = None
            
            with pytest.raises(ValueError, match="Kimi API密钥未设置"):
                KimiLLMClient(api_key=None)


class TestRAGGenerator:
    """RAG生成器测试"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        # Mock LLM客户端
        self.mock_llm_client = Mock()
        self.mock_llm_client.model = "test_model"
        self.mock_llm_client.generate.return_value = "这是生成的回答。"
        
        self.generator = RAGGenerator(
            llm_client=self.mock_llm_client,
            max_context_length=1000,
            min_similarity_score=0.3
        )
        
        # 创建测试检索结果
        self.test_retrieval_result = RetrievalResult(
            query="什么是人工智能？",
            hits=[
                SearchHit(
                    id="hit1",
                    score=0.9,
                    content="人工智能是计算机科学的一个分支。",
                    metadata={"filename": "ai.txt"},
                    doc_id="doc1",
                    chunk_index=0
                ),
                SearchHit(
                    id="hit2",
                    score=0.8,
                    content="AI技术在多个领域有应用。",
                    metadata={"filename": "ai_app.txt"},
                    doc_id="doc2",
                    chunk_index=0
                )
            ],
            dense_hits=[],
            sparse_hits=[],
            total_hits=2,
            retrieval_time=0.1,
            method="hybrid"
        )
    
    def test_generate_answer_success(self):
        """测试成功生成回答"""
        question = "什么是人工智能？"
        
        result = self.generator.generate_answer(
            question=question,
            retrieval_result=self.test_retrieval_result
        )
        
        assert isinstance(result, GenerationResult)
        assert result.question == question
        assert result.answer == "这是生成的回答。"
        assert len(result.sources) == 2  # 两个相关文档
        assert result.retrieval_result == self.test_retrieval_result
        assert result.model == "test_model"
        assert result.generation_time > 0
        assert result.total_time > 0
        
        # 验证调用了LLM
        self.mock_llm_client.generate.assert_called_once()
    
    def test_generate_answer_low_similarity_filtering(self):
        """测试低相似度文档过滤"""
        # 创建包含低相似度文档的检索结果
        low_similarity_result = RetrievalResult(
            query="测试问题",
            hits=[
                SearchHit("hit1", 0.9, "高相似度文档", {}, "doc1", 0),
                SearchHit("hit2", 0.2, "低相似度文档", {}, "doc2", 0),  # 低于阈值
                SearchHit("hit3", 0.5, "中等相似度文档", {}, "doc3", 0)
            ],
            dense_hits=[],
            sparse_hits=[],
            total_hits=3,
            retrieval_time=0.1,
            method="hybrid"
        )
        
        result = self.generator.generate_answer(
            question="测试问题",
            retrieval_result=low_similarity_result
        )
        
        # 应该过滤掉低相似度文档
        assert len(result.sources) == 2  # 只保留相似度>=0.3的文档
        assert all(source.score >= 0.3 for source in result.sources)
    
    def test_generate_answer_no_relevant_documents(self):
        """测试无相关文档的情况"""
        empty_result = RetrievalResult(
            query="测试问题",
            hits=[],
            dense_hits=[],
            sparse_hits=[],
            total_hits=0,
            retrieval_time=0.1,
            method="hybrid"
        )
        
        result = self.generator.generate_answer(
            question="测试问题",
            retrieval_result=empty_result
        )
        
        assert len(result.sources) == 0
        # 仍然应该生成回答（使用无上下文模板）
        assert result.answer == "这是生成的回答。"
    
    def test_generate_answer_llm_error(self):
        """测试LLM错误处理"""
        self.mock_llm_client.generate.side_effect = Exception("LLM服务错误")
        
        result = self.generator.generate_answer(
            question="测试问题",
            retrieval_result=self.test_retrieval_result
        )
        
        # 应该返回错误消息
        assert "生成回答时出现错误" in result.answer
        assert "LLM服务错误" in result.answer
        assert result.generation_time == 0.0
    
    def test_filter_chunks_by_length(self):
        """测试按长度过滤文档块"""
        # 创建包含长文档的hits
        long_hits = [
            SearchHit("hit1", 0.9, "短文档", {}, "doc1", 0),
            SearchHit("hit2", 0.8, "这是一个很长的文档。" * 100, {}, "doc2", 0),
            SearchHit("hit3", 0.7, "另一个短文档", {}, "doc3", 0)
        ]
        
        # 设置较小的上下文长度限制
        generator = RAGGenerator(
            llm_client=self.mock_llm_client,
            max_context_length=50
        )
        
        filtered = generator._filter_chunks_by_length(long_hits)
        
        # 应该根据长度限制过滤文档
        total_length = sum(len(chunk.content) for chunk in filtered)
        assert total_length <= 50
        assert len(filtered) <= len(long_hits)
    
    def test_chat_function(self):
        """测试聊天功能"""
        question = "你好"
        chat_history = [
            ChatMessage(role="user", content="之前的问题"),
            ChatMessage(role="assistant", content="之前的回答")
        ]
        
        result = self.generator.chat(question, chat_history)
        
        assert result == "这是生成的回答。"
        self.mock_llm_client.generate.assert_called_once()
    
    def test_chat_error_handling(self):
        """测试聊天错误处理"""
        self.mock_llm_client.generate.side_effect = Exception("聊天服务错误")
        
        result = self.generator.chat("测试问题")
        
        assert "聊天时出现错误" in result
        assert "聊天服务错误" in result
    
    def test_update_config(self):
        """测试更新配置"""
        new_max_length = 2000
        new_min_score = 0.5
        
        self.generator.update_config(
            max_context_length=new_max_length,
            min_similarity_score=new_min_score
        )
        
        assert self.generator.max_context_length == new_max_length
        assert self.generator.min_similarity_score == new_min_score
    
    def test_get_stats(self):
        """测试获取统计信息"""
        stats = self.generator.get_stats()
        
        assert isinstance(stats, dict)
        assert "model" in stats
        assert "max_context_length" in stats
        assert "min_similarity_score" in stats
        
        assert stats["model"] == "test_model"
        assert stats["max_context_length"] == 1000
        assert stats["min_similarity_score"] == 0.3


class TestGenerationIntegration:
    """生成功能集成测试"""
    
    @pytest.mark.skip(reason="需要真实API密钥")
    def test_end_to_end_generation(self):
        """端到端生成测试（需要真实API）"""
        # 这个测试需要真实的Kimi API密钥
        generator = RAGGenerator()
        
        # 创建测试检索结果
        retrieval_result = RetrievalResult(
            query="什么是RAG技术？",
            hits=[
                SearchHit(
                    id="test1",
                    score=0.9,
                    content="RAG（检索增强生成）是一种结合信息检索和文本生成的AI技术。",
                    metadata={"filename": "rag_intro.txt"},
                    doc_id="doc1",
                    chunk_index=0
                )
            ],
            dense_hits=[],
            sparse_hits=[],
            total_hits=1,
            retrieval_time=0.1,
            method="hybrid"
        )
        
        result = generator.generate_answer(
            question="什么是RAG技术？",
            retrieval_result=retrieval_result
        )
        
        assert isinstance(result, GenerationResult)
        assert len(result.answer) > 0
        assert "RAG" in result.answer or "检索" in result.answer
        assert len(result.sources) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
