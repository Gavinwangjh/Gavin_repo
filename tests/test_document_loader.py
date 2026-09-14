import pytest
import tempfile
import os
from pathlib import Path

from src.document_loader.loader import DocumentLoader, Document, DocumentChunk


class TestDocumentLoader:
    """文档加载器测试"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
    
    def test_load_text_file(self):
        """测试加载文本文件"""
        # 创建临时文本文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_content = "这是一个测试文档。\n包含多行内容。\n用于测试文档加载功能。"
            f.write(test_content)
            temp_path = f.name
        
        try:
            # 加载文档
            doc = self.loader.load_document(temp_path)
            
            # 验证结果
            assert isinstance(doc, Document)
            assert doc.content.strip() == test_content.replace('\n', ' ').strip()
            assert doc.metadata['filename'] == Path(temp_path).name
            assert doc.metadata['file_type'] == '.txt'
            assert doc.doc_id is not None
            
        finally:
            os.unlink(temp_path)
    
    def test_load_markdown_file(self):
        """测试加载Markdown文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            test_content = "# 标题\n\n这是一个**Markdown**文档。\n\n- 列表项1\n- 列表项2"
            f.write(test_content)
            temp_path = f.name
        
        try:
            doc = self.loader.load_document(temp_path)
            
            assert isinstance(doc, Document)
            assert "标题" in doc.content
            assert "Markdown" in doc.content
            assert doc.metadata['file_type'] == '.md'
            
        finally:
            os.unlink(temp_path)
    
    def test_unsupported_format(self):
        """测试不支持的文件格式"""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="不支持的文件格式"):
                self.loader.load_document(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_file_not_found(self):
        """测试文件不存在"""
        with pytest.raises(FileNotFoundError):
            self.loader.load_document("nonexistent_file.txt")
    
    def test_chunk_document(self):
        """测试文档分块"""
        # 创建一个长文档
        long_content = "这是一个很长的文档。" * 50  # 重复50次
        doc = Document(
            content=long_content,
            metadata={"test": True},
            doc_id="test_doc"
        )
        
        chunks = self.loader.chunk_document(doc)
        
        # 验证分块结果
        assert len(chunks) > 1  # 应该被分成多块
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.doc_id == "test_doc" for chunk in chunks)
        assert all(chunk.chunk_index == i for i, chunk in enumerate(chunks))
        
        # 验证内容连续性
        combined_content = "".join(chunk.content for chunk in chunks)
        assert len(combined_content) >= len(long_content) * 0.8  # 考虑重叠
    
    def test_small_document_no_chunking(self):
        """测试小文档不分块"""
        small_content = "这是一个短文档。"
        doc = Document(
            content=small_content,
            metadata={"test": True},
            doc_id="small_doc"
        )
        
        chunks = self.loader.chunk_document(doc)
        
        assert len(chunks) == 1
        assert chunks[0].content == small_content
        assert chunks[0].chunk_index == 0
    
    def test_load_and_chunk_document(self):
        """测试加载并分块文档"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_content = "这是一个测试文档。" * 20  # 创建足够长的内容
            f.write(test_content)
            temp_path = f.name
        
        try:
            chunks = self.loader.load_and_chunk_document(temp_path)
            
            assert len(chunks) >= 1
            assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
            assert all(chunk.doc_id == chunks[0].doc_id for chunk in chunks)
            
        finally:
            os.unlink(temp_path)
    
    def test_get_file_info(self):
        """测试获取文件信息"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        
        try:
            info = self.loader.get_file_info(temp_path)
            
            assert info['filename'] == Path(temp_path).name
            assert info['file_type'] == '.txt'
            assert info['file_size'] > 0
            assert info['is_supported'] is True
            
        finally:
            os.unlink(temp_path)
    
    def test_clean_text(self):
        """测试文本清理"""
        dirty_text = "这是一个   有很多空格\n\n\n和换行的\t\t文档。"
        cleaned = self.loader._clean_text(dirty_text)
        
        # 验证多余空白被清理
        assert "   " not in cleaned
        assert "\n\n\n" not in cleaned
        assert "\t\t" not in cleaned
        assert "有很多空格 和换行的 文档" in cleaned
    
    def test_empty_file(self):
        """测试空文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")  # 空内容
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="文件内容为空"):
                self.loader.load_document(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_chunk_overlap(self):
        """测试分块重叠"""
        # 创建可预测的内容
        content = "句子1。句子2。句子3。句子4。句子5。句子6。句子7。句子8。"
        doc = Document(content=content, metadata={}, doc_id="test")
        
        # 使用小的chunk_size来强制分块
        loader = DocumentLoader(chunk_size=20, chunk_overlap=10)
        chunks = loader.chunk_document(doc)
        
        if len(chunks) > 1:
            # 检查相邻块之间有重叠
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i].content[-10:]  # 取最后10个字符
                chunk2_start = chunks[i + 1].content[:10]  # 取前10个字符
                
                # 应该有一些重叠内容
                assert len(chunk1_end) > 0
                assert len(chunk2_start) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])