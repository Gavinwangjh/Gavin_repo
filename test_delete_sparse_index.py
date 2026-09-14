#!/usr/bin/env python3
"""
测试删除文档后稀疏索引重建的功能
"""

import logging
import os
import sys

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cli.cli import RAGSystem
from src.document_loader.loader import DocumentChunk

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_delete_and_sparse_index():
    """测试删除文档后稀疏索引的重建"""
    # 初始化系统
    rag_system = RAGSystem()
    if not rag_system.initialize():
        logger.error("系统初始化失败")
        return False
    
    # 创建测试文档
    test_chunks = [
        DocumentChunk(
            chunk_id="test_chunk_1",
            content="这是第一个测试文档，包含关键词：人工智能、机器学习",
            doc_id="test_doc_1",
            chunk_index=0,
            metadata={"filename": "test1.txt", "file_type": "txt"}
        ),
        DocumentChunk(
            chunk_id="test_chunk_2",
            content="这是第二个测试文档，包含关键词：深度学习、神经网络",
            doc_id="test_doc_2", 
            chunk_index=0,
            metadata={"filename": "test2.txt", "file_type": "txt"}
        )
    ]
    
    try:
        # 1. 插入测试文档
        logger.info("插入测试文档...")
        embeddings = []
        for chunk in test_chunks:
            embedding = rag_system.embedding_manager.embed_query(chunk.content)
            embeddings.append(embedding)
        
        result = rag_system.vector_store.insert_documents(test_chunks, embeddings)
        if not result.success:
            logger.error(f"插入失败: {result.message}")
            return False
        
        logger.info(f"插入成功: {len(result.ids)} 个文档块")
        
        # 2. 构建稀疏索引
        logger.info("构建稀疏索引...")
        rag_system.retriever.build_sparse_index(force_rebuild=True)
        
        # 3. 测试混合检索
        logger.info("测试删除前的检索...")
        results_before = rag_system.retriever.search("人工智能", top_k=5, method="hybrid")
        logger.info(f"删除前检索结果数量: {len(results_before)}")
        
        # 打印稀疏索引状态
        stats = rag_system.retriever.get_stats()
        logger.info(f"稀疏索引状态: {stats.get('sparse_index_built', False)}")
        
        # 4. 删除一个文档
        logger.info("删除文档 test_doc_1...")
        delete_success = rag_system.vector_store.delete_by_doc_id("test_doc_1")
        if not delete_success:
            logger.error("删除失败")
            return False
        
        logger.info("删除成功")
        
        # 5. 重建稀疏索引（模拟实际应用中的行为）
        logger.info("重建稀疏索引...")
        rag_system.retriever.build_sparse_index(force_rebuild=True)
        
        # 6. 测试删除后的检索
        logger.info("测试删除后的检索...")
        results_after = rag_system.retriever.search("人工智能", top_k=5, method="hybrid")
        logger.info(f"删除后检索结果数量: {len(results_after)}")
        
        # 7. 验证已删除文档不在结果中
        deleted_doc_found = any(hit.doc_id == "test_doc_1" for hit in results_after)
        if deleted_doc_found:
            logger.error("错误：已删除的文档仍在检索结果中！")
            return False
        
        logger.info("✅ 测试通过：删除的文档不在检索结果中")
        
        # 8. 清理剩余文档
        logger.info("清理剩余测试文档...")
        rag_system.vector_store.delete_by_doc_id("test_doc_2")
        
        return True
        
    except Exception as e:
        logger.error(f"测试过程中出现异常: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_delete_and_sparse_index()
    if success:
        logger.info("🎉 删除文档后稀疏索引重建测试成功！")
        sys.exit(0)
    else:
        logger.error("❌ 测试失败")
        sys.exit(1)