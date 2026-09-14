#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试特定文件的条文分块
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from document_loader.loader import DocumentLoader
from document_loader.article_chunker import ArticleChunker


def test_specific_file():
    """测试特定文件"""
    file_path = "data/nsfc_test/016_关于发布2025年度理论物理专款项目指南的通告_2025-07-30.md"

    if not Path(file_path).exists():
        print(f"文件不存在: {file_path}")
        return

    # 测试条文检测
    chunker = ArticleChunker()
    loader = DocumentLoader(use_article_chunking=True)

    print(f"测试文件: {file_path}")

    # 加载文档
    doc = loader.load_document(file_path)
    print(f"文档内容长度: {len(doc.content)}")
    print(f"文档内容预览: {doc.content[:500]}...")

    # 测试条文检测
    is_article = chunker.is_article_document(doc.content)
    print(f"是否为条文文档: {is_article}")

    if is_article:
        # 使用条文分块
        chunks = loader.chunk_document(doc)
        print(f"生成了 {len(chunks)} 个块")

        # 显示条文块的详细信息
        article_chunks = [c for c in chunks if c.metadata.get('chunk_type') == 'article_based']
        print(f"其中 {len(article_chunks)} 个为条文块")

        for i, chunk in enumerate(article_chunks[:10]):  # 只显示前10个
            meta = chunk.metadata
            print(f"\n--- 块 {i+1} ---")
            print(f"条文编号: {meta.get('article_number', '无')}")
            print(f"条文标题: {meta.get('article_title', '无')}")
            print(f"条文层级: {meta.get('article_level', '无')}")
            print(f"父条文: {meta.get('parent_article', '无')}")
            print(f"内容长度: {len(chunk.content)}")
            print(f"内容预览: {chunk.content[:150]}...")
    else:
        print("未被识别为条文文档")

        # 看看原始检测结果
        article_count = 0
        for pattern, level, section_type in chunker.article_patterns:
            matches = re.findall(pattern, doc.content, re.MULTILINE)
            if matches:
                print(f"模式 '{section_type}' 匹配到 {len(matches)} 个: {matches[:5]}")
                article_count += len(matches)

        lines = doc.content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            ratio = article_count / len(non_empty_lines)
            print(f"总条文标记: {article_count}, 非空行: {len(non_empty_lines)}, 比例: {ratio:.4f}")


if __name__ == "__main__":
    import re
    test_specific_file()