#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试条文分块功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from document_loader.loader import DocumentLoader
from document_loader.article_chunker import ArticleChunker


def test_article_chunker_basic():
    """测试基础条文分块功能"""
    print("=" * 50)
    print("测试基础条文分块功能")
    print("=" * 50)

    test_text = """
国家自然科学基金委员会关于发布2025年度理论物理专款项目指南的通告

一、申请项目类型
经理论物理专款学术领导小组研究确定，2025年度理论物理专款主要资助以下5个项目类型。

（一）"理论物理创新研究中心项目"
该类项目以搭建交流平台、促进合作研究为主旨，以凸显前沿性、交叉性和创新性为目标，支持围绕前沿方向和关键问题开展理论物理研究。通过多种形式的学术交流研讨活动，凝聚研究队伍，聚焦科学问题，培养青年学术骨干，动员全国优秀的理论物理研究力量集中攻关，产出重大创新性成果，推动理论物理学科发展。

（二）"理论物理前沿引领项目"
该类项目旨在探索资助具有理论物理特色的前沿引领性研究。资助具有先导性、原创性的理论物理研究方向，鼓励跨团队开展深层次、实质性的交叉合作，共同促进理论物理的创新发展。

优先支持以下10个研究方向：
1. 超越宇宙学标准模型新范式（申请代码2选择A2504）
2. 利用宇宙第一缕光和中性氢揭示暗物质属性（申请代码2选择A2504）
3. 量子多体系统新理论与普适性质（申请代码2选择A2502）

二、申报要求及注意事项

（一）限项规定
1. 资助期限不超过1年的理论物理专款项目不计入限项范围。
2. 资助期限超过1年的理论物理专款项目计入高级专业技术职务（职称）人员申请和承担总数2项的范围。
3. 申请人同一年度只能申请1项理论物理专款研究项目。

（二）申请注意事项
1. 本专项项目申请书采用在线方式撰写。对申请人具体要求如下：
（1）申请人在填报申请书前，应当认真阅读《国家自然科学基金专项项目管理办法》、《2025年度国家自然科学基金项目指南》和本指南的相关内容。
（2）在线申请的接收时间为2025年9月15日至2025年9月22日16时。

2. 申请书务必严格按照以下格式填写：
（1）申请书的项目起始时间一律填写2026年1月1日。
（2）"申请代码1"必须选择A25。

三、咨询联系方式
国家自然科学基金委员会数学物理科学部物理科学二处，联系电话：010-62325087。
"""

    chunker = ArticleChunker(max_chunk_size=800, min_chunk_size=100)

    print(f"检测是否为条文文档: {chunker.is_article_document(test_text)}")

    chunks = chunker.chunk_by_articles(test_text)

    print(f"\n生成了 {len(chunks)} 个块:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- 块 {i+1} ---")
        print(f"条文编号: {chunk.get('article_number', '无')}")
        print(f"条文标题: {chunk.get('article_title', '无')}")
        print(f"层级: {chunk.get('article_level', '无')}")
        print(f"父条文: {chunk.get('parent_article', '无')}")
        print(f"内容长度: {len(chunk['content'])}")
        print(f"内容: {chunk['content'][:200]}...")


def test_document_loader_integration():
    """测试文档加载器集成"""
    print("\n" + "=" * 50)
    print("测试文档加载器集成")
    print("=" * 50)

    # 找一个NSFC文件进行测试
    nsfc_dir = Path("data/nsfc_test")
    if not nsfc_dir.exists():
        print("NSFC测试目录不存在，跳过此测试")
        return

    # 找一个包含条文的文件
    test_file = None
    for file_path in nsfc_dir.glob("*.md"):
        if "理论物理" in file_path.name or "指南" in file_path.name:
            test_file = file_path
            break

    if not test_file:
        test_files = list(nsfc_dir.glob("*.md"))
        if test_files:
            test_file = test_files[0]
        else:
            print("未找到测试文件，跳过此测试")
            return

    print(f"测试文件: {test_file.name}")

    # 测试不使用条文分块
    loader_traditional = DocumentLoader(chunk_size=500, use_article_chunking=False)
    doc_traditional = loader_traditional.load_document(str(test_file))
    chunks_traditional = loader_traditional.chunk_document(doc_traditional)

    # 测试使用条文分块
    loader_article = DocumentLoader(chunk_size=500, use_article_chunking=True)
    doc_article = loader_article.load_document(str(test_file))
    chunks_article = loader_article.chunk_document(doc_article)

    print(f"\n传统分块: {len(chunks_traditional)} 个块")
    print(f"条文分块: {len(chunks_article)} 个块")

    # 展示几个条文分块的结果
    print(f"\n条文分块结果预览:")
    for i, chunk in enumerate(chunks_article[:3]):
        print(f"\n--- 块 {i+1} ---")
        print(f"分块类型: {chunk.metadata.get('chunk_type', '未知')}")
        if chunk.metadata.get('chunk_type') == 'article_based':
            print(f"条文编号: {chunk.metadata.get('article_number', '无')}")
            print(f"条文标题: {chunk.metadata.get('article_title', '无')}")
            print(f"条文层级: {chunk.metadata.get('article_level', '无')}")
        print(f"内容长度: {len(chunk.content)}")
        print(f"内容预览: {chunk.content[:150]}...")


def test_real_nsfc_files():
    """测试真实的NSFC文件"""
    print("\n" + "=" * 50)
    print("测试真实NSFC文件的条文识别")
    print("=" * 50)

    nsfc_dir = Path("data/nsfc_test")
    if not nsfc_dir.exists():
        print("NSFC测试目录不存在，跳过此测试")
        return

    chunker = ArticleChunker()
    loader = DocumentLoader(use_article_chunking=True)

    article_files = []
    normal_files = []

    # 检查前10个文件
    for i, file_path in enumerate(nsfc_dir.glob("*.md")):
        if i >= 10:
            break

        try:
            doc = loader.load_document(str(file_path))
            is_article = chunker.is_article_document(doc.content)

            if is_article:
                article_files.append(file_path.name)
            else:
                normal_files.append(file_path.name)

        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错: {e}")

    print(f"\n检测结果:")
    print(f"识别为条文文档的文件 ({len(article_files)} 个):")
    for filename in article_files:
        print(f"  - {filename}")

    print(f"\n识别为普通文档的文件 ({len(normal_files)} 个):")
    for filename in normal_files:
        print(f"  - {filename}")

    # 详细测试一个条文文档
    if article_files:
        test_file = nsfc_dir / article_files[0]
        print(f"\n详细测试条文文档: {test_file.name}")

        doc = loader.load_document(str(test_file))
        chunks = loader.chunk_document(doc)

        article_chunks = [c for c in chunks if c.metadata.get('chunk_type') == 'article_based']
        print(f"生成了 {len(chunks)} 个块，其中 {len(article_chunks)} 个为条文块")

        # 展示条文分块的层级结构
        print(f"\n条文层级结构:")
        for chunk in article_chunks[:5]:  # 只显示前5个
            level = chunk.metadata.get('article_level', 0)
            number = chunk.metadata.get('article_number', '')
            title = chunk.metadata.get('article_title', '')
            indent = "  " * (level - 1) if level > 0 else ""
            print(f"{indent}[层级{level}] {number} {title}")


def main():
    """主测试函数"""
    print("开始测试条文分块功能...")

    try:
        # 基础功能测试
        test_article_chunker_basic()

        # 文档加载器集成测试
        test_document_loader_integration()

        # 真实文件测试
        test_real_nsfc_files()

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n测试完成！")


if __name__ == "__main__":
    main()