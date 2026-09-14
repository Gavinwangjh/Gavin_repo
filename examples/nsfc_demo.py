#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSFC爬虫使用示例
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.nsfc_scraper import NSFCScraper
from src.scrapers.data_processor import process_nsfc_data

async def demo_scrape_and_process():
    """演示爬取和处理NSFC数据"""
    
    print("🚀 NSFC爬虫演示开始...")
    
    # 配置爬虫 - 限制页数用于演示
    scraper_config = {
        'max_pages': 2,  # 只爬取前2页用于演示
        'delay': 1.0,    # 请求间隔1秒
        'use_selenium': False  # 优先使用aiohttp
    }
    
    try:
        # 步骤1: 爬取原始数据
        print("\n📡 步骤1: 爬取NSFC网站数据...")
        async with NSFCScraper(**scraper_config) as scraper:
            documents = await scraper.scrape_all_pages()
            
            if not documents:
                print("❌ 未获取到任何数据")
                return
            
            print(f"✅ 爬取完成，获得 {len(documents)} 个文档")
            
            # 显示前3个文档的信息
            print("\n📋 前3个文档预览:")
            for i, doc in enumerate(documents[:3]):
                print(f"{i+1}. 标题: {doc.title}")
                print(f"   日期: {doc.date}")
                print(f"   内容长度: {len(doc.content)} 字符")
                print(f"   内容预览: {doc.content[:100]}...")
                print()
        
        # 步骤2: 数据处理
        print("🔄 步骤2: 处理和清洗数据...")
        processed_data = await process_nsfc_data(documents, "data/demo_nsfc_processed")
        
        print(f"✅ 数据处理完成:")
        print(f"  📊 有效文档: {len(processed_data['documents'])}")
        print(f"  🧩 文档块数: {len(processed_data['chunks'])}")
        print(f"  📈 平均每文档块数: {processed_data['stats']['avg_chunks_per_doc']:.1f}")
        
        # 显示分类统计
        print("\n📊 文档分类统计:")
        for category, count in processed_data['stats']['categories'].items():
            print(f"  📁 {category}: {count} 个文档")
        
        # 显示年份统计
        if processed_data['stats']['years']:
            print("\n📅 年份分布:")
            for year, count in sorted(processed_data['stats']['years'].items()):
                print(f"  🗓️ {year}: {count} 个文档")
        
        # 步骤3: 展示RAG格式数据
        print(f"\n📋 RAG格式数据示例 (共{len(processed_data['rag_documents'])}个块):")
        for i, rag_doc in enumerate(processed_data['rag_documents'][:2]):
            print(f"\n块 {i+1}:")
            print(f"  ID: {rag_doc['chunk_id']}")
            print(f"  标题: {rag_doc['title']}")
            print(f"  分类: {rag_doc['category']}")
            print(f"  内容长度: {len(rag_doc['content'])} 字符")
            print(f"  内容预览: {rag_doc['content'][:150]}...")
        
        print(f"\n🎉 演示完成！数据已保存到 data/demo_nsfc_processed/ 目录")
        print(f"💡 提示: 使用以下命令将数据上传到RAG系统:")
        print(f"   python -m src.cli.cli upload-nsfc data/demo_nsfc_processed/chunks.json")
        
    except Exception as e:
        print(f"❌ 演示过程出错: {str(e)}")
        raise

async def demo_quick_test():
    """快速测试爬虫功能"""
    print("🔧 快速测试NSFC爬虫...")
    
    try:
        async with NSFCScraper(max_pages=1, delay=0.5) as scraper:
            # 只获取第一页内容
            content = await scraper.get_page_content(scraper.list_url)
            
            if content:
                # 解析页面
                documents_info = scraper.parse_list_page(content)
                pagination_info = scraper.get_pagination_info(content)
                
                print(f"✅ 连接正常")
                print(f"📄 第一页发现 {len(documents_info)} 个文档")
                print(f"📊 总计 {pagination_info['total_records']} 条记录，{pagination_info['total_pages']} 页")
                
                if documents_info:
                    print(f"📋 第一个文档: {documents_info[0]['title']}")
                    print(f"📅 发布日期: {documents_info[0]['date']}")
                
                return True
            else:
                print("❌ 无法获取页面内容")
                return False
                
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NSFC爬虫演示')
    parser.add_argument('--test', action='store_true', help='快速测试')
    parser.add_argument('--demo', action='store_true', help='完整演示')
    
    args = parser.parse_args()
    
    if args.test:
        success = asyncio.run(demo_quick_test())
        sys.exit(0 if success else 1)
    elif args.demo:
        asyncio.run(demo_scrape_and_process())
    else:
        print("使用方法:")
        print("  python nsfc_demo.py --test   # 快速测试")
        print("  python nsfc_demo.py --demo   # 完整演示")

if __name__ == "__main__":
    main()