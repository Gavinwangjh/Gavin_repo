#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSFC数据处理器 - 清洗和格式化爬取的数据以适应RAG系统
"""

import re
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import os

from .nsfc_scraper import NSFCDocument
from ..document_loader.article_chunker import ArticleChunker

@dataclass
class ProcessedDocument:
    """处理后的文档结构"""
    doc_id: str
    title: str
    content: str
    metadata: Dict
    chunks: List[Dict] = None

class NSFCDataProcessor:
    """NSFC数据处理器"""

    def __init__(self, min_content_length: int = 100, max_chunk_size: int = 1000, use_article_chunking: bool = True):
        """
        初始化数据处理器

        Args:
            min_content_length: 最小内容长度，低于此长度的文档会被过滤
            max_chunk_size: 最大块大小
            use_article_chunking: 是否使用条文分块
        """
        self.min_content_length = min_content_length
        self.max_chunk_size = max_chunk_size
        self.use_article_chunking = use_article_chunking
        self.logger = logging.getLogger(__name__)

        # 初始化条文分块器
        if self.use_article_chunking:
            self.article_chunker = ArticleChunker(
                max_chunk_size=max_chunk_size * 2,
                min_chunk_size=min_content_length
            )

        # 定义需要过滤的无用内容模式
        self.noise_patterns = [
            r'^(首页|返回|上一页|下一页|打印|收藏)$',
            r'^字体[:：]?\s*(大|中|小|\+|\-)$',
            r'^(分享|关注|评论|点赞|转发)$',
            r'^(更新时间|发布时间|浏览次数)[:：]',
            r'^\s*\d+\s*$',  # 纯数字行
            r'^[^\u4e00-\u9fa5a-zA-Z]{1,5}$',  # 特殊符号行
        ]
        
    def clean_text(self, text: str) -> str:
        """清洗文本内容"""
        if not text:
            return ""
        
        # 基础清理
        text = text.strip()
        
        # 规范化空白字符
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        text = re.sub(r'\t', ' ', text)
        
        # 移除多余空格
        text = re.sub(r'[ ]{2,}', ' ', text)
        
        # 规范化换行
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # 逐行清理
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否匹配噪音模式
            is_noise = False
            for pattern in self.noise_patterns:
                if re.match(pattern, line):
                    is_noise = True
                    break
            
            if not is_noise:
                cleaned_lines.append(line)
        
        # 重新组合
        cleaned_text = '\n'.join(cleaned_lines)
        
        # 最终清理
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        return cleaned_text.strip()
    
    def extract_metadata(self, doc: NSFCDocument) -> Dict:
        """提取文档元数据"""
        metadata = {
            'source': doc.source,
            'url': doc.url,
            'date': doc.date,
            'doc_type': doc.doc_type,
            'title': doc.title,
            'content_length': len(doc.content),
            'processed_at': datetime.now().isoformat(),
        }
        
        # 从标题提取更多信息
        title_lower = doc.title.lower()
        
        # 识别文档类型
        if any(keyword in title_lower for keyword in ['指南', '申请须知', '指导']):
            metadata['category'] = '申请指南'
        elif any(keyword in title_lower for keyword in ['公告', '通知', '通报']):
            metadata['category'] = '公告通知'
        elif any(keyword in title_lower for keyword in ['项目', '基金', '资助']):
            metadata['category'] = '项目信息'
        elif any(keyword in title_lower for keyword in ['国际', '合作', '交流']):
            metadata['category'] = '国际合作'
        else:
            metadata['category'] = '其他'
        
        # 提取年份
        if doc.date:
            try:
                year_match = re.search(r'(\d{4})', doc.date)
                if year_match:
                    metadata['year'] = int(year_match.group(1))
            except:
                pass
        
        # 从标题提取关键词
        keywords = []
        keyword_patterns = [
            r'(国际.*?合作)',
            r'(.*?基金)',
            r'(.*?项目)',
            r'(.*?计划)',
            r'(.*?专项)',
        ]
        
        for pattern in keyword_patterns:
            matches = re.findall(pattern, doc.title)
            keywords.extend(matches)
        
        if keywords:
            metadata['keywords'] = keywords
        
        return metadata
    
    def generate_doc_id(self, doc: NSFCDocument) -> str:
        """生成唯一文档ID"""
        # 使用URL和标题生成唯一ID
        content_for_hash = f"{doc.url}_{doc.title}_{doc.date}"
        return hashlib.md5(content_for_hash.encode('utf-8')).hexdigest()[:16]
    
    def split_into_chunks(self, text: str, max_size: int = None) -> List[str]:
        """将长文本分割成适合的块"""
        if max_size is None:
            max_size = self.max_chunk_size
        
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        
        # 优先按段落分割
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            # 如果单个段落就超过最大长度，需要进一步分割
            if len(paragraph) > max_size:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # 分割超长段落
                sentences = re.split(r'[。！？；;]', paragraph)
                temp_chunk = ""
                
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    sentence = sentence.strip() + '。'
                    
                    if len(temp_chunk + sentence) <= max_size:
                        temp_chunk += sentence
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = sentence
                
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                # 检查加入当前段落是否会超出限制
                if len(current_chunk + paragraph) <= max_size:
                    if current_chunk:
                        current_chunk += '\n\n' + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    # 保存当前块，开始新块
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
        
        # 保存最后的块
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 过滤太短的块
        chunks = [chunk for chunk in chunks if len(chunk) >= 50]
        
        return chunks
    
    def create_chunks_with_metadata(self, doc: ProcessedDocument) -> List[Dict]:
        """为文档创建带元数据的块"""
        # 检查是否使用条文分块
        if self.use_article_chunking and hasattr(self, 'article_chunker'):
            if self.article_chunker.is_article_document(doc.content):
                self.logger.info(f"检测到条文文档，使用条文分块: {doc.title}")
                return self._create_article_chunks(doc)

        # 使用传统分块方法
        self.logger.info(f"使用传统分块方法: {doc.title}")
        return self._create_traditional_chunks(doc)

    def _create_article_chunks(self, doc: ProcessedDocument) -> List[Dict]:
        """使用条文分块器创建块"""
        try:
            article_chunks = self.article_chunker.chunk_by_articles(doc.content)
            chunk_list = []

            for chunk_data in article_chunks:
                chunk = {
                    'doc_id': doc.doc_id,
                    'chunk_id': f"{doc.doc_id}_chunk_{chunk_data['chunk_index']}",
                    'content': chunk_data['content'],
                    'chunk_index': chunk_data['chunk_index'],
                    'total_chunks': len(article_chunks),
                    'metadata': {
                        **doc.metadata,
                        'chunk_size': len(chunk_data['content']),
                        'chunk_type': chunk_data.get('chunk_type', 'article_based'),
                        'article_number': chunk_data.get('article_number'),
                        'article_title': chunk_data.get('article_title'),
                        'article_level': chunk_data.get('article_level'),
                        'parent_article': chunk_data.get('parent_article'),
                        'is_first_chunk': chunk_data['chunk_index'] == 0,
                        'is_last_chunk': chunk_data['chunk_index'] == len(article_chunks) - 1,
                    }
                }
                chunk_list.append(chunk)

            return chunk_list

        except Exception as e:
            self.logger.error(f"条文分块失败，回退到传统分块: {str(e)}")
            return self._create_traditional_chunks(doc)

    def _create_traditional_chunks(self, doc: ProcessedDocument) -> List[Dict]:
        """使用传统方法创建块"""
        chunks = self.split_into_chunks(doc.content)

        chunk_list = []
        for i, chunk_content in enumerate(chunks):
            chunk = {
                'doc_id': doc.doc_id,
                'chunk_id': f"{doc.doc_id}_chunk_{i}",
                'content': chunk_content,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'metadata': {
                    **doc.metadata,
                    'chunk_size': len(chunk_content),
                    'chunk_type': 'size_based',
                    'is_first_chunk': i == 0,
                    'is_last_chunk': i == len(chunks) - 1,
                }
            }
            chunk_list.append(chunk)

        return chunk_list
    
    def process_documents(self, documents: List[NSFCDocument]) -> List[ProcessedDocument]:
        """处理文档列表"""
        processed_docs = []
        
        self.logger.info(f"开始处理 {len(documents)} 个文档...")
        
        for doc in documents:
            try:
                # 清洗内容
                cleaned_content = self.clean_text(doc.content)
                
                # 过滤内容太短的文档
                if len(cleaned_content) < self.min_content_length:
                    self.logger.debug(f"跳过内容过短的文档: {doc.title}")
                    continue
                
                # 生成文档ID
                doc_id = self.generate_doc_id(doc)
                
                # 提取元数据
                metadata = self.extract_metadata(doc)
                
                # 创建处理后的文档
                processed_doc = ProcessedDocument(
                    doc_id=doc_id,
                    title=doc.title,
                    content=cleaned_content,
                    metadata=metadata
                )
                
                # 创建块
                chunks = self.create_chunks_with_metadata(processed_doc)
                processed_doc.chunks = chunks
                
                processed_docs.append(processed_doc)
                
                self.logger.debug(f"处理完成: {doc.title} -> {len(chunks)} 个块")
                
            except Exception as e:
                self.logger.error(f"处理文档时出错 {doc.title}: {e}")
                continue
        
        self.logger.info(f"处理完成，得到 {len(processed_docs)} 个有效文档")
        return processed_docs
    
    def save_processed_data(self, processed_docs: List[ProcessedDocument], output_dir: str = "data/processed_nsfc"):
        """保存处理后的数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存文档级别的数据
        docs_data = []
        all_chunks = []
        
        for doc in processed_docs:
            doc_data = {
                'doc_id': doc.doc_id,
                'title': doc.title,
                'content': doc.content,
                'metadata': doc.metadata,
                'chunk_count': len(doc.chunks) if doc.chunks else 0
            }
            docs_data.append(doc_data)
            
            if doc.chunks:
                all_chunks.extend(doc.chunks)
        
        # 保存文档数据
        docs_file = os.path.join(output_dir, 'documents.json')
        with open(docs_file, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        
        # 保存块数据
        chunks_file = os.path.join(output_dir, 'chunks.json')
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        # 保存统计信息
        stats = {
            'total_documents': len(processed_docs),
            'total_chunks': len(all_chunks),
            'avg_chunks_per_doc': len(all_chunks) / len(processed_docs) if processed_docs else 0,
            'avg_content_length': sum(len(doc.content) for doc in processed_docs) / len(processed_docs) if processed_docs else 0,
            'categories': {},
            'years': {},
            'processed_at': datetime.now().isoformat()
        }
        
        # 统计分类
        for doc in processed_docs:
            category = doc.metadata.get('category', '未知')
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            year = doc.metadata.get('year')
            if year:
                stats['years'][str(year)] = stats['years'].get(str(year), 0) + 1
        
        stats_file = os.path.join(output_dir, 'stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"处理后的数据已保存到 {output_dir}")
        self.logger.info(f"文档: {len(processed_docs)}, 块: {len(all_chunks)}")
        
        return {
            'documents': docs_data,
            'chunks': all_chunks,
            'stats': stats
        }
    
    def get_rag_ready_format(self, processed_docs: List[ProcessedDocument]) -> List[Dict]:
        """转换为RAG系统可直接使用的格式"""
        rag_documents = []
        
        for doc in processed_docs:
            if not doc.chunks:
                continue
            
            for chunk in doc.chunks:
                rag_doc = {
                    'content': chunk['content'],
                    'doc_id': chunk['doc_id'],
                    'chunk_id': chunk['chunk_id'],
                    'title': doc.title,
                    'source': 'NSFC',
                    'url': doc.metadata.get('url', ''),
                    'date': doc.metadata.get('date', ''),
                    'category': doc.metadata.get('category', ''),
                    'metadata': json.dumps(chunk['metadata'], ensure_ascii=False)
                }
                rag_documents.append(rag_doc)
        
        return rag_documents

async def process_nsfc_data(raw_documents: List[NSFCDocument], output_dir: str = "data/processed_nsfc") -> Dict:
    """处理NSFC爬取数据的主函数"""
    processor = NSFCDataProcessor(
        min_content_length=100,
        max_chunk_size=1000
    )
    
    # 处理文档
    processed_docs = processor.process_documents(raw_documents)
    
    # 保存处理后的数据
    result = processor.save_processed_data(processed_docs, output_dir)
    
    # 生成RAG格式数据
    rag_documents = processor.get_rag_ready_format(processed_docs)
    
    return {
        **result,
        'rag_documents': rag_documents
    }

if __name__ == "__main__":
    # 测试代码
    import asyncio
    from nsfc_scraper import NSFCScraper
    
    async def test_process():
        # 先爬取少量数据用于测试
        async with NSFCScraper(max_pages=1, delay=0.5) as scraper:
            documents = await scraper.scrape_all_pages()
        
        if documents:
            # 处理数据
            result = await process_nsfc_data(documents)
            print(f"处理结果: {len(result['documents'])} 个文档, {len(result['chunks'])} 个块")
    
    asyncio.run(test_process())