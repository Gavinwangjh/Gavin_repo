#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSFC (National Natural Science Foundation of China) Web Scraper
爬取国家自然科学基金委员会公开信息
"""

import asyncio
import time
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import logging

import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

@dataclass
class NSFCDocument:
    """NSFC文档数据结构"""
    title: str
    url: str
    date: str
    content: str = ""
    doc_type: str = "公告"
    source: str = "NSFC"

class NSFCScraper:
    """国家自然科学基金委员会爬虫"""
    
    def __init__(
        self,
        max_pages: int = None,
        delay: float = 1.0,
        use_selenium: bool = False,
        list_url: Optional[str] = None
    ):
        """
        初始化爬虫
        
        Args:
            max_pages: 最大爬取页数，None表示全部
            delay: 请求间隔时间（秒）
            use_selenium: 是否使用Selenium处理JavaScript
        """
        self.base_url = "https://www.nsfc.gov.cn"
        self.list_url = list_url or "https://www.nsfc.gov.cn/p1/2931/3441/2025ndxmznlb.html"
        self.max_pages = max_pages
        self.delay = delay
        self.use_selenium = use_selenium
        self.session = None
        self.driver = None
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    def _get_list_page_url(self, page_num: int) -> str:
        """Build list page URLs for the current NSFC static pagination format."""
        if page_num <= 1:
            return self.list_url

        if self.list_url.endswith(".html"):
            base, ext = self.list_url.rsplit(".", 1)
            if re.search(r"_\d+$", base):
                base = re.sub(r"_\d+$", "", base)
            return f"{base}_{page_num}.{ext}"

        return f"{self.list_url}?page={page_num}"

    async def __aenter__(self):
        """异步上下文管理器入口"""
        if self.use_selenium:
            await self._init_selenium()
        else:
            await self._init_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.driver:
            self.driver.quit()
        if self.session:
            await self.session.close()

    async def _init_session(self):
        """初始化aiohttp会话"""
        connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            connector=connector,
            timeout=timeout
        )

    async def _init_selenium(self):
        """初始化Selenium WebDriver"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument(f'--user-agent={self.headers["User-Agent"]}')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(10)

    async def get_page_content(self, url: str) -> str:
        """获取页面内容"""
        if self.use_selenium:
            return await self._get_page_with_selenium(url)
        else:
            return await self._get_page_with_aiohttp(url)

    async def _get_page_with_aiohttp(self, url: str) -> str:
        """使用aiohttp获取页面"""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text(encoding='utf-8')
                    return content
                else:
                    self.logger.warning(f"HTTP {response.status} for {url}")
                    return ""
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return ""

    async def _get_page_with_selenium(self, url: str) -> str:
        """使用Selenium获取页面"""
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "list"))
            )
            return self.driver.page_source
        except TimeoutException:
            self.logger.warning(f"Timeout loading {url}")
            return self.driver.page_source
        except Exception as e:
            self.logger.error(f"Selenium error for {url}: {e}")
            return ""

    def parse_list_page(self, html_content: str) -> List[Dict[str, str]]:
        """解析列表页面，提取文档链接和基本信息"""
        soup = BeautifulSoup(html_content, 'html.parser')
        documents = []
        
        # 查找文档列表
        list_items = soup.find_all('li')
        
        for item in list_items:
            try:
                # 查找链接
                link = item.find('a')
                if not link or not link.get('href'):
                    continue
                
                title = link.get_text(strip=True)
                if not title:
                    continue
                    
                href = link.get('href')
                full_url = urljoin(self.base_url, href)
                
                # 查找日期
                date_text = ""
                date_span = item.find('span', class_='time') or item.find(string=re.compile(r'\d{4}-\d{2}-\d{2}'))
                if date_span:
                    if hasattr(date_span, 'get_text'):
                        date_text = date_span.get_text(strip=True)
                    else:
                        date_text = str(date_span).strip()
                
                # 清理日期格式
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', date_text)
                if date_match:
                    date_text = date_match.group(1)
                
                documents.append({
                    'title': title,
                    'url': full_url,
                    'date': date_text
                })
                
            except Exception as e:
                self.logger.warning(f"Error parsing list item: {e}")
                continue
        
        return documents

    async def get_document_content(self, url: str) -> str:
        """获取文档详细内容"""
        content = await self.get_page_content(url)
        if not content:
            return ""
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # 尝试多种内容选择器
        content_selectors = [
            '.content',
            '.main-content', 
            '.article-content',
            '.text-content',
            '#content',
            '.detail-content',
            '.TRS_Editor',
            '.article',
            '.detail'
        ]
        
        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                # 清理内容
                text = content_div.get_text(separator='\n', strip=True)
                return self._clean_content(text)
        
        # 如果没找到特定容器，尝试获取body内容
        body = soup.find('body')
        if body:
            # 移除导航、脚本等无关内容
            for tag in body.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            
            text = body.get_text(separator='\n', strip=True)
            return self._clean_content(text)
        
        return ""

    def _clean_content(self, text: str) -> str:
        """清理文档内容"""
        if not text:
            return ""
        
        # 移除多余空白
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 移除常见的页面元素
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 跳过导航类文本
            if any(keyword in line for keyword in ['首页', '返回', '上一页', '下一页', '打印', '收藏', '字体']):
                continue
            
            # 跳过过短的行（通常是导航元素）
            if len(line) < 3:
                continue
                
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def get_pagination_info(self, html_content: str) -> Dict[str, int]:
        """获取分页信息"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 查找分页信息
        pagination_info = {'current_page': 1, 'total_pages': 1, 'total_records': 0}
        
        # 查找页码信息
        page_info = soup.find(string=re.compile(r'第\s*\d+\s*页.*共\s*\d+\s*页'))
        if page_info:
            page_match = re.search(r'第\s*(\d+)\s*页.*共\s*(\d+)\s*页', page_info)
            if page_match:
                pagination_info['current_page'] = int(page_match.group(1))
                pagination_info['total_pages'] = int(page_match.group(2))
        
        # 查找总记录数
        record_info = soup.find(string=re.compile(r'共\s*\d+\s*条'))
        if record_info:
            record_match = re.search(r'共\s*(\d+)\s*条', record_info)
            if record_match:
                pagination_info['total_records'] = int(record_match.group(1))
        
        page_text = soup.get_text(" ", strip=True)

        modern_page_match = re.search(r'当前页\s*[:：]\s*(\d+)\s*/\s*(\d+)', page_text)
        if modern_page_match:
            pagination_info['current_page'] = int(modern_page_match.group(1))
            pagination_info['total_pages'] = int(modern_page_match.group(2))

        modern_record_match = re.search(r'总记录数\s*[:：]\s*(\d+)', page_text)
        if modern_record_match:
            pagination_info['total_records'] = int(modern_record_match.group(1))

        static_page_numbers = [
            int(number)
            for number in re.findall(r'_(\d+)\.html', html_content)
        ]
        if static_page_numbers:
            pagination_info['total_pages'] = max(
                pagination_info['total_pages'],
                max(static_page_numbers)
            )

        return pagination_info

    async def scrape_all_pages(self) -> List[NSFCDocument]:
        """爬取所有页面"""
        all_documents = []
        
        self.logger.info("开始爬取NSFC网站...")
        
        # 获取第一页以了解总页数
        first_page_content = await self.get_page_content(self.list_url)
        if not first_page_content:
            self.logger.error("无法获取首页内容")
            return []
        
        pagination_info = self.get_pagination_info(first_page_content)
        total_pages = pagination_info['total_pages']
        total_records = pagination_info['total_records']
        
        self.logger.info(f"发现 {total_records} 条记录，共 {total_pages} 页")
        
        # 如果设置了最大页数限制
        if self.max_pages:
            total_pages = min(total_pages, self.max_pages)
            self.logger.info(f"限制爬取页数为: {total_pages}")
        
        # 爬取每一页
        for page_num in range(1, total_pages + 1):
            self.logger.info(f"正在爬取第 {page_num}/{total_pages} 页...")
            
            # 构造页面URL
            page_url = self._get_list_page_url(page_num)
            
            page_content = await self.get_page_content(page_url)
            if not page_content:
                self.logger.warning(f"无法获取第 {page_num} 页内容")
                continue
            
            # 解析页面文档列表
            documents_info = self.parse_list_page(page_content)
            self.logger.info(f"第 {page_num} 页找到 {len(documents_info)} 个文档")
            
            # 获取每个文档的详细内容
            for doc_info in documents_info:
                try:
                    self.logger.debug(f"获取文档内容: {doc_info['title']}")
                    content = await self.get_document_content(doc_info['url'])
                    
                    document = NSFCDocument(
                        title=doc_info['title'],
                        url=doc_info['url'],
                        date=doc_info['date'],
                        content=content
                    )
                    
                    all_documents.append(document)
                    
                    # 添加延迟避免过于频繁的请求
                    await asyncio.sleep(self.delay)
                    
                except Exception as e:
                    self.logger.error(f"处理文档时出错 {doc_info['url']}: {e}")
                    continue
            
            # 页面间延迟
            await asyncio.sleep(self.delay * 2)
        
        self.logger.info(f"爬取完成，共获得 {len(all_documents)} 个文档")
        return all_documents

    async def save_to_files(self, documents: List[NSFCDocument], output_dir: str = "data/nsfc_docs"):
        """保存文档到文件"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, doc in enumerate(documents):
            # 生成安全的文件名
            safe_title = re.sub(r'[^\w\s-]', '', doc.title)[:50]
            filename = f"{i+1:03d}_{safe_title}_{doc.date}.md"
            filepath = os.path.join(output_dir, filename)
            
            content = f"""# {doc.title}

**发布日期**: {doc.date}
**来源**: {doc.source}
**链接**: {doc.url}

---

{doc.content}
"""
            
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(content)
        
        self.logger.info(f"文档已保存到 {output_dir} 目录")

async def main():
    """主函数示例"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建爬虫实例
    scraper_config = {
        'max_pages': 3,  # 限制爬取页数用于测试
        'delay': 1.0,    # 请求间隔
        'use_selenium': False  # 如果页面需要JavaScript渲染，设为True
    }
    
    async with NSFCScraper(**scraper_config) as scraper:
        # 爬取文档
        documents = await scraper.scrape_all_pages()
        
        # 保存到文件
        if documents:
            await scraper.save_to_files(documents)
            
            # 打印统计信息
            print(f"\n爬取统计:")
            print(f"总文档数: {len(documents)}")
            print(f"平均内容长度: {sum(len(doc.content) for doc in documents) // len(documents) if documents else 0} 字符")
            
            # 显示前几个文档标题
            print(f"\n前5个文档:")
            for i, doc in enumerate(documents[:5]):
                print(f"{i+1}. {doc.title} ({doc.date})")

if __name__ == "__main__":
    asyncio.run(main())
