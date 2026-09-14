import asyncio
import httpx
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import tiktoken
from transformers import AutoTokenizer

from loguru import logger
from src.utils.helpers import get_config, retry_on_failure


@dataclass
class EmbeddingResult:
    """嵌入结果数据结构"""
    embeddings: List[List[float]]
    texts: List[str]
    model: str
    dimension: int
    token_count: Optional[int] = None


class SiliconFlowEmbedder:
    """SiliconFlow嵌入服务客户端"""

    def __init__(
            self,
            api_key: Optional[str] = None,
            model: str = "BAAI/bge-large-zh-v1.5",
            api_base: str = "https://api.siliconflow.cn/v1",
            batch_size: int = 10,  # 减小批量大小以减少潜在问题
            max_retries: int = 3,
            timeout: int = 30,
            enable_cache: bool = True,
            cache_dir: str = "data/embedding_cache",
            max_concurrent: int = 3,
            request_timeout: int = 60
            
    ):
        self.api_key = api_key or get_config("api_keys.siliconflow_api_key")
        self.model = model
        self.api_base = api_base.rstrip('/')
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_cache = enable_cache
        self.max_concurrent = max_concurrent
        self.request_timeout = request_timeout

            # 初始化缓存管理器
        if enable_cache:
            from .cache_manager import EmbeddingCacheManager
            self.cache_manager = EmbeddingCacheManager(cache_dir=cache_dir)
        else:
            self.cache_manager = None
        
        # 异步信号量控制并发
        self.semaphore = None  # 在异步方法中初始化
        
        # 性能统计
        self.performance_stats = {
            "total_requests": 0,
            "total_texts": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "total_time": 0.0,
            "avg_time_per_request": 0.0
        }
        
        logger.info(f"嵌入器优化配置: 缓存={enable_cache}, 并发数={max_concurrent}, 超时={timeout}s")

        if not self.api_key:
            raise ValueError("SiliconFlow API密钥未设置")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 获取模型维度
        self.dimension = get_config("embedding.dimension", 1024)

        # 模型token限制
        model_token_limits = {
            "BAAI/bge-large-zh-v1.5": 512,
            "BAAI/bge-large-en-v1.5": 512,
            "netease-youdao/bce-embedding-base_v1": 512,
            "BAAI/bge-m3": 8192,
            "Pro/BAAI/bge-m3": 8192,
            "Qwen/Qwen3-Embedding-8B": 32768,
            "Qwen/Qwen3-Embedding-4B": 32768,
            "Qwen/Qwen3-Embedding-0.6B": 32768,
        }
        self.max_input_tokens = model_token_limits.get(self.model, 512)

        # 加载模型特定的 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model, local_files_only=True)
            logger.info(f"成功加载 tokenizer for model: {self.model}")
        except Exception as e:
            logger.warning(f"无法加载模型 tokenizer，使用 fallback tiktoken: {str(e)}")
            self.tokenizer = None  # fallback to tiktoken

        logger.info(
            f"SiliconFlow嵌入器初始化: model={self.model}, dimension={self.dimension}, max_tokens={self.max_input_tokens}")

    @retry_on_failure(max_retries=3, delay=1.0)
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """嵌入文本列表"""
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                texts=[],
                model=self.model,
                dimension=self.dimension
            )

        logger.info(f"开始嵌入 {len(texts)} 个文本")

        # 分批处理
        all_embeddings = []
        all_texts = []
        total_tokens = 0

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_result = self._embed_batch(batch_texts)

            all_embeddings.extend(batch_result.embeddings)
            all_texts.extend(batch_result.texts)
            if batch_result.token_count:
                total_tokens += batch_result.token_count

            logger.debug(f"批次 {i // self.batch_size + 1} 完成: {len(batch_texts)} 个文本")

        result = EmbeddingResult(
            embeddings=all_embeddings,
            texts=all_texts,
            model=self.model,
            dimension=self.dimension,
            token_count=total_tokens if total_tokens > 0 else None
        )

        logger.info(f"嵌入完成: {len(all_embeddings)} 个向量, 维度: {self.dimension}")
        return result

    def _embed_batch(self, texts: List[str]) -> EmbeddingResult:
        """嵌入单个批次"""
        # 过滤空字符串并分割超长文本
        filtered_texts = []
        for t in texts:
            if not t.strip():
                continue
            # 分割超长文本
            chunks = TextProcessor.split_long_text(t, max_tokens=self.max_input_tokens, tokenizer=self.tokenizer)
            for chunk in chunks:
                token_count = self._count_tokens(chunk)
                if token_count >= self.max_input_tokens:
                    logger.warning(f"分割后仍超长，跳过: {token_count} tokens, 预览: {chunk[:50]}...")
                    continue
                filtered_texts.append(chunk)

        if not filtered_texts:
            logger.warning("批次中无有效文本可嵌入")
            return EmbeddingResult(
                embeddings=[],
                texts=[],
                model=self.model,
                dimension=self.dimension
            )

        # 记录批次中最大 token
        max_tokens_in_batch = max(self._count_tokens(t) for t in filtered_texts)
        logger.debug(f"批次最大 token 数: {max_tokens_in_batch}")

        payload = {
            "model": self.model,
            "input": filtered_texts,
            "encoding_format": "float"
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.api_base}/embeddings",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()

                data = response.json()

                # 提取嵌入向量
                embeddings = []
                for item in data["data"]:
                    embeddings.append(item["embedding"])

                # 获取token使用量
                token_count = data.get("usage", {}).get("total_tokens")

                return EmbeddingResult(
                    embeddings=embeddings,
                    texts=filtered_texts,
                    model=self.model,
                    dimension=len(embeddings[0]) if embeddings else self.dimension,
                    token_count=token_count
                )

        except httpx.HTTPStatusError as e:
            logger.error(f"SiliconFlow API HTTP错误: {e.response.status_code} - {e.response.text}")
            raise Exception(f"嵌入API调用失败: HTTP {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"SiliconFlow API请求错误: {str(e)}")
            raise Exception(f"嵌入API请求失败: {str(e)}")
        except Exception as e:
            logger.error(f"嵌入处理错误: {str(e)}")
            raise

    def _count_tokens(self, text: str) -> int:
        """使用模型 tokenizer 或 fallback 计数 tokens"""
        if not text:
            return 0
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            return max(1, len(text) // 2)

    def embed_single_text(self, text: str) -> List[float]:
        """嵌入单个文本"""
        result = self.embed_texts([text])
        return result.embeddings[0] if result.embeddings else []

    async def embed_texts_async(self, texts: List[str]) -> EmbeddingResult:
        """异步嵌入文本列表"""
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                texts=[],
                model=self.model,
                dimension=self.dimension
            )

        logger.info(f"开始异步嵌入 {len(texts)} 个文本")

        # 分批处理
        all_embeddings = []
        all_texts = []
        total_tokens = 0

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = []

            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                task = self._embed_batch_async(client, batch_texts)
                tasks.append(task)

            # 并发执行所有批次
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"批次 {i} 处理失败: {str(result)}")
                    raise result

                all_embeddings.extend(result.embeddings)
                all_texts.extend(result.texts)
                if result.token_count:
                    total_tokens += result.token_count

        result = EmbeddingResult(
            embeddings=all_embeddings,
            texts=all_texts,
            model=self.model,
            dimension=self.dimension,
            token_count=total_tokens if total_tokens > 0 else None
        )

        logger.info(f"异步嵌入完成: {len(all_embeddings)} 个向量")
        return result

    async def _embed_batch_async(self, client: httpx.AsyncClient, texts: List[str]) -> EmbeddingResult:
        """异步嵌入单个批次"""
        # 过滤空字符串并分割超长文本
        filtered_texts = []
        for t in texts:
            if not t.strip():
                continue
            # 分割超长文本
            chunks = TextProcessor.split_long_text(t, max_tokens=self.max_input_tokens, tokenizer=self.tokenizer)
            for chunk in chunks:
                token_count = self._count_tokens(chunk)
                if token_count >= self.max_input_tokens:
                    logger.warning(f"分割后仍超长，跳过: {token_count} tokens, 预览: {chunk[:50]}...")
                    continue
                filtered_texts.append(chunk)

        if not filtered_texts:
            logger.warning("批次中无有效文本可嵌入")
            return EmbeddingResult(
                embeddings=[],
                texts=[],
                model=self.model,
                dimension=self.dimension
            )

        # 记录批次中最大 token
        max_tokens_in_batch = max(self._count_tokens(t) for t in filtered_texts)
        logger.debug(f"批次最大 token 数: {max_tokens_in_batch}")

        payload = {
            "model": self.model,
            "input": filtered_texts,
            "encoding_format": "float"
        }

        try:
            response = await client.post(
                f"{self.api_base}/embeddings",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()

            # 提取嵌入向量
            embeddings = []
            for item in data["data"]:
                embeddings.append(item["embedding"])

            # 获取token使用量
            token_count = data.get("usage", {}).get("total_tokens")

            return EmbeddingResult(
                embeddings=embeddings,
                texts=filtered_texts,
                model=self.model,
                dimension=len(embeddings[0]) if embeddings else self.dimension,
                token_count=token_count
            )

        except Exception as e:
            logger.error(f"异步嵌入批次失败: {str(e)}")
            raise
        
    def embed_texts_optimized(self, texts: List[str]) -> EmbeddingResult:
        """优化版嵌入文本（带缓存和批量处理）"""
        start_time = time.time()
        self.performance_stats["total_requests"] += 1
        self.performance_stats["total_texts"] += len(texts)
        
        if not texts:
            return EmbeddingResult(
                embeddings=[], texts=[], model=self.model, dimension=self.dimension
            )
        
        # 1. 使用缓存
        if self.enable_cache and self.cache_manager:
            return self._embed_with_cache(texts)
        else:
            # 2. 直接调用API
            return self._embed_batch(texts)
    
    def _embed_with_cache(self, texts: List[str]) -> EmbeddingResult:
        """使用缓存的嵌入"""
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        # 分离缓存和未缓存的文本
        for i, text in enumerate(texts):
            cached = self.cache_manager.get(self.model, text)
            if cached is not None:
                cached_embeddings[i] = cached
                self.performance_stats["cache_hits"] += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # 如果没有未缓存的文本，直接返回
        if not uncached_texts:
            all_embeddings = [cached_embeddings[i] for i in range(len(texts))]
            return EmbeddingResult(
                embeddings=all_embeddings,
                texts=texts,
                model=self.model,
                dimension=self.dimension
            )
        
        # 处理未缓存的文本
        self.performance_stats["api_calls"] += 1
        uncached_result = self._embed_batch(uncached_texts)
        
        # 缓存新计算的结果
        for idx, text, embedding in zip(
            uncached_indices, 
            uncached_result.texts, 
            uncached_result.embeddings
        ):
            self.cache_manager.set(self.model, text, embedding)
            cached_embeddings[idx] = embedding
        
        # 按原始顺序重组
        all_embeddings = [cached_embeddings[i] for i in range(len(texts))]
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            texts=texts,
            model=self.model,
            dimension=self.dimension
        )
    
    async def embed_texts_async_optimized(self, texts: List[str]) -> EmbeddingResult:
        """优化版异步嵌入（并发控制）"""
        if not texts:
            return EmbeddingResult(
                embeddings=[], texts=[], model=self.model, dimension=self.dimension
            )
        
        # 初始化信号量
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_batch(batch_texts: List[str]) -> EmbeddingResult:
            """处理单个批次"""
            async with self.semaphore:
                return await self._embed_batch_async_with_retry(batch_texts)
        
        # 分批处理
        batch_size = min(self.batch_size, 5)  # 减小批次大小提高并发性
        tasks = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            task = asyncio.create_task(process_batch(batch))
            tasks.append(task)
        
        # 并发执行并收集结果
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并结果
        all_embeddings = []
        all_texts = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"批次嵌入失败: {result}")
                continue
            all_embeddings.extend(result.embeddings)
            all_texts.extend(result.texts)
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            texts=all_texts,
            model=self.model,
            dimension=self.dimension
        )
    
    async def _embed_batch_async_with_retry(self, texts: List[str], max_retries: int = 3) -> EmbeddingResult:
        """带重试机制的异步嵌入"""
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                    return await self._embed_batch_async(client, texts)
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt  # 指数退避
                await asyncio.sleep(wait_time)
                logger.warning(f"嵌入请求失败，第{attempt+1}次重试: {e}")
            except Exception as e:
                logger.error(f"嵌入请求异常: {e}")
                raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if self.performance_stats["total_requests"] > 0:
            self.performance_stats["avg_time_per_request"] = \
                self.performance_stats["total_time"] / self.performance_stats["total_requests"]
        
        cache_stats = self.cache_manager.get_stats() if self.cache_manager else {}
        
        return {
            **self.performance_stats,
            **cache_stats,
            "model": self.model,
            "dimension": self.dimension,
            "batch_size": self.batch_size
        }

class TextProcessor:
    """文本预处理器"""

    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本"""
        import re

        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)

        # 移除控制字符
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        return text.strip()

    @staticmethod
    def split_long_text(
        text: str,
        max_tokens: int = 512,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: Optional[int] = None
    ) -> List[str]:
        """分割过长文本基于token数量"""
        if max_length is not None:
            if len(text) <= max_length:
                return [text]
            return [text[i:i + max_length] for i in range(0, len(text), max_length)]

        token_count = TextProcessor._count_tokens(text, tokenizer)
        if token_count < max_tokens:  # 使用 < 而非 <= 以确保安全
            return [text]

        if tokenizer:
            try:
                tokens = tokenizer.encode(text)
                decode_func = tokenizer.decode
                len(tokens)
            except Exception:
                tokenizer = None

        if not tokenizer:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                tokens = encoding.encode(text)
                decode_func = encoding.decode
            except Exception:
                char_limit = max(1, max_tokens * 2)
                return [text[i:i + char_limit] for i in range(0, len(text), char_limit)]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens - 2, len(tokens))  # 留出余量以避免特殊 token 问题
            chunk_tokens = tokens[start:end]
            chunk = decode_func(chunk_tokens).strip()
            chunk_token_count = TextProcessor._count_tokens(chunk, tokenizer)
            if chunk and chunk_token_count < max_tokens:
                chunks.append(chunk)
                logger.debug(f"生成子文本: {chunk[:50]}..., token 数: {chunk_token_count}")
            else:
                if chunk_token_count >= max_tokens:
                    logger.warning(f"分割子文本超长: {chunk_token_count} tokens, 预览: {chunk[:50]}...")
            start = end

        return chunks

    @staticmethod
    def _count_tokens(text: str, tokenizer: Optional[AutoTokenizer] = None) -> int:
        """估算token数量"""
        if not text:
            return 0
        if tokenizer:
            try:
                return len(tokenizer.encode(text))
            except Exception:
                pass
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            return max(1, len(text) // 2)


class EmbeddingManager:
    """嵌入管理器 - 统一接口"""

    def __init__(self, provider: str = "siliconflow", **kwargs):
        self.provider = provider

        if provider == "siliconflow":
            self.embedder = SiliconFlowEmbedder(**kwargs)
        else:
            raise ValueError(f"不支持的嵌入提供商: {provider}")

        self.text_processor = TextProcessor()
        logger.info(f"嵌入管理器初始化: provider={provider}")

    def embed_documents(self, documents: List[str]) -> EmbeddingResult:
        """嵌入文档列表"""
        # 预处理文本
        processed_texts = []
        for text in documents:
            cleaned = self.text_processor.clean_text(text)
            if not cleaned:
                continue
            # 分割过长文本，使用小于最大限制
            max_tokens = getattr(self.embedder, "max_input_tokens", 512)
            if not isinstance(max_tokens, int):
                max_tokens = 512
            chunks = self.text_processor.split_long_text(
                cleaned,
                max_tokens=max_tokens,
                tokenizer=self.embedder.tokenizer
            )
            processed_texts.extend(chunks)

        logger.info(f"文档预处理完成: {len(documents)} -> {len(processed_texts)} 个文本块")

        return self.embedder.embed_texts(processed_texts)

    def embed_query(self, query: str) -> List[float]:
        """嵌入查询文本"""
        cleaned_query = self.text_processor.clean_text(query)
        if not cleaned_query:
            return []
        # 如果查询过长，也分割，但对于查询通常短，嵌入第一个chunk
        max_tokens = getattr(self.embedder, "max_input_tokens", 512)
        if not isinstance(max_tokens, int):
            max_tokens = 512
        chunks = self.text_processor.split_long_text(
            cleaned_query,
            max_tokens=max_tokens,
            tokenizer=self.embedder.tokenizer
        )
        return self.embedder.embed_single_text(chunks[0])

    async def embed_documents_async(self, documents: List[str]) -> EmbeddingResult:
        """异步嵌入文档列表"""
        processed_texts = []
        for text in documents:
            cleaned = self.text_processor.clean_text(text)
            if not cleaned:
                continue
            max_tokens = getattr(self.embedder, "max_input_tokens", 512)
            if not isinstance(max_tokens, int):
                max_tokens = 512
            chunks = self.text_processor.split_long_text(
                cleaned,
                max_tokens=max_tokens,
                tokenizer=self.embedder.tokenizer
            )
            processed_texts.extend(chunks)

        return await self.embedder.embed_texts_async(processed_texts)

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """计算余弦相似度"""
        arr1 = np.array(embedding1)
        arr2 = np.array(embedding2)

        # 计算余弦相似度
        dot_product = np.dot(arr1, arr2)
        norm_a = np.linalg.norm(arr1)
        norm_b = np.linalg.norm(arr2)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    @property
    def dimension(self) -> int:
        """获取嵌入维度"""
        return self.embedder.dimension
