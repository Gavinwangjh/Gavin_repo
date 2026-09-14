import hashlib
import pickle
import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
from dataclasses import dataclass
import threading


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    memory_size: int = 0
    disk_size: int = 0
    total_requests: int = 0


class EmbeddingCacheManager:
    """嵌入式向量缓存管理器"""
    
    def __init__(self, cache_dir: str = "data/embedding_cache", max_memory_size: int = 10000):
        self.cache_dir = cache_dir
        self.max_memory_size = max_memory_size
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 内存缓存（LRU实现）
        self.memory_cache = {}
        self.cache_order = []  # 用于LRU策略
        
        # 磁盘缓存文件
        self.disk_cache_file = os.path.join(cache_dir, "embeddings_cache.pkl")
        self.stats_file = os.path.join(cache_dir, "cache_stats.json")
        
        # 统计信息
        self.stats = CacheStats()
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 加载现有缓存
        self._load_cache()
        self._load_stats()
        
        logger.info(f"嵌入缓存管理器初始化: 内存缓存={len(self.memory_cache)}，磁盘缓存={self._get_disk_cache_size()}")
    
    def _get_cache_key(self, model: str, text: str) -> str:
        """生成缓存键"""
        # 使用模型名称和文本内容的哈希
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return f"{model}:{content_hash}"
    
    def get(self, model: str, text: str) -> Optional[List[float]]:
        """获取缓存的嵌入向量"""
        with self.lock:
            self.stats.total_requests += 1
            cache_key = self._get_cache_key(model, text)
            
            # 1. 检查内存缓存
            if cache_key in self.memory_cache:
                self.stats.hits += 1
                # 更新访问顺序（LRU）
                if cache_key in self.cache_order:
                    self.cache_order.remove(cache_key)
                self.cache_order.append(cache_key)
                return self.memory_cache[cache_key]
            
            # 2. 检查磁盘缓存
            if hasattr(self, 'disk_cache') and cache_key in self.disk_cache:
                self.stats.hits += 1
                # 放入内存缓存
                embedding = self.disk_cache[cache_key]
                self._add_to_memory_cache(cache_key, embedding)
                return embedding
            
            self.stats.misses += 1
            return None
    
    def set(self, model: str, text: str, embedding: List[float]):
        """设置缓存"""
        with self.lock:
            cache_key = self._get_cache_key(model, text)
            
            # 添加到内存缓存
            self._add_to_memory_cache(cache_key, embedding)
            
            # 添加到磁盘缓存
            if hasattr(self, 'disk_cache'):
                self.disk_cache[cache_key] = embedding
            
            # 定期保存
            if self.stats.total_requests % 50 == 0:
                self._save_cache()
                self._save_stats()
    
    def _add_to_memory_cache(self, cache_key: str, embedding: List[float]):
        """添加到内存缓存（LRU策略）"""
        if cache_key not in self.memory_cache:
            # 检查是否超过最大大小
            if len(self.memory_cache) >= self.max_memory_size:
                # 移除最久未使用的
                oldest_key = self.cache_order.pop(0)
                del self.memory_cache[oldest_key]
        
        self.memory_cache[cache_key] = embedding
        if cache_key in self.cache_order:
            self.cache_order.remove(cache_key)
        self.cache_order.append(cache_key)
    
    def _load_cache(self):
        """加载磁盘缓存"""
        self.disk_cache = {}
        if os.path.exists(self.disk_cache_file):
            try:
                with open(self.disk_cache_file, 'rb') as f:
                    self.disk_cache = pickle.load(f)
                logger.info(f"加载磁盘缓存: {len(self.disk_cache)} 个条目")
            except Exception as e:
                logger.warning(f"加载磁盘缓存失败: {e}")
    
    def _save_cache(self):
        """保存缓存到磁盘"""
        try:
            with open(self.disk_cache_file, 'wb') as f:
                pickle.dump(self.disk_cache, f)
            logger.debug(f"缓存已保存: {len(self.disk_cache)} 个条目")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    def _load_stats(self):
        """加载统计信息"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    stats_data = json.load(f)
                    self.stats = CacheStats(**stats_data)
            except Exception as e:
                logger.warning(f"加载统计信息失败: {e}")
    
    def _save_stats(self):
        """保存统计信息"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats.__dict__, f, indent=2)
        except Exception as e:
            logger.error(f"保存统计信息失败: {e}")
    
    def _get_disk_cache_size(self) -> int:
        """获取磁盘缓存大小"""
        return len(self.disk_cache) if hasattr(self, 'disk_cache') else 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        hit_rate = self.stats.hits / max(self.stats.total_requests, 1)
        
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "total_requests": self.stats.total_requests,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_size": self._get_disk_cache_size(),
            "max_memory_size": self.max_memory_size
        }
    
    def clear_cache(self, clear_disk: bool = False):
        """清空缓存"""
        with self.lock:
            self.memory_cache.clear()
            self.cache_order.clear()
            
            if clear_disk:
                self.disk_cache.clear()
                self._save_cache()
            
            logger.info("缓存已清空")