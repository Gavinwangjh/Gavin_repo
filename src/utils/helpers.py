import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import hashlib

from loguru import logger
from dotenv import load_dotenv


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self._config = None
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        load_dotenv()
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 处理环境变量替换
            config = self._resolve_env_vars(config)
            self._config = config
            
            logger.info(f"配置文件加载成功: {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"配置文件加载失败: {str(e)}")
            raise
    
    def _resolve_env_vars(self, obj: Any) -> Any:
        """递归解析环境变量"""
        if isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            default_value = None
            
            if ":" in env_var:
                env_var, default_value = env_var.split(":", 1)
            
            value = os.getenv(env_var, default_value)
            if isinstance(value, str):
                if value.lower() in {"true", "false"}:
                    return value.lower() == "true"
                if value.isdigit():
                    return int(value)
            return value
        else:
            return obj
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的嵌套键"""
        if self._config is None:
            self.load_config()
        
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        if self._config is None:
            self._config = {}
        
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    @property
    def config(self) -> Dict[str, Any]:
        """获取完整配置"""
        if self._config is None:
            self.load_config()
        return self._config


class Logger:
    """日志管理器"""
    
    @staticmethod
    def setup_logger(
        log_level: str = "INFO",
        log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        log_file: Optional[str] = None,
        rotation: str = "1 week",
        retention: str = "1 month"
    ):
        """设置日志配置"""
        logger.remove()  # 移除默认handler
        
        # 控制台输出
        logger.add(
            sink=lambda msg: print(msg, end=""),
            level=log_level,
            format=log_format,
            colorize=True
        )
        
        # 文件输出
        if log_file:
            logger.add(
                sink=log_file,
                level=log_level,
                format=log_format,
                rotation=rotation,
                retention=retention,
                encoding="utf-8"
            )
        
        logger.info("日志系统初始化完成")


def generate_id(content: str, prefix: str = "") -> str:
    """生成基于内容的唯一ID"""
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{timestamp}_{content_hash[:8]}" if prefix else f"{timestamp}_{content_hash[:8]}"


def safe_json_loads(text: str, default: Any = None) -> Any:
    """安全的JSON解析"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """安全的JSON序列化"""
    try:
        return json.dumps(obj, ensure_ascii=False, **kwargs)
    except (TypeError, ValueError) as e:
        logger.warning(f"JSON序列化失败: {str(e)}")
        return "{}"


def ensure_directory(path: Union[str, Path]) -> Path:
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """获取文件大小（MB）"""
    return Path(file_path).stat().st_size / (1024 * 1024)


def validate_file_size(file_path: Union[str, Path], max_size_mb: float = 10.0) -> bool:
    """验证文件大小"""
    return get_file_size_mb(file_path) <= max_size_mb


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_timestamp(timestamp: Optional[float] = None) -> str:
    """格式化时间戳"""
    if timestamp is None:
        timestamp = datetime.now().timestamp()
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def clean_filename(filename: str) -> str:
    """清理文件名，移除不安全字符"""
    import re
    # 移除或替换不安全字符
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 移除前后空格和点号
    cleaned = cleaned.strip('. ')
    return cleaned or "untitled"


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """深度合并字典"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


class PerformanceTimer:
    """性能计时器"""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.debug(f"{self.name} 开始计时")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"{self.name} 耗时: {duration:.2f}秒")
    
    @property
    def duration(self) -> Optional[float]:
        """获取耗时（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""
    import time
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"{func.__name__} 第{attempt + 1}次尝试失败: {str(e)}, {delay}秒后重试")
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} 所有重试均失败")
            
            raise last_exception
        
        return wrapper
    return decorator


# 全局配置管理器实例
config_manager = ConfigManager()


def get_config(key: str, default: Any = None) -> Any:
    """获取配置的便捷函数"""
    return config_manager.get(key, default)
