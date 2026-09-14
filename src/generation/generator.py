# src/generation/generator.py
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator
from dataclasses import dataclass
import httpx
import asyncio
import json
import os
import time
import re
from enum import Enum

from loguru import logger
from src.utils.helpers import retry_on_failure, PerformanceTimer, get_config
from src.retrieval.retriever import RetrievalResult, SearchMethod
from src.vector_store.milvus_store import SearchHit


@dataclass
class GenerationResult:
    """生成结果数据结构"""
    question: str
    answer: str
    sources: List[SearchHit]
    retrieval_result: Optional[RetrievalResult] = None
    generation_time: float = 0.0
    total_time: float = 0.0
    model: str = ""
    token_usage: Optional[Dict[str, int]] = None
    confidence: float = 0.0  # 置信度分数


@dataclass
class ChatMessage:
    """聊天消息数据结构"""
    role: str  # system, user, assistant
    content: str


class GenerationStrategy(str, Enum):
    """生成策略枚举"""
    RAG = "rag"           # 基于检索的生成
    CHAT = "chat"         # 纯聊天
    HYBRID = "hybrid"     # 混合模式
    EXTRACTIVE = "extractive"  # 抽取式回答


class KimiLLMClient:
    """Kimi (月之暗面) LLM客户端"""
    
    DEFAULT_MODEL = "kimi-k3"
    DEFAULT_API_BASE = "https://api.moonshot.ai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        api_base: str = DEFAULT_API_BASE,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        max_retries: int = 3,
        timeout: int = 60
    ):
        # 方法1：直接导入并使用你项目中正确的get_config方式
        # 如果你的get_config()函数需要参数，这里是最关键的地方
        try:
            # 尝试无参数调用（如果你的get_config支持）
            config = get_config()
        except TypeError:
            # 如果需要参数，可能是这样调用：
            # config = get_config()  # 或者需要具体路径
            # 查看你的helpers.py中get_config函数的定义
            # 常见的方式是：get_config()返回整个配置
            config = {}
        
        # 设置API密钥
        if api_key:
            self.api_key = api_key
        else:
            # 从配置文件中读取
            if isinstance(config, dict):
                api_keys = config.get("api_keys", {})
                self.api_key = (
                    api_keys.get("moonshot_api_key")
                    or api_keys.get("kimi_api_key")
                    or os.getenv("MOONSHOT_API_KEY")
                    or os.getenv("KIMI_API_KEY")
                    or ""
                )
            else:
                self.api_key = ""
        
        # 如果还是没有API密钥，使用硬编码（从你的配置文件）
        if not self.api_key:
            self.api_key = (
                get_config("api_keys.moonshot_api_key", "")
                or get_config("api_keys.kimi_api_key", "")
                or os.getenv("MOONSHOT_API_KEY")
                or os.getenv("KIMI_API_KEY")
                or ""
            )
        
        # 设置其他参数
        if model != self.DEFAULT_MODEL:
            self.model = model
        elif isinstance(config, dict):
            self.model = config.get("llm", {}).get("model", self.DEFAULT_MODEL)
        else:
            self.model = self.DEFAULT_MODEL
        
        self.model = self.model or get_config("llm.model", self.DEFAULT_MODEL)
        if api_base != self.DEFAULT_API_BASE:
            resolved_api_base = api_base
        elif isinstance(config, dict):
            resolved_api_base = config.get("llm", {}).get("api_base", self.DEFAULT_API_BASE)
        else:
            resolved_api_base = get_config("llm.api_base", self.DEFAULT_API_BASE)
        self.api_base = (resolved_api_base or self.DEFAULT_API_BASE).rstrip('/')
        self.max_tokens = max_tokens or get_config("llm.max_tokens", 4096)
        self.temperature = temperature if temperature is not None else get_config("llm.temperature", 0.7)
        self.max_retries = max_retries or get_config("llm.max_retries", 3)
        self.timeout = timeout
        
        # 验证API密钥
        if not self.api_key:
            raise ValueError("Kimi API密钥未设置")
        
        # 设置请求头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        logger.info(f"Kimi LLM客户端初始化: model={self.model}")
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def generate(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs
    ) -> str:
        """生成文本回复"""
        try:
            # 准备请求数据
            payload = {
                "model": self.model,
                "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "stream": stream
            }
            
            # 移除None值
            payload = {k: v for k, v in payload.items() if v is not None}
            
            # 发送请求
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                
                # 提取回复内容
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                    return content.strip()
                else:
                    raise ValueError("API返回格式异常")
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"Kimi API HTTP错误: {e.response.status_code} - {e.response.text}")
            raise Exception(f"LLM API调用失败: HTTP {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Kimi API请求错误: {str(e)}")
            raise Exception(f"LLM API请求失败: {str(e)}")
        except Exception as e:
            logger.error(f"LLM生成错误: {str(e)}")
            raise
    
    def generate_stream(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> str:
        """流式生成文本（简化版）"""
        # 对于非流式场景，直接调用generate
        return self.generate(messages, stream=False, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取LLM使用统计"""
        return {
            "model": self.model,
            "api_base": self.api_base
        }


class PromptTemplate:
    """增强版提示词模板管理器"""
    
    # 系统提示词 - 针对RAG优化
    RAG_SYSTEM_PROMPT = """你是一个专业、准确的信息提取助手，专门处理基于文档的问答。

请严格遵守以下规则：
1. 严格基于提供的文档内容进行回答，绝不编造、猜测或添加文档中没有的信息
2. 如果文档中没有相关信息，请明确说明"文档中没有相关信息"
3. 引用要具体：回答时应明确说明信息来自哪个文档片段
4. 保持客观：直接引用文档原文，不添加个人观点
5. 结构化回答：对于条文、列表等结构化内容，保持原有格式

重要说明：
- 你收到的上下文是从检索系统中提取的最相关文档片段
- 请根据这些片段回答用户问题
- 如果多个片段提供的信息有冲突，以最先出现的或更权威的片段为准
- 法律条文、规章制度等必须准确引用"""

    # 法律条文专用系统提示词
    LAW_SYSTEM_PROMPT = """你是专业的法律条文解释助手，专门处理法律、法规、政策等结构化文档。

请遵循以下原则：
1. 严格忠于条文原文，一字不差地引用关键条款
2. 明确指出条文编号（如第一条、第一款、（一）等）
3. 对于模糊查询（如"第一条"），确认是否指代当前文档的第一条
4. 如果涉及多个条款，按逻辑顺序组织回答
5. 避免解释或演绎，直接引用原文"""

    POLICY_SYSTEM_PROMPT = """你是专业的政策、制度和规范性文件问答助手，适用于政府政策、企业制度、办事指南、申报通知和法律法规文件。

请遵循以下原则：
1. 严格依据检索到的文件片段回答，不编造未出现的政策条件、金额、期限或责任主体。
2. 优先识别并回答：适用对象、申报/申请条件、材料、流程、期限、扶持/处罚标准、责任部门、监督管理、附则。
3. 如果问题涉及具体条款、章节、栏目或编号，必须优先引用对应编号或栏目。
4. 如果多个文件片段存在差异，应说明来源片段，并提示以文件原文或更新文件为准。
5. 无依据时明确回答“检索到的文件片段中没有相关依据”。
"""

    # RAG回答模板 - 强制引用格式
    RAG_TEMPLATE = """请基于以下文档片段回答用户的问题。

重要提示：
1. 必须基于提供的片段回答
2. 必须注明信息来自哪个片段
3. 必须保持原文的准确性

文档片段：
{context}

用户问题：{question}

请按照以下格式回答：
【答案】
[你的回答]

【引用来源】
[列出使用的片段编号和关键内容]

【注意事项】
[如有需要，补充说明]"""

    # 简洁版RAG模板（用于简单问答）
    RAG_SIMPLE_TEMPLATE = """基于以下文档回答问题：

{context}

问题：{question}

回答："""

    # 无相关文档模板
    NO_CONTEXT_TEMPLATE = """我没有找到与您的问题相关的文档内容。

您的问题：{question}

可能的原因：
1. 相关文档尚未收录到知识库中
2. 问题表述与文档内容不完全匹配
3. 检索系统未能找到最相关的内容

建议：
1. 尝试使用更具体的关键词
2. 确认问题涉及的内容是否在已上传的文档范围内
3. 检查是否有相关文档需要上传"""

    @classmethod
    def build_rag_prompt(
        cls, 
        question: str, 
        context_chunks: List[SearchHit],
        strategy: str = "standard"
    ) -> List[ChatMessage]:
        """构建RAG提示词 - 根据策略选择模板"""
        if not context_chunks:
            return cls._build_no_context_prompt(question)
        
        # 检测问题类型
        question_type = cls._detect_question_type(question)
        
        # 选择系统提示词
        system_prompt = cls.RAG_SYSTEM_PROMPT
        if question_type == "law":
            system_prompt = cls.LAW_SYSTEM_PROMPT
        elif question_type == "policy":
            system_prompt = cls.POLICY_SYSTEM_PROMPT
        
        # 构建上下文
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            # 增强的片段标识
            identifier = f"[片段{i}]"
            
            # 添加元数据信息
            metadata_info = []
            if chunk.metadata:
                if 'article_number' in chunk.metadata:
                    metadata_info.append(f"条文：{chunk.metadata['article_number']}")
                if 'title' in chunk.metadata:
                    metadata_info.append(f"标题：{chunk.metadata['title']}")
                if 'filename' in chunk.metadata:
                    metadata_info.append(f"文件：{chunk.metadata['filename']}")
            
            if metadata_info:
                identifier += f" ({' | '.join(metadata_info)})"
            
            context_parts.append(f"{identifier}\n{chunk.content}")
        
        context = "\n\n".join(context_parts)
        
        # 选择用户模板
        if strategy == "simple":
            user_content = cls.RAG_SIMPLE_TEMPLATE.format(
                context=context,
                question=question
            )
        else:
            user_content = cls.RAG_TEMPLATE.format(
                context=context,
                question=question
            )
        
        return [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_content)
        ]
    
    @classmethod
    def _build_no_context_prompt(cls, question: str) -> List[ChatMessage]:
        """构建无相关文档的提示词"""
        return [
            ChatMessage(role="system", content=cls.RAG_SYSTEM_PROMPT),
            ChatMessage(role="user", content=cls.NO_CONTEXT_TEMPLATE.format(question=question))
        ]
    
    @classmethod
    def build_chat_prompt(
        cls,
        question: str,
        chat_history: Optional[List[ChatMessage]] = None
    ) -> List[ChatMessage]:
        """Build a plain chat prompt, keeping backward compatibility."""
        messages = [
            ChatMessage(role="system", content="你是一个准确、简洁的中文助手。")
        ]
        messages.extend(chat_history or [])
        messages.append(ChatMessage(role="user", content=question))
        return messages

    @classmethod
    def _detect_question_type(cls, question: str) -> str:
        """检测问题类型"""
        law_patterns = ['第.*条', '第.*款', '第.*项', '（.*）', '法规', '法律', '条例', '规定']
        for pattern in law_patterns:
            if re.search(pattern, question):
                return "law"
        policy_terms = [
            "政策", "制度", "办法", "通知", "指南", "申报", "申请", "补贴", "扶持",
            "奖励", "资助", "认定", "备案", "公示", "流程", "材料", "条件",
            "适用范围", "责任部门", "监督管理", "有效期", "截止时间",
        ]
        if any(term in question for term in policy_terms):
            return "policy"
        return "general"


class EnhancedRAGGenerator:
    def __init__(
        self,
        llm_client=None,
        config=None,
        max_context_length: Optional[int] = None,
        min_similarity_score: Optional[float] = None,
        **kwargs
    ):
        self.llm_client = llm_client or KimiLLMClient()
        
        # 核心修复：直接设置分数阈值为0.01
        self.min_similarity_score = min_similarity_score if min_similarity_score is not None else 0.01
        self.max_context_length = max_context_length if max_context_length is not None else 8000
        self.default_strategy = "standard"
        self.confidence_threshold = 0.5
        
        # 添加缺失的性能统计属性
        self.total_generations = 0
        self.total_successful = 0
        
        logger.info(f"增强版RAG生成器初始化完成")
        logger.info(f"  最小相似度阈值: {self.min_similarity_score}")
    
    def generate_answer(
        self,
        question: str,
        retrieval_result: RetrievalResult,
        strategy: str = None,
        stream: bool = False,
        **kwargs
    ) -> GenerationResult:
        """生成RAG回答 - 增强版，解决分数阈值问题"""
        total_start_time = time.time()
        strategy = strategy or self.default_strategy
        
        try:
            # 详细调试信息
            logger.info("=" * 60)
            logger.info(f"开始生成回答")
            logger.info(f"  问题: '{question}'")
            logger.info(f"  检索方法: {retrieval_result.method}")
            logger.info(f"  策略: {strategy}")
            
            # 1. 处理检索结果 - 针对不同算法调整过滤策略
            processed_chunks = self._process_retrieval_results(
                question, retrieval_result, strategy
            )
            
            if not processed_chunks:
                logger.warning("没有通过过滤的文档片段")
                return self._generate_no_context_response(question, total_start_time)
            
            # 2. 限制上下文长度
            filtered_chunks = self._filter_chunks_by_length(processed_chunks)
            
            # 3. 计算置信度
            confidence = self._calculate_confidence(
                filtered_chunks, retrieval_result.method
            )
            
            # 4. 构建提示词
            messages = PromptTemplate.build_rag_prompt(
                question, filtered_chunks, strategy
            )
            
            # 5. 生成回答
            generation_start = time.time()
            
            try:
                if stream:
                    answer_parts = []
                    for chunk in self.llm_client.generate_stream(messages, **kwargs):
                        answer_parts.append(chunk)
                    answer = "".join(answer_parts)
                else:
                    answer = self.llm_client.generate(messages, **kwargs)
                
                generation_time = max(time.time() - generation_start, 1e-9)
                self.total_successful += 1
                
            except Exception as e:
                logger.error(f"LLM生成失败: {str(e)}")
                answer = f"抱歉，生成回答时出现错误: {str(e)}"
                generation_time = 0.0
                confidence = 0.0
            
            total_time = time.time() - total_start_time
            self.total_generations += 1
            
            # 6. 后处理答案
            processed_answer = self._postprocess_answer(
                answer, question, filtered_chunks
            )
            
            result = GenerationResult(
                question=question,
                answer=processed_answer,
                sources=filtered_chunks,
                retrieval_result=retrieval_result,
                generation_time=generation_time,
                total_time=total_time,
                model=self.llm_client.model,
                confidence=confidence
            )
            
            logger.info(f"生成完成")
            logger.info(f"  置信度: {confidence:.2f}")
            logger.info(f"  生成时间: {generation_time:.2f}s")
            logger.info(f"  总时间: {total_time:.2f}s")
            logger.info(f"  使用片段数: {len(filtered_chunks)}")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            logger.error(f"生成回答失败: {str(e)}")
            return GenerationResult(
                question=question,
                answer=f"抱歉，生成回答时出现错误: {str(e)}",
                sources=[],
                retrieval_result=retrieval_result,
                generation_time=0.0,
                total_time=time.time() - total_start_time,
                model=self.llm_client.model,
                confidence=0.0
            )
    
    def _process_retrieval_results(
        self,
        question: str,
        retrieval_result: RetrievalResult,
        strategy: str
    ) -> List[SearchHit]:
        """处理检索结果 - 针对不同算法调整过滤"""
        
        if not retrieval_result.hits:
            logger.debug("检索结果为空")
            return []
        
        # 详细日志输出
        logger.debug("=== 检索结果处理 ===")
        logger.debug(f"原始结果数: {len(retrieval_result.hits)}")
        
        for i, hit in enumerate(retrieval_result.hits[:5]):
            logger.debug(f"  原始结果{i+1}: score={hit.score:.6f}")
            if hit.metadata:
                logger.debug(f"      元数据: {hit.metadata.get('article_number', '无条文号')}")
        
        # 针对RRF算法的特殊处理
        if retrieval_result.method == SearchMethod.HYBRID_RRF:
            # RRF分数范围是[0, 0.033]，需要降低阈值
            dynamic_threshold = self.min_similarity_score
            
            logger.debug(f"RRF算法，使用动态阈值: {dynamic_threshold}")
            
            relevant_chunks = [
                hit for hit in retrieval_result.hits
                if hit.score >= dynamic_threshold
            ]
            
        elif retrieval_result.method == SearchMethod.HYBRID:
            # 加权融合算法，分数范围可能不同
            dynamic_threshold = max(self.min_similarity_score, 0.3)
            relevant_chunks = [
                hit for hit in retrieval_result.hits
                if hit.score >= dynamic_threshold
            ]
            
        else:
            # 其他算法使用默认阈值
            relevant_chunks = [
                hit for hit in retrieval_result.hits
                if hit.score >= self.min_similarity_score
            ]
        
        logger.debug(f"过滤后结果数: {len(relevant_chunks)}")
        
        # 如果分数都很低但非零，至少保留一个
        if not relevant_chunks and retrieval_result.hits:
            logger.warning("所有结果分数都低于阈值，保留最高分结果")
            max_score_hit = max(retrieval_result.hits, key=lambda x: x.score)
            if max_score_hit.score > 0:
                relevant_chunks = [max_score_hit]
                logger.debug(f"保留最高分结果: score={max_score_hit.score:.6f}")
        
        # 结构化boost后处理
        relevant_chunks = self._apply_post_retrieval_boost(
            question, relevant_chunks
        )
        
        return relevant_chunks
    
    def _apply_post_retrieval_boost(
        self, 
        question: str, 
        chunks: List[SearchHit]
    ) -> List[SearchHit]:
        """应用后检索boost，增强结构化信息"""
        if not chunks:
            return chunks
        
        # 提取问题中的条文编号
        article_patterns = [
            r'第[一二三四五六七八九十\d]+条',
            r'第[一二三四五六七八九十\d]+款',
            r'第[一二三四五六七八九十\d]+项',
        ]
        
        query_articles = []
        for pattern in article_patterns:
            matches = re.findall(pattern, question)
            query_articles.extend(matches)
        
        if not query_articles:
            return chunks
        
        boosted_chunks = []
        for chunk in chunks:
            boost_factor = 1.0
            
            # 检查元数据中的条文编号
            if chunk.metadata and 'article_number' in chunk.metadata:
                chunk_article = chunk.metadata['article_number']
                
                # 精确匹配
                for query_article in query_articles:
                    if query_article == chunk_article:
                        boost_factor = 2.0  # 精确匹配高boost
                        logger.debug(f"精确条文匹配: {query_article} -> {chunk_article}, boost=2.0")
                        break
                    elif query_article in chunk_article:
                        boost_factor = 1.5  # 部分匹配
                        logger.debug(f"部分条文匹配: {query_article} -> {chunk_article}, boost=1.5")
                        break
            
            # 应用boost
            boosted_hit = SearchHit(
                id=chunk.id,
                score=chunk.score * boost_factor,
                content=chunk.content,
                metadata=chunk.metadata,
                doc_id=chunk.doc_id,
                chunk_index=chunk.chunk_index
            )
            boosted_chunks.append(boosted_hit)
        
        # 重新排序
        boosted_chunks.sort(key=lambda x: x.score, reverse=True)
        return boosted_chunks
    
    def _filter_chunks_by_length(self, chunks: List[SearchHit]) -> List[SearchHit]:
        """智能上下文长度过滤"""
        if not chunks:
            return []
        
        filtered_chunks = []
        total_length = 0
        
        # 优先保留高分片段
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
        
        for chunk in sorted_chunks:
            chunk_length = len(chunk.content)
            
            # 如果还有足够空间
            if total_length + chunk_length <= self.max_context_length:
                filtered_chunks.append(chunk)
                total_length += chunk_length
            else:
                # 计算剩余空间
                remaining = self.max_context_length - total_length
                
                # 如果剩余空间大于最小可读长度（200字符），则截断
                if remaining > 200:
                    # 智能截断：尽量保留完整句子
                    truncated = self._smart_truncate(chunk.content, remaining)
                    if truncated:
                        truncated_chunk = SearchHit(
                            id=chunk.id,
                            score=chunk.score,
                            content=truncated,
                            metadata=chunk.metadata,
                            doc_id=chunk.doc_id,
                            chunk_index=chunk.chunk_index
                        )
                        filtered_chunks.append(truncated_chunk)
                break
        
        logger.debug(f"长度过滤: {len(chunks)} -> {len(filtered_chunks)} 个片段")
        logger.debug(f"总长度: {total_length} 字符")
        
        return filtered_chunks
    
    def _smart_truncate(self, text: str, max_length: int) -> str:
        """智能截断文本，保留完整句子"""
        if len(text) <= max_length:
            return text
        
        # 在max_length处找最近的句子结束符
        end_chars = ['。', '！', '？', '\n', '.', '!', '?']
        truncated = text[:max_length]
        
        # 向后找句子结束
        for i in range(min(100, len(text) - max_length)):
            pos = max_length + i
            if pos < len(text) and text[pos] in end_chars:
                return text[:pos+1]
        
        # 向前找句子结束
        for i in range(min(50, max_length)):
            pos = max_length - i
            if text[pos] in end_chars:
                return text[:pos+1]
        
        # 如果都找不到，在max_length-3处截断并加省略号
        return text[:max_length-3] + "..."
    
    def _calculate_confidence(
        self, 
        chunks: List[SearchHit], 
        method: SearchMethod
    ) -> float:
        """计算回答置信度"""
        if not chunks:
            return 0.0
        
        # 基础置信度：最高分数归一化
        max_score = max(chunk.score for chunk in chunks)
        
        if method == SearchMethod.HYBRID_RRF:
            # RRF分数范围[0, 0.033]，归一化到[0, 1]
            confidence = min(max_score * 30, 1.0)  # 0.033 * 30 ≈ 1.0
        else:
            # 其他方法，假设分数范围[0, 1]
            confidence = min(max_score, 1.0)
        
        # 考虑片段数量和质量
        num_factor = min(len(chunks) / 3, 1.0)  # 最多3个片段就够
        
        # 结构化信息加分
        structure_factor = 0.0
        for chunk in chunks:
            if chunk.metadata and 'article_number' in chunk.metadata:
                structure_factor += 0.1
        
        final_confidence = 0.7 * confidence + 0.2 * num_factor + 0.1 * min(structure_factor, 0.3)
        
        return round(final_confidence, 2)
    
    def _postprocess_answer(
        self, 
        answer: str, 
        question: str, 
        sources: List[SearchHit]
    ) -> str:
        """后处理答案，优化格式"""
        # 清理多余的空格和换行
        answer = re.sub(r'\n{3,}', '\n\n', answer.strip())
        
        # 如果答案以"抱歉"开头且很短，可能有问题
        if answer.startswith("抱歉") and len(answer) < 50:
            # 检查是否有结构化查询
            if re.search(r'第.*[条款项]', question):
                # 尝试给出更具体的提示
                answer += "\n\n提示：如果您在查询具体的法律条文，请确认条文编号是否准确，或者尝试使用完整的条文内容进行查询。"
        
        return answer
    
    def _generate_no_context_response(
        self, 
        question: str, 
        start_time: float
    ) -> GenerationResult:
        """生成无上下文时的响应"""
        messages = PromptTemplate._build_no_context_prompt(question)
        
        try:
            answer = self.llm_client.generate(messages)
        except Exception as e:
            answer = f"抱歉，无法生成回答。错误: {str(e)}"
        
        return GenerationResult(
            question=question,
            answer=answer,
            sources=[],
            retrieval_result=None,
            generation_time=0.0,
            total_time=time.time() - start_time,
            model=self.llm_client.model,
            confidence=0.0
        )

    def chat(
        self,
        question: str,
        chat_history: Optional[List[ChatMessage]] = None,
        **kwargs
    ) -> str:
        """Generate a plain chat response without retrieval."""
        messages = PromptTemplate.build_chat_prompt(question, chat_history)
        try:
            return self.llm_client.generate(messages, **kwargs)
        except Exception as e:
            return f"抱歉，聊天时出现错误: {str(e)}"
        messages = list(chat_history or [])
        if not messages or messages[0].role != "system":
            messages.insert(
                0,
                ChatMessage(
                    role="system",
                    content="你是一个准确、简洁的中文助手。"
                )
            )
        messages.append(ChatMessage(role="user", content=question))
        return self.llm_client.generate(messages, **kwargs)
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"更新配置: {key}={value}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取生成器统计信息"""
        llm_stats = self.llm_client.get_stats() if self.llm_client else {}
        
        return {
            "model": getattr(self.llm_client, "model", ""),
            "max_context_length": self.max_context_length,
            "min_similarity_score": self.min_similarity_score,
            "generator_config": {
                "min_similarity_score": self.min_similarity_score,
                "max_context_length": self.max_context_length,
                "default_strategy": self.default_strategy,
                "confidence_threshold": self.confidence_threshold
            },
            "performance": {
                "total_generations": self.total_generations,
                "successful_generations": self.total_successful,
                "success_rate": self.total_successful / max(self.total_generations, 1)
            },
            "llm_stats": llm_stats
        }


# 兼容旧版本
RAGGenerator = EnhancedRAGGenerator
