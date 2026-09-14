# src/rerank/reranker.py
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from src.vector_store.milvus_store import SearchHit
from loguru import logger
import time
import re
import json
import hashlib
from collections import defaultdict


@dataclass
class RerankResult:
    """重排序结果"""
    query: str
    original_hits: List[SearchHit]
    reranked_hits: List[SearchHit]
    reranker_name: str
    rerank_time: float
    metrics: Dict[str, Any] = field(default_factory=dict)


class BaseReranker:
    """重排序器基类"""
    
    def __init__(self, name: str = "base"):
        self.name = name
        self.total_calls = 0
        self.total_time = 0.0
    
    def rerank(self, query: str, hits: List[SearchHit], top_k: Optional[int] = None) -> List[SearchHit]:
        """重排序接口"""
        raise NotImplementedError
    
    def rerank_with_metrics(self, query: str, hits: List[SearchHit], top_k: Optional[int] = None) -> RerankResult:
        """重排序并返回带指标的结果"""
        start_time = time.time()
        reranked_hits = self.rerank(query, hits, top_k)
        rerank_time = time.time() - start_time
        
        # 更新统计
        self.total_calls += 1
        self.total_time += rerank_time
        
        # 计算指标
        metrics = self._calculate_rerank_metrics(hits, reranked_hits)
        metrics.update({
            "rerank_time": rerank_time,
            "total_calls": self.total_calls,
            "avg_rerank_time": self.total_time / max(self.total_calls, 1)
        })
        
        return RerankResult(
            query=query,
            original_hits=hits,
            reranked_hits=reranked_hits,
            reranker_name=self.name,
            rerank_time=rerank_time,
            metrics=metrics
        )
    
    def _calculate_rerank_metrics(self, original: List[SearchHit], reranked: List[SearchHit]) -> Dict[str, Any]:
        """计算重排序指标"""
        if not original or not reranked:
            return {}
        
        # 计算排名变化
        rank_changes = []
        for i, hit in enumerate(reranked):
            try:
                original_idx = original.index(hit)
                rank_changes.append(original_idx - i)  # 正数表示提升
            except ValueError:
                rank_changes.append(len(original))  # 如果不在原列表中
        
        # 计算分数提升
        score_improvements = []
        for hit in reranked:
            for orig_hit in original:
                if hit.id == orig_hit.id:
                    improvement = hit.score - orig_hit.score
                    score_improvements.append(improvement)
                    break
        
        return {
            "num_hits": len(reranked),
            "avg_rank_change": sum(rank_changes) / len(rank_changes) if rank_changes else 0,
            "max_rank_change": max(rank_changes) if rank_changes else 0,
            "avg_score_improvement": sum(score_improvements) / len(score_improvements) if score_improvements else 0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "total_time": self.total_time,
            "avg_time_per_call": self.total_time / max(self.total_calls, 1)
        }


class SimpleScoreReranker(BaseReranker):
    """简单分数重排序器 - 按分数排序"""
    
    def __init__(self):
        super().__init__("simple_score")
    
    def rerank(self, query: str, hits: List[SearchHit], top_k: Optional[int] = None) -> List[SearchHit]:
        """按分数降序排序"""
        if not hits:
            return []
        
        start_time = time.time()
        sorted_hits = sorted(hits, key=lambda x: x.score, reverse=True)
        rerank_time = time.time() - start_time
        
        logger.debug(f"简单重排序完成，耗时: {rerank_time:.3f}s，处理 {len(hits)} 个文档")
        
        if top_k:
            return sorted_hits[:top_k]
        return sorted_hits


class RuleBasedReranker(BaseReranker):
    """基于规则的重排序器 - 适用于法律文档"""
    
    def __init__(self):
        super().__init__("rule_based")
        # 正则模式匹配
        self.article_patterns = [
            r'第[一二三四五六七八九十\d]+条',
            r'第[一二三四五六七八九十\d]+款',
            r'第[一二三四五六七八九十\d]+项',
            r'（[一二三四五六七八九十\d]+）',
        ]
    
    def rerank(self, query: str, hits: List[SearchHit], top_k: Optional[int] = None) -> List[SearchHit]:
        """基于规则重排序"""
        if not hits:
            return []
        
        start_time = time.time()
        
        # 提取查询中的条文编号
        query_articles = []
        for pattern in self.article_patterns:
            matches = re.findall(pattern, query)
            query_articles.extend(matches)
        
        # 如果没有条文编号，直接返回原始顺序
        if not query_articles:
            logger.debug("查询中未检测到条文编号，保持原顺序")
            return hits[:top_k] if top_k else hits
        
        # 计算每个文档的匹配分数
        scored_hits = []
        for hit in hits:
            score = self._calculate_rule_score(query, query_articles, hit)
            hit.rule_score = score
            hit.combined_score = 0.5 * hit.score + 0.5 * score
            scored_hits.append(hit)
        
        # 按组合分数排序
        sorted_hits = sorted(scored_hits, key=lambda x: x.combined_score, reverse=True)
        
        rerank_time = time.time() - start_time
        logger.info(f"规则重排序完成，耗时: {rerank_time:.3f}s，匹配到 {len(query_articles)} 个条文编号")
        
        if top_k:
            return sorted_hits[:top_k]
        return sorted_hits
    
    def _calculate_rule_score(self, query: str, query_articles: List[str], hit: SearchHit) -> float:
        """计算规则分数"""
        score = 0.0
        
        # 检查元数据中的条文编号
        if hit.metadata and 'article_number' in hit.metadata:
            chunk_article = hit.metadata['article_number']
            
            for query_article in query_articles:
                if query_article == chunk_article:
                    score += 2.0  # 精确匹配
                elif query_article in chunk_article:
                    score += 1.0  # 包含关系
                elif chunk_article in query_article:
                    score += 0.5  # 反向包含
        
        # 检查内容中的条文编号
        for query_article in query_articles:
            if query_article in hit.content:
                score += 0.3
        
        return score


class PolicyDocumentReranker(BaseReranker):
    """Rule-based reranker for legal, government policy, and enterprise policy documents."""

    def __init__(self):
        super().__init__("policy_document")
        self.structure_patterns = [
            r'第[一二三四五六七八九十百千万\d]+[条章节款项]',
            r'[一二三四五六七八九十]+、',
            r'（[一二三四五六七八九十\d]+）',
            r'\d+[.、]',
        ]
        self.policy_terms = {
            "适用范围", "申报条件", "申请条件", "支持对象", "扶持对象", "补贴对象",
            "扶持标准", "补助标准", "奖励标准", "资助标准", "认定标准",
            "申报材料", "申请材料", "办理流程", "申报流程", "审批流程",
            "责任部门", "主管部门", "牵头部门", "职责分工", "监督管理",
            "实施期限", "有效期", "截止时间", "附则", "解释权",
            "处罚", "违规", "考核", "验收", "公示", "备案",
        }

    def rerank(self, query: str, hits: List[SearchHit], top_k: Optional[int] = None) -> List[SearchHit]:
        if not hits:
            return []

        query_markers = self._extract_structure_markers(query)
        query_terms = {term for term in self.policy_terms if term in query}
        query_tokens = self._extract_meaningful_tokens(query)

        scored_hits = []
        for hit in hits:
            rule_score = self._score_hit(query, query_markers, query_terms, query_tokens, hit)
            combined_score = (0.20 * hit.score) + (0.80 * rule_score)
            scored_hits.append(SearchHit(
                id=hit.id,
                score=combined_score,
                content=hit.content,
                metadata={
                    **(hit.metadata or {}),
                    "base_score": hit.score,
                    "policy_rule_score": rule_score,
                },
                doc_id=hit.doc_id,
                chunk_index=hit.chunk_index,
            ))

        scored_hits.sort(key=lambda x: x.score, reverse=True)
        return scored_hits[:top_k] if top_k else scored_hits

    def _extract_structure_markers(self, text: str) -> List[str]:
        markers = []
        for pattern in self.structure_patterns:
            markers.extend(re.findall(pattern, text or ""))
        return markers

    def _extract_meaningful_tokens(self, text: str) -> List[str]:
        tokens = re.findall(r'[\u4e00-\u9fffA-Za-z0-9]{2,}', text or "")
        stop = {"什么", "如何", "怎么", "是否", "有关", "规定", "政策", "文件", "企业", "政府"}
        return [token for token in tokens if token not in stop][:20]

    def _score_hit(
        self,
        query: str,
        query_markers: List[str],
        query_terms: set,
        query_tokens: List[str],
        hit: SearchHit,
    ) -> float:
        metadata = hit.metadata or {}
        searchable = " ".join([
            hit.content or "",
            str(metadata.get("article_number", "")),
            str(metadata.get("article_title", "")),
            str(metadata.get("title", "")),
            str(metadata.get("section_path", "")),
            str(metadata.get("filename", "")),
        ])

        score = 0.0

        for marker in query_markers:
            if marker and marker in searchable:
                score += 2.5

        for term in query_terms:
            if term in searchable:
                score += 1.6

        for token in query_tokens:
            if token in searchable:
                score += 0.25

        if metadata.get("article_number") and any(marker in str(metadata.get("article_number")) for marker in query_markers):
            score += 1.5

        if metadata.get("section_path") and any(term in str(metadata.get("section_path")) for term in query_terms):
            score += 1.0

        if query and hit.content and query in hit.content:
            score += 1.0

        return min(score / 3.0, 1.0)


class LLMReranker(BaseReranker):
    """LLM重排序器 - 使用Kimi进行相关性判断"""
    
    def __init__(self, llm_client, max_hits_to_rerank: int = 5):
        """
        初始化LLM重排序器
        
        Args:
            llm_client: Kimi LLM客户端实例
            max_hits_to_rerank: 最多重排序的文档数
        """
        super().__init__("llm_reranker")
        self.llm_client = llm_client
        self.max_hits_to_rerank = max_hits_to_rerank
        
        # 缓存
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_cache_key(self, query: str, hits: List[SearchHit]) -> str:
        """生成缓存键"""
        # 使用查询和文档ID的哈希
        doc_ids = ",".join(sorted([str(hit.id) for hit in hits[:self.max_hits_to_rerank]]))
        cache_str = f"{query}::{doc_ids}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def rerank(self, query: str, hits: List[SearchHit], top_k: Optional[int] = None) -> List[SearchHit]:
        """使用LLM进行重排序"""
        if not hits or len(hits) <= 1:
            return hits[:top_k] if top_k else hits
        
        start_time = time.time()
        
        # 限制处理的文档数量
        hits_to_rerank = hits[:min(self.max_hits_to_rerank, len(hits))]
        
        # 检查缓存
        cache_key = self._get_cache_key(query, hits_to_rerank)
        if cache_key in self.cache:
            self.cache_hits += 1
            logger.debug(f"使用缓存的重排序结果")
            cached_result = self.cache[cache_key]
            
            # 重新组装结果
            reranked_hits = cached_result + [h for h in hits if h not in hits_to_rerank]
            
            if top_k:
                return reranked_hits[:top_k]
            return reranked_hits
        
        self.cache_misses += 1
        
        try:
            # 构建提示词
            prompt = self._build_rerank_prompt(query, hits_to_rerank)
            
            # 这里需要根据您的llm_client实际接口来调用
            # 假设您的llm_client有generate方法
            response = self._call_llm_simple(prompt)
            
            # 解析响应
            reranked_indices = self._parse_llm_response(response, len(hits_to_rerank))
            
            if reranked_indices:
                # 重新排列
                reranked = []
                used_indices = set()
                
                for idx in reranked_indices:
                    if 0 <= idx < len(hits_to_rerank) and idx not in used_indices:
                        reranked.append(hits_to_rerank[idx])
                        used_indices.add(idx)
                
                # 添加未排序的文档
                for i, hit in enumerate(hits_to_rerank):
                    if i not in used_indices:
                        reranked.append(hit)
                
                # 缓存结果
                self.cache[cache_key] = reranked
                
                # 组装完整结果
                final_reranked = reranked + [h for h in hits if h not in hits_to_rerank]
                
                rerank_time = time.time() - start_time
                logger.info(f"LLM重排序完成，耗时: {rerank_time:.3f}s，处理 {len(hits_to_rerank)} 个文档")
                
                if top_k:
                    return final_reranked[:top_k]
                return final_reranked
            else:
                # 解析失败时使用简单排序
                simple_reranker = SimpleScoreReranker()
                return simple_reranker.rerank(query, hits, top_k)
                
        except Exception as e:
            logger.error(f"LLM重排序失败: {str(e)}，使用简单排序")
            simple_reranker = SimpleScoreReranker()
            return simple_reranker.rerank(query, hits, top_k)
    
    def _build_rerank_prompt(self, query: str, hits: List[SearchHit]) -> str:
        """构建重排序提示词"""
        # 构建文档列表
        docs_text = ""
        for i, hit in enumerate(hits):
            # 截断文档内容
            content = hit.content[:500].strip()
            docs_text += f"文档{i+1}:\n{content}\n\n"
        
        prompt = f"""请根据相关性对以下文档进行排序。

问题: {query}

{docs_text}
请根据文档与问题的相关性，从最相关到最不相关排序。
请严格按照以下格式输出，只输出数字序列，用逗号分隔：
1,3,2,4,5,...

例如，如果你认为文档2最相关，然后是文档1，最后是文档3，就输出：2,1,3

排序结果: """
        
        return prompt
    
    def _call_llm_simple(self, prompt: str) -> str:
        """简化版的LLM调用"""
        # 这里需要根据您的llm_client实际接口调整
        # 假设您的llm_client有ChatMessage和generate方法
        from src.generation.generator import ChatMessage
        
        messages = [
            ChatMessage(role="system", content="你是一个专业的相关性排序助手。"),
            ChatMessage(role="user", content=prompt)
        ]
        
        try:
            # 调用LLM
            response = self.llm_client.generate(messages)
            return response.strip()
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            raise
    
    def _parse_llm_response(self, response: str, num_docs: int) -> List[int]:
        """解析LLM响应"""
        try:
            # 清理响应
            response = response.strip()
            
            # 尝试提取数字序列
            match = re.search(r'([\d,\s]+)', response)
            if match:
                numbers = match.group(1)
                # 分割并转换
                indices = []
                for num in numbers.split(','):
                    num = num.strip()
                    if num.isdigit():
                        idx = int(num) - 1  # 转换为0-based索引
                        if 0 <= idx < num_docs:
                            indices.append(idx)
                
                # 验证：是否包含所有文档
                if len(set(indices)) == min(num_docs, len(indices)):
                    return indices
            
            # 尝试解析其他格式
            for line in response.split('\n'):
                if '排序' in line or '结果' in line:
                    # 尝试提取数字
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        indices = [int(num) - 1 for num in numbers if 0 <= int(num) - 1 < num_docs]
                        if indices:
                            return indices
            
            logger.warning(f"无法解析LLM响应: {response}")
            return []
            
        except Exception as e:
            logger.error(f"解析LLM响应失败: {str(e)}")
            return []


class HybridReranker(BaseReranker):
    """混合重排序器 - 组合多个重排序方法"""
    
    def __init__(self, rerankers: List[BaseReranker], weights: Optional[List[float]] = None):
        super().__init__("hybrid")
        self.rerankers = rerankers
        
        if weights is None:
            weights = [1.0 / len(rerankers)] * len(rerankers)
        self.weights = weights
        
        logger.info(f"混合重排序器初始化: 包含 {len(rerankers)} 个子重排序器")
    
    def rerank(self, query: str, hits: List[SearchHit], top_k: Optional[int] = None) -> List[SearchHit]:
        """混合重排序"""
        if not hits:
            return []
        
        if len(self.rerankers) == 0:
            return hits[:top_k] if top_k else hits
        
        start_time = time.time()
        
        # 存储每个重排序器的结果
        all_scores = {}
        
        for i, reranker in enumerate(self.rerankers):
            try:
                # 获取重排序结果
                reranked = reranker.rerank(query, hits, None)  # 先不限制top_k
                
                # 记录排名分数
                for rank, hit in enumerate(reranked):
                    if hit.id not in all_scores:
                        all_scores[hit.id] = {
                            'hit': hit,
                            'scores': [0.0] * len(self.rerankers)
                        }
                    # 排名越靠前，分数越高
                    all_scores[hit.id]['scores'][i] = 1.0 / (rank + 1)
            except Exception as e:
                logger.warning(f"子重排序器 {reranker.name} 失败: {str(e)}")
        
        # 计算加权分数
        final_hits = []
        for data in all_scores.values():
            weighted_score = sum(
                score * weight 
                for score, weight in zip(data['scores'], self.weights)
            )
            
            # 创建新结果
            hit = data['hit']
            hit.rerank_score = weighted_score
            hit.combined_score = 0.3 * hit.score + 0.7 * weighted_score
            final_hits.append(hit)
        
        # 按组合分数排序
        final_hits.sort(key=lambda x: x.combined_score, reverse=True)
        
        rerank_time = time.time() - start_time
        logger.info(f"混合重排序完成，耗时: {rerank_time:.3f}s，处理 {len(hits)} 个文档")
        
        if top_k:
            return final_hits[:top_k]
        return final_hits


# 使用示例
if __name__ == "__main__":
    # 测试示例
    from src.generation.generator import KimiLLMClient
    
    # 创建测试数据
    test_hits = [
        SearchHit(
            id="1",
            score=0.8,
            content="第一条 本法适用于所有国家机关工作人员",
            metadata={"article_number": "第一条"},
            doc_id="doc1",
            chunk_index=0
        ),
        SearchHit(
            id="2", 
            score=0.7,
            content="第二条 工作人员的职责包括...",
            metadata={"article_number": "第二条"},
            doc_id="doc1",
            chunk_index=1
        ),
        SearchHit(
            id="3",
            score=0.9,
            content="关于培训的具体规定在第三条中",
            metadata={"article_number": "第三条"},
            doc_id="doc1", 
            chunk_index=2
        )
    ]
    
    # 测试简单重排序
    print("=== 测试简单重排序 ===")
    simple_reranker = SimpleScoreReranker()
    result = simple_reranker.rerank("第一条是什么？", test_hits)
    for i, hit in enumerate(result):
        print(f"{i+1}. ID={hit.id}, Score={hit.score}, 内容={hit.content[:30]}...")
    
    # 测试规则重排序
    print("\n=== 测试规则重排序 ===")
    rule_reranker = RuleBasedReranker()
    result = rule_reranker.rerank("第三条是什么内容？", test_hits)
    for i, hit in enumerate(result):
        print(f"{i+1}. ID={hit.id}, 规则分={getattr(hit, 'rule_score', 0):.2f}, 组合分={getattr(hit, 'combined_score', 0):.2f}")
    
    print("\n✅ 重排序器测试完成")
