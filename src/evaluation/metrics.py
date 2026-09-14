"""
评估指标计算
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
import json
from collections import defaultdict


@dataclass
class RetrievalMetrics:
    """检索评估指标"""
    recall_at_k: Dict[int, float]  # Recall@K
    precision_at_k: Dict[int, float]  # Precision@K
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_k: Dict[int, float]  # nDCG@K
    avg_precision: float  # Average Precision
    
    def to_dict(self):
        return {
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "avg_precision": self.avg_precision
        }


@dataclass
class GenerationMetrics:
    """生成评估指标"""
    faithfulness_score: float  # 事实一致性分数
    citation_accuracy: float  # 引用准确率
    coverage_score: float  # 覆盖度
    bleu_score: float  # BLEU分数（可选）
    rouge_scores: Dict[str, float]  # ROUGE分数（可选）
    
    def to_dict(self):
        return {
            "faithfulness_score": self.faithfulness_score,
            "citation_accuracy": self.citation_accuracy,
            "coverage_score": self.coverage_score,
            "bleu_score": self.bleu_score,
            "rouge_scores": self.rouge_scores
        }


@dataclass
class EfficiencyMetrics:
    """效率评估指标"""
    p50_latency: float  # 50百分位延迟
    p95_latency: float  # 95百分位延迟
    avg_latency: float  # 平均延迟
    qps: float  # 每秒查询数
    avg_tokens_per_query: float  # 平均token数
    cache_hit_rate: float  # 缓存命中率
    
    def to_dict(self):
        return {
            "p50_latency": self.p50_latency,
            "p95_latency": self.p95_latency,
            "avg_latency": self.avg_latency,
            "qps": self.qps,
            "avg_tokens_per_query": self.avg_tokens_per_query,
            "cache_hit_rate": self.cache_hit_rate
        }


class MetricsCalculator:
    """指标计算器"""
    
    @staticmethod
    def calculate_retrieval_metrics(
        retrieved_docs: List[List[str]],  # 检索到的文档ID列表（每个查询一个列表）
        ground_truth_docs: List[List[str]],  # 真实相关文档ID列表
        k_values: List[int] = [1, 3, 5, 10]
    ) -> RetrievalMetrics:
        """计算检索指标"""
        recalls = {}
        precisions = {}
        ndcgs = {}
        
        total_reciprocal_rank = 0
        total_avg_precision = 0
        num_queries = len(retrieved_docs)
        
        for k in k_values:
            recall_sum = 0
            precision_sum = 0
            ndcg_sum = 0
            
            for i in range(num_queries):
                retrieved = retrieved_docs[i][:k]
                relevant = set(ground_truth_docs[i])
                
                # Recall@K
                relevant_retrieved = [doc for doc in retrieved if doc in relevant]
                recall = len(relevant_retrieved) / max(len(relevant), 1)
                recall_sum += recall
                
                # Precision@K
                precision = len(relevant_retrieved) / max(len(retrieved), 1)
                precision_sum += precision
                
                # nDCG@K
                dcg = 0
                for rank, doc in enumerate(retrieved, 1):
                    if doc in relevant:
                        dcg += 1 / np.log2(rank + 1)
                
                # 理想DCG
                ideal_retrieved = list(relevant)[:k]
                idcg = sum(1 / np.log2(r + 1) for r in range(1, min(len(ideal_retrieved), k) + 1))
                
                ndcg = dcg / max(idcg, 1e-10)
                ndcg_sum += ndcg
                
                # 只计算一次MRR和AP（使用K=10）
                if k == 10:
                    # Mean Reciprocal Rank
                    for rank, doc in enumerate(retrieved, 1):
                        if doc in relevant:
                            total_reciprocal_rank += 1 / rank
                            break
                    
                    # Average Precision
                    ap = 0
                    num_relevant_found = 0
                    for rank, doc in enumerate(retrieved, 1):
                        if doc in relevant:
                            num_relevant_found += 1
                            ap += num_relevant_found / rank
                    ap = ap / max(len(relevant), 1)
                    total_avg_precision += ap
            
            recalls[k] = recall_sum / max(num_queries, 1)
            precisions[k] = precision_sum / max(num_queries, 1)
            ndcgs[k] = ndcg_sum / max(num_queries, 1)
        
        mrr = total_reciprocal_rank / max(num_queries, 1)
        avg_precision = total_avg_precision / max(num_queries, 1)
        
        return RetrievalMetrics(
            recall_at_k=recalls,
            precision_at_k=precisions,
            mrr=mrr,
            ndcg_at_k=ndcgs,
            avg_precision=avg_precision
        )
    
    @staticmethod
    def calculate_generation_metrics(
        generated_answers: List[str],
        ground_truth_answers: List[str],
        retrieved_docs: List[List[str]],
        ground_truth_docs: List[List[str]],
        citations: List[List[str]] = None  # 引用的文档ID
    ) -> GenerationMetrics:
        """计算生成指标"""
        num_queries = len(generated_answers)
        
        # 事实一致性（简化版）
        faithfulness_sum = 0
        for i in range(num_queries):
            # 简单检查：生成的答案是否包含真实答案中的关键词
            gen_answer = generated_answers[i].lower()
            truth_answer = ground_truth_answers[i].lower()
            
            # 提取关键词（简单方法）
            truth_words = set(word for word in truth_answer.split() if len(word) > 2)
            if truth_words:
                matched_words = sum(1 for word in truth_words if word in gen_answer)
                faithfulness = matched_words / len(truth_words)
            else:
                faithfulness = 0
            
            faithfulness_sum += min(faithfulness, 1.0)  # 限制在0-1之间
        
        faithfulness_score = faithfulness_sum / max(num_queries, 1)
        
        # 引用准确率
        citation_accuracy_sum = 0
        if citations:
            for i in range(num_queries):
                cited_docs = set(citations[i])
                relevant_docs = set(ground_truth_docs[i])
                
                if relevant_docs:
                    # 精确匹配：引用的文档是否都是相关的
                    correct_citations = cited_docs.intersection(relevant_docs)
                    accuracy = len(correct_citations) / max(len(cited_docs), 1) if cited_docs else 0
                    citation_accuracy_sum += accuracy
                else:
                    citation_accuracy_sum += 0
        
        citation_accuracy = citation_accuracy_sum / max(num_queries, 1) if citations else 0
        
        # 覆盖度
        coverage_sum = 0
        for i in range(num_queries):
            retrieved_set = set(retrieved_docs[i])
            relevant_set = set(ground_truth_docs[i])
            
            if relevant_set:
                coverage = len(retrieved_set.intersection(relevant_set)) / len(relevant_set)
                coverage_sum += coverage
        
        coverage_score = coverage_sum / max(num_queries, 1)
        
        # 文本相似度指标（可选）
        bleu_score = 0
        rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        
        # 这里可以添加BLEU和ROUGE计算，需要安装相应库
        # from nltk.translate.bleu_score import sentence_bleu
        # from rouge import Rouge
        
        return GenerationMetrics(
            faithfulness_score=faithfulness_score,
            citation_accuracy=citation_accuracy,
            coverage_score=coverage_score,
            bleu_score=bleu_score,
            rouge_scores=rouge_scores
        )
    
    @staticmethod
    def calculate_efficiency_metrics(
        latencies: List[float],  # 每个查询的延迟（秒）
        tokens_used: List[int],  # 每个查询的token数
        cache_hits: int,  # 缓存命中次数
        total_requests: int  # 总请求数
    ) -> EfficiencyMetrics:
        """计算效率指标"""
        if not latencies:
            return EfficiencyMetrics(
                p50_latency=0,
                p95_latency=0,
                avg_latency=0,
                qps=0,
                avg_tokens_per_query=0,
                cache_hit_rate=0
            )
        
        # 排序延迟
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        # 百分位延迟
        p50_idx = int(n * 0.5)
        p95_idx = int(n * 0.95)
        
        p50_latency = sorted_latencies[p50_idx] if p50_idx < n else sorted_latencies[-1]
        p95_latency = sorted_latencies[p95_idx] if p95_idx < n else sorted_latencies[-1]
        
        # 平均延迟
        avg_latency = sum(latencies) / n
        
        # QPS（假设所有查询是串行的）
        total_time = sum(latencies)
        qps = n / max(total_time, 1e-10)
        
        # 平均token数
        avg_tokens = sum(tokens_used) / max(len(tokens_used), 1)
        
        # 缓存命中率
        cache_hit_rate = cache_hits / max(total_requests, 1)
        
        return EfficiencyMetrics(
            p50_latency=p50_latency,
            p95_latency=p95_latency,
            avg_latency=avg_latency,
            qps=qps,
            avg_tokens_per_query=avg_tokens,
            cache_hit_rate=cache_hit_rate
        )
    
    @staticmethod
    def calculate_improvement_metrics(
        baseline_metrics: Dict[str, Any],
        optimized_metrics: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """计算改进指标"""
        improvements = {}
        
        for metric_name in baseline_metrics:
            if metric_name in optimized_metrics:
                baseline = baseline_metrics[metric_name]
                optimized = optimized_metrics[metric_name]
                
                if isinstance(baseline, (int, float)) and isinstance(optimized, (int, float)):
                    if baseline != 0:
                        improvement = (optimized - baseline) / abs(baseline) * 100
                    else:
                        improvement = 100 if optimized > 0 else 0
                    
                    improvements[metric_name] = {
                        "baseline": baseline,
                        "optimized": optimized,
                        "improvement_percent": improvement,
                        "absolute_change": optimized - baseline
                    }
        
        return improvements


if __name__ == "__main__":
    # 测试指标计算
    calculator = MetricsCalculator()
    
    # 测试检索指标
    retrieved = [
        ["doc1", "doc2", "doc3"],
        ["doc2", "doc4", "doc1"],
        ["doc3", "doc5", "doc2"]
    ]
    ground_truth = [
        ["doc1", "doc2"],
        ["doc2"],
        ["doc3", "doc5"]
    ]
    
    retrieval_metrics = calculator.calculate_retrieval_metrics(retrieved, ground_truth)
    print("检索指标:")
    print(f"Recall@1: {retrieval_metrics.recall_at_k.get(1, 0):.3f}")
    print(f"Recall@3: {retrieval_metrics.recall_at_k.get(3, 0):.3f}")
    print(f"MRR: {retrieval_metrics.mrr:.3f}")
    print(f"nDCG@3: {retrieval_metrics.ndcg_at_k.get(3, 0):.3f}")
    
    # 测试效率指标
    latencies = [0.1, 0.2, 0.15, 0.3, 0.25]
    tokens = [100, 150, 120, 200, 180]
    
    efficiency_metrics = calculator.calculate_efficiency_metrics(latencies, tokens, 2, 5)
    print("\n效率指标:")
    print(f"P50延迟: {efficiency_metrics.p50_latency:.3f}s")
    print(f"P95延迟: {efficiency_metrics.p95_latency:.3f}s")
    print(f"平均延迟: {efficiency_metrics.avg_latency:.3f}s")
    print(f"QPS: {efficiency_metrics.qps:.1f}")
    print(f"缓存命中率: {efficiency_metrics.cache_hit_rate:.1%}")