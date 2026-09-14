"""
src/knowledge_base/quality_assessor.py
文档和chunk质量评估器 - 第二阶段核心组件
"""
"""
src/knowledge_base/quality_assessor.py - 修复版
"""
import re
import json
import hashlib
import time  # 添加这行
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger

# 其余代码保持不变...

@dataclass
class QualityMetrics:
    """质量评估指标"""
    doc_score: float = 0.0  # 文档总体评分 (0-1)
    chunk_scores: List[float] = None  # 每个chunk的评分
    issues: List[str] = None  # 发现的问题列表
    suggestions: List[str] = None  # 改进建议
    
    def __post_init__(self):
        if self.chunk_scores is None:
            self.chunk_scores = []
        if self.issues is None:
            self.issues = []
        if self.suggestions is None:
            self.suggestions = []
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ChunkQuality:
    """Chunk质量详情"""
    chunk_index: int
    score: float
    issues: List[str]
    word_count: int
    sentence_count: int
    has_truncation: bool
    chunk_hash: str = ""


class QualityAssessor:
    """文档和Chunk质量评估器 - 按第二阶段方案实现"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # 质量阈值配置
        self.thresholds = {
            'min_doc_length': 50,      # 文档最小字符数
            'max_doc_length': 1000000,  # 文档最大字符数
            'min_chunk_length': 50,    # chunk最小字符数
            'max_chunk_length': 2000,  # chunk最大字符数
            'min_chunk_sentences': 1,  # chunk最少句子数
            'max_repetition_ratio': 0.3,  # 最大重复率
            'encoding_issue_threshold': 0.01,  # 编码问题阈值
        }
        # 更新配置
        self.thresholds.update(self.config.get('thresholds', {}))
        
        # 关键词列表（用于评估语义丰富度）
        self.semantic_keywords = [
            '规定', '条例', '法律', '政策', '标准', '要求', '禁止', '允许',
            '应当', '必须', '不得', '可以', '需要', '确保', '责任', '义务'
        ]
    
    def assess_document(self, document_content: str, filename: str = None) -> QualityMetrics:
        """
        评估文档级质量
        返回包含评分、问题和建议的QualityMetrics对象
        """
        metrics = QualityMetrics()
        content_length = len(document_content)
        
        # 1. 长度检查
        if content_length < self.thresholds['min_doc_length']:
            metrics.issues.append(f"文档过短 ({content_length} 字符)")
            metrics.suggestions.append("建议检查文档内容是否完整")
        elif content_length > self.thresholds['max_doc_length']:
            metrics.issues.append(f"文档过长 ({content_length} 字符)")
            metrics.suggestions.append("建议分割为多个文档处理")
        
        # 2. 编码问题检查
        encoding_issues = self._check_encoding_issues(document_content)
        metrics.issues.extend(encoding_issues)
        
        # 3. 内容重复率检查
        repetition_ratio = self._calculate_repetition_ratio(document_content)
        if repetition_ratio > self.thresholds['max_repetition_ratio']:
            metrics.issues.append(f"内容重复率高 ({repetition_ratio:.1%})")
            metrics.suggestions.append("建议去重或检查文档来源")
        
        # 4. 语义丰富度评估（针对规章文档）
        semantic_score = self._assess_semantic_richness(document_content)
        if semantic_score < 0.3:
            metrics.suggestions.append("文档语义丰富度较低，可能影响检索效果")
        
        # 5. 结构化程度评估
        structure_score = self._assess_structure(document_content)
        metrics.doc_score = self._calculate_overall_score(
            content_length, repetition_ratio, len(metrics.issues), 
            semantic_score, structure_score
        )
        
        logger.info(f"文档质量评估完成: 评分={metrics.doc_score:.2f}, 问题数={len(metrics.issues)}")
        return metrics
    
    def assess_chunks(self, chunks: List[Dict[str, Any]]) -> List[ChunkQuality]:
        """批量评估chunk质量"""
        chunk_qualities = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.get('content', '')
            metadata = chunk.get('metadata', {})
            
            # 计算chunk哈希（用于去重）
            chunk_hash = hashlib.md5(content.encode()).hexdigest()[:16]
            
            # 评估单个chunk
            quality = self._assess_single_chunk(content, i, chunk_hash)
            chunk_qualities.append(quality)
            
            # 将质量信息添加到metadata
            if metadata:
                metadata['quality_score'] = quality.score
                metadata['chunk_hash'] = chunk_hash
                metadata['word_count'] = quality.word_count
                metadata['sentence_count'] = quality.sentence_count
        
        return chunk_qualities
    
    def _assess_single_chunk(self, content: str, chunk_index: int, chunk_hash: str) -> ChunkQuality:
        """评估单个chunk质量"""
        issues = []
        
        # 1. 长度检查
        content_length = len(content)
        if content_length < self.thresholds['min_chunk_length']:
            issues.append(f"chunk过短 ({content_length} 字符)")
        elif content_length > self.thresholds['max_chunk_length']:
            issues.append(f"chunk过长 ({content_length} 字符)")
        
        # 2. 句子完整性检查
        sentences = re.split(r'[。！？；.!?;]+', content)
        sentence_count = len([s for s in sentences if s.strip()])
        
        if sentence_count < self.thresholds['min_chunk_sentences']:
            issues.append("可能包含不完整句子")
        
        # 3. 截断检查
        has_truncation = self._has_truncated_sentence(content)
        if has_truncation:
            issues.append("检测到句子被截断")
        
        # 4. 单词数统计
        word_count = len(content.split())
        
        # 5. 计算chunk评分
        score = self._calculate_chunk_score(
            content_length, sentence_count, has_truncation, len(issues)
        )
        
        return ChunkQuality(
            chunk_index=chunk_index,
            score=score,
            issues=issues,
            word_count=word_count,
            sentence_count=sentence_count,
            has_truncation=has_truncation,
            chunk_hash=chunk_hash
        )
    
    def _check_encoding_issues(self, content: str) -> List[str]:
        """检查编码问题"""
        issues = []
        
        # 检查乱码字符（如�）
        if '�' in content:
            issues.append("检测到乱码字符(�)")
        
        # 检查不可打印字符
        non_printable_pattern = r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]'
        non_printable_chars = re.findall(non_printable_pattern, content)
        
        if non_printable_chars:
            non_printable_ratio = len(non_printable_chars) / max(len(content), 1)
            if non_printable_ratio > self.thresholds['encoding_issue_threshold']:
                issues.append(f"不可打印字符过多 ({len(non_printable_chars)} 个)")
        
        return issues
    
    def _calculate_repetition_ratio(self, content: str) -> float:
        """计算内容重复率"""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) < 3:
            return 0.0
        
        # 计算连续重复行比例
        repeated_lines = 0
        for i in range(1, len(lines)):
            if lines[i] == lines[i-1]:
                repeated_lines += 1
        
        return repeated_lines / len(lines)
    
    def _assess_semantic_richness(self, content: str) -> float:
        """评估语义丰富度（针对规章文档）"""
        if not content:
            return 0.0
        
        # 统计关键词出现频率
        keyword_count = 0
        for keyword in self.semantic_keywords:
            keyword_count += content.count(keyword)
        
        # 归一化到0-1
        word_count = len(content.split())
        if word_count == 0:
            return 0.0
        
        return min(1.0, keyword_count / (word_count / 100))  # 每100词中的关键词数
    
    def _assess_structure(self, content: str) -> float:
        """评估文档结构化程度"""
        # 检测条文编号模式
        article_patterns = [
            r'第[一二三四五六七八九十百]+条',
            r'[一二三四五六七八九十]+、',
            r'（[一二三四五六七八九十]+）',
            r'\d+\.',
            r'[A-Z]\.',
            r'[a-z]\)',
        ]
        
        total_matches = 0
        for pattern in article_patterns:
            total_matches += len(re.findall(pattern, content))
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return 0.0
        
        # 计算结构化密度
        structure_density = total_matches / len(lines)
        
        # 归一化：每行有0.1个结构标记得满分1.0
        return min(1.0, structure_density / 0.1)
    
    def _has_truncated_sentence(self, content: str) -> bool:
        """检查是否有句子被截断"""
        if not content:
            return False
        
        # 句子结束符
        sentence_endings = {'。', '！', '？', '；', '.', '!', '?', ';'}
        
        # 如果最后字符不是句子结束符，且长度足够，可能被截断
        last_char = content[-1]
        return last_char not in sentence_endings and len(content) > 50
    
    def _calculate_overall_score(self, length: int, repetition: float, 
                               issue_count: int, semantic_score: float, 
                               structure_score: float) -> float:
        """计算文档总体评分"""
        # 长度分数 (0-0.3)
        length_score = 0.0
        if self.thresholds['min_doc_length'] <= length <= self.thresholds['max_doc_length']:
            ideal_length = 5000  # 假设5000字符为理想长度
            length_diff = abs(length - ideal_length)
            length_score = max(0.0, 0.3 - (length_diff / ideal_length * 0.3))
        
        # 内容质量分数 (0-0.3)
        content_score = 0.3 * (1.0 - min(repetition, 1.0))
        
        # 语义和结构分数 (0-0.3)
        semantic_structure_score = 0.15 * semantic_score + 0.15 * structure_score
        
        # 问题惩罚 (0-0.1)
        issue_penalty = min(0.1, issue_count * 0.02)
        
        total_score = length_score + content_score + semantic_structure_score - issue_penalty
        return max(0.0, min(1.0, total_score))
    
    def _calculate_chunk_score(self, length: int, sentence_count: int, 
                             has_truncation: bool, issue_count: int) -> float:
        """计算chunk评分"""
        # 长度分数 (0-0.4)
        length_score = 0.0
        if self.thresholds['min_chunk_length'] <= length <= self.thresholds['max_chunk_length']:
            ideal_length = 500  # 假设500字符为理想长度
            length_diff = abs(length - ideal_length)
            length_score = max(0.0, 0.4 - (length_diff / ideal_length * 0.4))
        
        # 句子完整性分数 (0-0.3)
        sentence_score = 0.0
        if sentence_count >= self.thresholds['min_chunk_sentences']:
            sentence_score = 0.3
        elif sentence_count > 0:
            sentence_score = 0.15
        
        # 截断惩罚 (0-0.2)
        truncation_penalty = 0.2 if has_truncation else 0.0
        
        # 问题惩罚 (0-0.1)
        issue_penalty = min(0.1, issue_count * 0.05)
        
        total_score = length_score + sentence_score - truncation_penalty - issue_penalty
        return max(0.0, min(1.0, total_score))
    
    def generate_quality_report(self, doc_metrics: QualityMetrics, 
                               chunk_qualities: List[ChunkQuality],
                               filename: str = None) -> Dict[str, Any]:
        """生成质量评估报告"""
        # 统计chunk质量分布
        excellent_chunks = sum(1 for cq in chunk_qualities if cq.score >= 0.8)
        good_chunks = sum(1 for cq in chunk_qualities if 0.6 <= cq.score < 0.8)
        fair_chunks = sum(1 for cq in chunk_qualities if 0.4 <= cq.score < 0.6)
        poor_chunks = sum(1 for cq in chunk_qualities if cq.score < 0.4)
        
        # 统计问题类型
        total_issues = len(doc_metrics.issues) + sum(len(cq.issues) for cq in chunk_qualities)
        truncation_issues = sum(1 for cq in chunk_qualities if cq.has_truncation)
        short_chunks = sum(1 for cq in chunk_qualities if len(cq.chunk_hash) > 0 and cq.score < 0.3)
        
        report = {
            "report_id": hashlib.md5(str(time.time()).encode()).hexdigest()[:12],
            "timestamp": datetime.now().isoformat(),
            "filename": filename or "unknown",
            "document_metrics": doc_metrics.to_dict(),
            "chunk_summary": {
                "total_chunks": len(chunk_qualities),
                "avg_chunk_score": sum(cq.score for cq in chunk_qualities) / max(len(chunk_qualities), 1),
                "quality_distribution": {
                    "excellent": excellent_chunks,
                    "good": good_chunks,
                    "fair": fair_chunks,
                    "poor": poor_chunks
                },
                "issue_statistics": {
                    "total_issues": total_issues,
                    "truncation_issues": truncation_issues,
                    "short_chunks": short_chunks
                }
            },
            "recommendations": self._generate_recommendations(doc_metrics, chunk_qualities),
            "chunk_details": [
                {
                    "chunk_index": cq.chunk_index,
                    "score": round(cq.score, 3),
                    "word_count": cq.word_count,
                    "sentence_count": cq.sentence_count,
                    "has_truncation": cq.has_truncation,
                    "issues": cq.issues,
                    "chunk_hash": cq.chunk_hash
                }
                for cq in chunk_qualities
            ]
        }
        
        return report
    
    def _generate_recommendations(self, doc_metrics: QualityMetrics, 
                                chunk_qualities: List[ChunkQuality]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 文档级建议
        recommendations.extend(doc_metrics.suggestions)
        
        # Chunk级建议
        poor_chunks = sum(1 for cq in chunk_qualities if cq.score < 0.3)
        if poor_chunks > len(chunk_qualities) * 0.2:  # 超过20%的chunk质量差
            recommendations.append(f"{poor_chunks}个chunk质量较低，建议调整分块参数")
        
        truncated_chunks = sum(1 for cq in chunk_qualities if cq.has_truncation)
        if truncated_chunks > 0:
            recommendations.append(f"{truncated_chunks}个chunk存在句子截断，建议使用句子边界分割")
        
        # 重复chunk检测
        chunk_hashes = [cq.chunk_hash for cq in chunk_qualities if cq.chunk_hash]
        unique_hashes = set(chunk_hashes)
        if len(chunk_hashes) > len(unique_hashes):
            duplicates = len(chunk_hashes) - len(unique_hashes)
            recommendations.append(f"检测到{duplicates}个重复chunk，建议检查分块逻辑")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], output_dir: str = "data/quality_reports"):
        """保存质量报告到文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = report.get("filename", "unknown").replace("/", "_").replace("\\", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"quality_{filename}_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"质量报告已保存: {report_file}")
        return str(report_file)