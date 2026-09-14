"""
QA数据集构建器
"""
import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import random


@dataclass
class QAExample:
    """QA数据示例"""
    id: str
    question: str
    ground_truth_answer: str
    ground_truth_docs: List[str]  # 相关文档ID列表
    metadata: Dict[str, Any] = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "question": self.question,
            "ground_truth_answer": self.ground_truth_answer,
            "ground_truth_docs": self.ground_truth_docs,
            "metadata": self.metadata or {}
        }


class QADatasetBuilder:
    """QA数据集构建器"""
    
    def __init__(self, data_dir: str = "data/evaluation"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def build_from_documents(self, documents: List[Dict], num_questions: int = 50) -> List[QAExample]:
        """从文档构建QA数据集"""
        logger.info(f"从 {len(documents)} 个文档构建QA数据集，目标问题数: {num_questions}")
        
        qa_examples = []
        
        # 策略1：基于标题的问题
        qa_examples.extend(self._generate_title_questions(documents))
        
        # 策略2：基于关键句的问题
        qa_examples.extend(self._generate_sentence_questions(documents))
        
        # 策略3：基于摘要的问题
        qa_examples.extend(self._generate_summary_questions(documents))
        
        # 如果还不够，随机生成一些
        if len(qa_examples) < num_questions:
            qa_examples.extend(self._generate_random_questions(documents, num_questions - len(qa_examples)))
        
        # 打乱顺序并限制数量
        random.shuffle(qa_examples)
        qa_examples = qa_examples[:num_questions]
        
        # 分配ID
        for i, qa in enumerate(qa_examples):
            qa.id = f"qa_{i+1:04d}"
        
        logger.info(f"QA数据集构建完成: {len(qa_examples)} 个问题")
        return qa_examples
    
    def _generate_title_questions(self, documents: List[Dict]) -> List[QAExample]:
        """基于标题生成问题"""
        examples = []
        
        for doc in documents:
            title = doc.get("title") or doc.get("filename", "").replace(".txt", "").replace(".pdf", "")
            if not title or len(title) < 3:
                continue
            
            # 提取标题中的关键信息
            title_clean = title.strip()
            
            # 生成不同类型的问题
            question_templates = [
                f"关于'{title_clean}'的主要内容是什么？",
                f"请介绍一下'{title_clean}'。",
                f"'{title_clean}'的核心要点有哪些？",
                f"你能总结一下'{title_clean}'吗？"
            ]
            
            for template in question_templates:
                examples.append(QAExample(
                    id="",
                    question=template,
                    ground_truth_answer=f"[基于文档标题的问题] 文档标题: {title_clean}",
                    ground_truth_docs=[doc.get("doc_id", "")],
                    metadata={"source": "title", "doc_id": doc.get("doc_id", "")}
                ))
        
        return examples
    
    def _generate_sentence_questions(self, documents: List[Dict]) -> List[QAExample]:
        """基于关键句生成问题"""
        examples = []
        
        for doc in documents:
            content = doc.get("content", "")
            if not content or len(content) < 50:
                continue
            
            # 分割成句子
            sentences = self._split_sentences(content)
            
            # 选择关键句（包含数字、关键词的句子）
            key_sentences = []
            for sent in sentences:
                if len(sent) > 20 and len(sent) < 200:
                    # 检查是否包含重要信息
                    if any(keyword in sent.lower() for keyword in ["要求", "规定", "必须", "应当", "禁止", "不得"]):
                        key_sentences.append(sent)
                    elif re.search(r'\d+', sent):  # 包含数字
                        key_sentences.append(sent)
            
            # 为每个关键句生成问题
            for sent in key_sentences[:3]:  # 每个文档最多3个
                # 提取句子中的关键信息
                if "要求" in sent:
                    question = f"根据文档，有哪些具体要求？"
                elif "规定" in sent:
                    question = f"文档中的相关规定是什么？"
                elif re.search(r'\d+', sent):
                    question = f"文档中提到的数字信息有哪些？"
                else:
                    question = f"这句话的主要意思是什么？"
                
                examples.append(QAExample(
                    id="",
                    question=question,
                    ground_truth_answer=f"[基于关键句的问题] 关键句: {sent[:100]}...",
                    ground_truth_docs=[doc.get("doc_id", "")],
                    metadata={"source": "sentence", "doc_id": doc.get("doc_id", ""), "sentence": sent[:200]}
                ))
        
        return examples
    
    def _generate_summary_questions(self, documents: List[Dict]) -> List[QAExample]:
        """基于摘要生成问题"""
        examples = []
        
        for doc in documents:
            content = doc.get("content", "")
            if not content or len(content) < 100:
                continue
            
            # 提取前200字符作为摘要
            summary = content[:200] + "..." if len(content) > 200 else content
            
            # 生成摘要相关的问题
            question_templates = [
                f"这个文档主要讲了什么？",
                f"请概括一下这个文档的主要内容。",
                f"文档的核心思想是什么？"
            ]
            
            for template in question_templates:
                examples.append(QAExample(
                    id="",
                    question=template,
                    ground_truth_answer=f"[基于摘要的问题] 文档摘要: {summary}",
                    ground_truth_docs=[doc.get("doc_id", "")],
                    metadata={"source": "summary", "doc_id": doc.get("doc_id", ""), "summary": summary}
                ))
        
        return examples
    
    def _generate_random_questions(self, documents: List[Dict], num_questions: int) -> List[QAExample]:
        """生成随机问题"""
        examples = []
        
        general_questions = [
            "这个文档是什么类型的？",
            "文档的作者是谁？",
            "文档的发布日期是什么时候？",
            "文档的主要章节有哪些？",
            "文档的关键词是什么？",
            "文档的目标读者是谁？",
            "文档的用途是什么？",
            "文档中的表格和图表说明了什么？",
            "文档的结论是什么？",
            "文档的建议有哪些？"
        ]
        
        for i in range(min(num_questions, len(general_questions))):
            doc = random.choice(documents) if documents else {}
            
            examples.append(QAExample(
                id="",
                question=general_questions[i],
                ground_truth_answer=f"[通用问题] 这是一个关于文档内容的问题。",
                ground_truth_docs=[doc.get("doc_id", "")] if doc else [],
                metadata={"source": "general", "doc_id": doc.get("doc_id", "") if doc else ""}
            ))
        
        return examples
    
    def _split_sentences(self, text: str) -> List[str]:
        """简单的句子分割"""
        # 使用标点符号分割句子
        sentences = re.split(r'[。！？!?;；]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def save_dataset(self, qa_examples: List[QAExample], filename: str = "qa_dataset.json"):
        """保存数据集"""
        filepath = os.path.join(self.data_dir, filename)
        
        data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "num_examples": len(qa_examples),
            "examples": [qa.to_dict() for qa in qa_examples]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据集已保存: {filepath} ({len(qa_examples)} 个问题)")
        return filepath
    
    def load_dataset(self, filename: str = "qa_dataset.json") -> List[QAExample]:
        """加载数据集"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"数据集文件不存在: {filepath}")
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        for item in data.get("examples", []):
            examples.append(QAExample(
                id=item["id"],
                question=item["question"],
                ground_truth_answer=item["ground_truth_answer"],
                ground_truth_docs=item["ground_truth_docs"],
                metadata=item.get("metadata", {})
            ))
        
        logger.info(f"数据集已加载: {filepath} ({len(examples)} 个问题)")
        return examples


if __name__ == "__main__":
    # 测试数据集构建
    from datetime import datetime
    
    builder = QADatasetBuilder()
    
    # 模拟文档数据
    test_documents = [
        {
            "doc_id": "doc_001",
            "title": "项目管理规范",
            "content": "项目启动阶段必须完成可行性分析。项目执行阶段要求每周提交进度报告。项目验收标准包括功能完整性和性能指标。",
            "filename": "项目管理规范.pdf"
        },
        {
            "doc_id": "doc_002", 
            "title": "技术方案设计",
            "content": "系统架构采用微服务设计。数据库选用MySQL 8.0版本。前端使用React框架，后端使用Python Flask。",
            "filename": "技术方案.docx"
        }
    ]
    
    qa_dataset = builder.build_from_documents(test_documents, num_questions=10)
    builder.save_dataset(qa_dataset, "test_dataset.json")
    
    # 加载验证
    loaded_dataset = builder.load_dataset("test_dataset.json")
    print(f"加载了 {len(loaded_dataset)} 个问题")
    for i, qa in enumerate(loaded_dataset[:3]):
        print(f"{i+1}. 问题: {qa.question}")
        print(f"   答案: {qa.ground_truth_answer[:50]}...")
        print()