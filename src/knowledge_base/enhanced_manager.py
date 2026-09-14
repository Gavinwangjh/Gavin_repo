"""
src/knowledge_base/enhanced_manager.py
增强的知识库管理器 - 集成质量评估和完整KB管理功能
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from loguru import logger

from src.document_loader.loader import DocumentLoader, Document, DocumentChunk
from src.embedding.embedder import EmbeddingManager
from src.vector_store.milvus_store import MilvusVectorStore
from src.utils.helpers import PerformanceTimer

# 导入质量评估器
from .quality_assessor import QualityAssessor, QualityMetrics, ChunkQuality


@dataclass
class EnhancedSyncResult:
    """增强的同步结果（包含质量信息）"""
    doc_id: str
    mode: str
    inserted: int
    deleted: int
    skipped: bool
    reason: str = ""
    doc_hash: str = ""
    version: int = 1
    quality_score: float = 0.0
    chunk_count: int = 0
    quality_report_path: str = ""


@dataclass
class DocumentMetadata:
    """文档元数据增强"""
    source_type: str = "upload"  # upload, nsfc, web, api
    source_url: str = ""
    author: str = ""
    category: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self):
        return asdict(self)


class EnhancedKnowledgeBaseManager:
    """
    增强的知识库管理器 - 实现第二阶段所有功能
    1. 增量更新 + 版本控制
    2. 数据质量评估
    3. 完整KB管理字段
    4. 来源追溯
    """
    
    def __init__(
        self,
        loader: DocumentLoader,
        embedder: EmbeddingManager,
        store: MilvusVectorStore,
        state_dir: str = "data/kb_state",
        quality_report_dir: str = "data/quality_reports",
    ):
        self.loader = loader
        self.embedder = embedder
        self.store = store
        
        # 目录配置
        self.state_dir = Path(state_dir)
        self.quality_report_dir = Path(quality_report_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.quality_report_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化质量评估器
        self.quality_assessor = QualityAssessor()
        
        # 配置
        self.min_quality_threshold = 0.3  # 最低质量阈值
        self.enable_quality_check = True  # 是否启用质量检查
        
        logger.info(f"[KB-Enhanced] 初始化完成: state_dir={self.state_dir}, quality_dir={self.quality_report_dir}")
    
    # ----------------------------
    # 核心API：增强的文件同步
    # ----------------------------
    
    def sync_file_with_quality(
        self, 
        file_path: str, 
        mode: str = "incremental",
        source_type: str = "upload",
        source_url: str = "",
        metadata: Optional[DocumentMetadata] = None
    ) -> EnhancedSyncResult:
        """
        带质量检查的文件同步
        实现第二阶段所有功能：
        1. 质量评估
        2. 增量更新
        3. 来源追溯
        4. 质量报告生成
        """
        file_path = str(file_path)
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with PerformanceTimer("增强KB同步"):
            # 1. 加载文档
            doc: Document = self.loader.load_document(file_path)
            
            # 2. 文档级质量评估
            if self.enable_quality_check:
                doc_quality = self.quality_assessor.assess_document(
                    doc.content, 
                    filename=path.name
                )
                
                # 质量过滤：低于阈值给出警告
                if doc_quality.doc_score < self.min_quality_threshold:
                    logger.warning(f"文档质量过低 ({doc_quality.doc_score:.2f}): {path.name}")
            
            # 3. 分块
            chunks: List[DocumentChunk] = self.loader.chunk_document(doc)
            
            if not chunks:
                return EnhancedSyncResult(
                    doc_id=doc.doc_id,
                    mode=mode,
                    inserted=0,
                    deleted=0,
                    skipped=True,
                    reason="no chunks",
                    quality_score=doc_quality.doc_score if self.enable_quality_check else 0.0
                )
            
            # 4. Chunk级质量评估
            chunk_data = [
                {"content": chunk.content, "metadata": chunk.metadata or {}}
                for chunk in chunks
            ]
            
            chunk_qualities = []
            if self.enable_quality_check:
                chunk_qualities = self.quality_assessor.assess_chunks(chunk_data)
            
            # 5. 计算文档哈希和检查增量
            doc_hash = self._calculate_doc_hash(doc.content, metadata)
            prev_state = self._load_state(doc.doc_id)
            
            # 版本管理
            version = int(prev_state.get("version", 1)) if prev_state else 1
            if prev_state and prev_state.get("doc_hash") and prev_state["doc_hash"] != doc_hash:
                version += 1
            
            # 增量检查
            if mode == "incremental" and prev_state and prev_state.get("doc_hash") == doc_hash:
                logger.info(f"[KB-Enhanced] 跳过未变更文档: {path.name} (v{version})")
                return EnhancedSyncResult(
                    doc_id=doc.doc_id,
                    mode=mode,
                    inserted=0,
                    deleted=0,
                    skipped=True,
                    reason="unchanged",
                    doc_hash=doc_hash,
                    version=version,
                    quality_score=doc_quality.doc_score if self.enable_quality_check else 0.0,
                    chunk_count=len(chunks)
                )
            
            # 6. 删除旧数据（如果需要）
            deleted = 0
            if prev_state and (mode == "full" or prev_state.get("doc_hash") != doc_hash):
                deleted_count = self._delete_old_chunks(doc.doc_id)
                deleted = deleted_count
            
            # 7. 增强chunk元数据
            updated_at = int(time.time())
            for i, (chunk, chunk_quality) in enumerate(zip(chunks, chunk_qualities if chunk_qualities else [None] * len(chunks))):
                # 确保metadata是字典
                if not isinstance(chunk.metadata, dict):
                    chunk.metadata = {}
                
                # 基础KB字段
                chunk.metadata.update({
                    "doc_version": version,
                    "updated_at": updated_at,
                    "doc_hash": doc_hash,
                    "source_type": source_type,
                    "source_url": source_url,
                })
                
                # 文档元数据
                if metadata:
                    chunk.metadata.update(metadata.to_dict())
                
                # 质量信息
                if chunk_quality:
                    chunk.metadata.update({
                        "quality_score": chunk_quality.score,
                        "chunk_hash": chunk_quality.chunk_hash,
                        "word_count": chunk_quality.word_count,
                        "sentence_count": chunk_quality.sentence_count,
                        "quality_issues": chunk_quality.issues,
                    })
                
                # 结构化信息（如果可用）
                if "article_number" in chunk.metadata:
                    chunk.metadata["section_path"] = self._build_section_path(chunk.metadata)
            
            # 8. 生成嵌入和插入
            texts = [c.content for c in chunks]
            embeddings = self._extract_embeddings(self.embedder.embed_documents(texts))
            
            insert_result = self._insert_with_enhanced_fields(
                chunks, embeddings, version, updated_at, source_type, source_url
            )
            
            if not getattr(insert_result, "success", True):
                err = getattr(insert_result, "error", "插入失败")
                raise RuntimeError(f"[KB-Enhanced] Milvus插入失败: {err}")
            
            inserted = getattr(insert_result, "insert_count", len(chunks))
            
            # 9. 生成质量报告
            quality_report_path = ""
            if self.enable_quality_check and chunk_qualities:
                report = self.quality_assessor.generate_quality_report(
                    doc_quality, chunk_qualities, path.name
                )
                quality_report_path = self.quality_assessor.save_report(
                    report, str(self.quality_report_dir)
                )
            
            # 10. 保存状态
            self._save_state(
                doc_id=doc.doc_id,
                state={
                    "doc_id": doc.doc_id,
                    "file_path": str(path.resolve()),
                    "filename": path.name,
                    "doc_hash": doc_hash,
                    "chunk_count": len(chunks),
                    "version": version,
                    "updated_at": updated_at,
                    "source_type": source_type,
                    "source_url": source_url,
                    "quality_score": doc_quality.doc_score if self.enable_quality_check else 0.0,
                    "last_sync": datetime.now().isoformat(),
                }
            )
            
            logger.info(f"[KB-Enhanced] {mode}同步完成: {path.name}, chunks={len(chunks)}, inserted={inserted}, quality={doc_quality.doc_score if self.enable_quality_check else 'N/A'}")
            
            return EnhancedSyncResult(
                doc_id=doc.doc_id,
                mode=mode,
                inserted=inserted,
                deleted=deleted,
                skipped=False,
                reason="synced",
                doc_hash=doc_hash,
                version=version,
                quality_score=doc_quality.doc_score if self.enable_quality_check else 0.0,
                chunk_count=len(chunks),
                quality_report_path=quality_report_path
            )
    
    def batch_sync_with_quality(
        self,
        dir_path: str,
        recursive: bool = False,
        mode: str = "incremental",
        source_type: str = "upload",
        metadata: Optional[DocumentMetadata] = None
    ) -> List[EnhancedSyncResult]:
        """批量同步目录（带质量检查）"""
        dirp = Path(dir_path)
        if not dirp.exists() or not dirp.is_dir():
            raise FileNotFoundError(f"目录不存在: {dir_path}")
        
        supported = getattr(self.loader, "SUPPORTED_FORMATS", {".pdf", ".txt", ".md", ".docx"})
        pattern = "**/*" if recursive else "*"
        
        results = []
        for fp in dirp.glob(pattern):
            if not fp.is_file():
                continue
            if fp.suffix.lower() not in supported:
                continue
            
            try:
                # 为每个文件构建源URL（相对路径）
                source_url = str(fp.relative_to(dirp))
                
                result = self.sync_file_with_quality(
                    str(fp), mode, source_type, source_url, metadata
                )
                results.append(result)
                
                # 输出进度
                status = "✅" if not result.skipped else "⏭️"
                quality_display = f" (质量: {result.quality_score:.2f})" if result.quality_score > 0 else ""
                logger.info(f"{status} {fp.name}: {result.inserted}块{quality_display}")
                
            except Exception as e:
                logger.error(f"[KB-Enhanced] 文件同步失败: {fp.name}: {e}")
                results.append(EnhancedSyncResult(
                    doc_id="",
                    mode=mode,
                    inserted=0,
                    deleted=0,
                    skipped=True,
                    reason=f"error: {str(e)}",
                    quality_score=0.0
                ))
        
        # 生成批量处理汇总
        if results:
            self._generate_batch_summary(results, dir_path)
        
        return results
    
    # ----------------------------
    # KB管理功能
    # ----------------------------
    
    def delete_document(self, doc_id: str, soft_delete: bool = True) -> bool:
        """删除文档（支持软删除）"""
        try:
            if soft_delete:
                # 标记为删除而不是物理删除
                # 需要Milvus支持更新操作
                logger.info(f"[KB-Enhanced] 软删除文档: {doc_id}")
                # TODO: 实现软删除逻辑
                return True
            else:
                # 物理删除
                success = self.store.delete_by_doc_id(doc_id)
                if success:
                    # 清理本地状态
                    self._clear_state(doc_id)
                    logger.info(f"[KB-Enhanced] 物理删除文档: {doc_id}")
                return success
        except Exception as e:
            logger.error(f"[KB-Enhanced] 删除文档失败: {doc_id}: {e}")
            return False
    
    def get_document_info(self, doc_id: str) -> Dict[str, Any]:
        """获取文档信息（包含质量和状态）"""
        # 加载本地状态
        state = self._load_state(doc_id)
        
        if not state:
            return {"error": "文档不存在或状态信息丢失"}
        
        # 从Milvus获取chunk统计
        try:
            chunks_info = self.store.get_chunks_by_doc_id(doc_id)
            
            # 计算质量统计
            quality_scores = []
            for chunk in chunks_info:
                metadata = chunk.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                quality_score = metadata.get("quality_score", 1.0)
                quality_scores.append(quality_score)
            
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
            return {
                "doc_id": doc_id,
                "filename": state.get("filename"),
                "file_path": state.get("file_path"),
                "version": state.get("version", 1),
                "doc_hash": state.get("doc_hash"),
                "chunk_count": len(chunks_info),
                "avg_quality": round(avg_quality, 3),
                "last_sync": state.get("last_sync"),
                "source_type": state.get("source_type"),
                "source_url": state.get("source_url"),
                "chunks": chunks_info[:10]  # 只返回前10个chunk
            }
        except Exception as e:
            logger.error(f"[KB-Enhanced] 获取文档信息失败: {doc_id}: {e}")
            return {"error": str(e)}
    
    def list_documents(self, filter_source: str = None) -> List[Dict[str, Any]]:
        """列出所有文档（支持过滤）"""
        documents = []
    
        # 遍历状态文件
        for state_file in self.state_dir.glob("*.json"):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            
                doc_id = state.get("doc_id")
                if not doc_id:
                    continue
            
                # 源类型过滤
                source_type = state.get("source_type", "upload")
                if filter_source and source_type != filter_source:
                    continue
            
                # 确保所有字段都有默认值
                documents.append({
                    "doc_id": doc_id,
                    "filename": state.get("filename", "unknown"),
                    "version": state.get("version", 1),
                    "source_type": source_type,
                    "source_url": state.get("source_url", ""),
                    "last_sync": state.get("last_sync", "1970-01-01T00:00:00"),  # 默认值
                    "chunk_count": state.get("chunk_count", 0),
                    "quality_score": state.get("quality_score", 0.0)
                })
            except Exception as e:
                logger.warning(f"[KB-Enhanced] 读取状态文件失败: {state_file}: {e}")
    
        # 安全排序：确保last_sync不是None
        return sorted(documents, key=lambda x: x.get("last_sync", ""), reverse=True)
    
    def get_quality_reports(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取质量报告列表"""
        reports = []
        
        for report_file in self.quality_report_dir.glob("*.json"):
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                reports.append({
                    "filename": report.get("filename"),
                    "timestamp": report.get("timestamp"),
                    "document_score": report.get("document_metrics", {}).get("doc_score", 0),
                    "avg_chunk_score": report.get("chunk_summary", {}).get("avg_chunk_score", 0),
                    "total_issues": report.get("chunk_summary", {}).get("issue_statistics", {}).get("total_issues", 0),
                    "file_path": str(report_file)
                })
            except Exception as e:
                logger.warning(f"[KB-Enhanced] 读取质量报告失败: {report_file}: {e}")
        
        # 按时间排序并限制数量
        reports.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return reports[:limit]
    
    # ----------------------------
    # 内部方法
    # ----------------------------
    
    def _state_path(self, doc_id: str) -> Path:
        safe = doc_id.replace("/", "_").replace("\\", "_")
        return self.state_dir / f"{safe}.json"
    
    def _load_state(self, doc_id: str) -> Dict[str, Any]:
        p = self._state_path(doc_id)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[KB-Enhanced] 加载状态失败: {doc_id}: {e}")
            return {}
    
    def _save_state(self, doc_id: str, state: Dict[str, Any]):
        p = self._state_path(doc_id)
        p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    
    def _clear_state(self, doc_id: str):
        """清理文档状态"""
        p = self._state_path(doc_id)
        if p.exists():
            try:
                p.unlink()
                logger.info(f"[KB-Enhanced] 清理状态文件: {doc_id}")
            except Exception as e:
                logger.warning(f"[KB-Enhanced] 清理状态文件失败: {doc_id}: {e}")
    
    def _calculate_doc_hash(self, content: str, metadata: Optional[DocumentMetadata] = None) -> str:
        """计算文档哈希（考虑内容和元数据）"""
        hash_input = content
        
        if metadata:
            # 将元数据转换为字符串并加入哈希计算
            meta_str = json.dumps(metadata.to_dict(), sort_keys=True)
            hash_input += meta_str
        
        return hashlib.md5(hash_input.encode("utf-8")).hexdigest()
    
    def _delete_old_chunks(self, doc_id: str) -> int:
        """删除旧chunks并返回删除数量"""
        try:
            # 先查询现有chunks数量
            existing_chunks = self.store.get_chunks_by_doc_id(doc_id)
            
            # 执行删除
            success = self.store.delete_by_doc_id(doc_id)
            
            if success:
                logger.info(f"[KB-Enhanced] 删除旧chunks: {doc_id}, 数量={len(existing_chunks)}")
                return len(existing_chunks)
            else:
                logger.warning(f"[KB-Enhanced] 删除旧chunks失败: {doc_id}")
                return 0
        except Exception as e:
            logger.error(f"[KB-Enhanced] 删除旧chunks异常: {doc_id}: {e}")
            return 0
    
    def _build_section_path(self, metadata: Dict[str, Any]) -> str:
        """构建条文层级路径"""
        path_parts = []
        
        # 添加父级条文
        if "parent_article" in metadata and metadata["parent_article"]:
            path_parts.append(metadata["parent_article"])
        
        # 添加当前条文
        if "article_number" in metadata and metadata["article_number"]:
            path_parts.append(metadata["article_number"])
        
        # 添加标题
        if "article_title" in metadata and metadata["article_title"]:
            path_parts.append(metadata["article_title"])
        
        return " > ".join(path_parts) if path_parts else ""
    
    def _extract_embeddings(self, emb_ret: Any) -> List[List[float]]:
        """提取嵌入向量（兼容多种格式）"""
        if emb_ret is None:
            raise RuntimeError("嵌入返回为空")
        
        # EmbeddingResult对象
        if hasattr(emb_ret, "embeddings"):
            embeddings = getattr(emb_ret, "embeddings")
            if embeddings is None:
                err = getattr(emb_ret, "error", "嵌入向量为空")
                raise RuntimeError(f"嵌入失败: {err}")
            return embeddings
        
        # 字典格式
        if isinstance(emb_ret, dict) and "embeddings" in emb_ret:
            embeddings = emb_ret["embeddings"]
            if embeddings is None:
                raise RuntimeError("嵌入失败: 字典中的嵌入向量为空")
            return embeddings
        
        # 列表格式
        if isinstance(emb_ret, list):
            if len(emb_ret) == 0:
                return []
            if isinstance(emb_ret[0], list):
                return emb_ret
            raise RuntimeError("嵌入返回列表但不是List[List[float]]")
        
        raise RuntimeError(f"不支持的嵌入返回类型: {type(emb_ret)}")
    
    def _insert_with_enhanced_fields(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        version: int,
        updated_at: int,
        source_type: str,
        source_url: str
    ):
        """插入文档（使用增强字段）"""
        try:
            # 尝试使用增强的insert_documents方法（如果存在）
            return self.store.insert_documents(
                chunks, embeddings, 
                doc_version=version, 
                updated_at=updated_at,
                is_deleted=False,
                source_type=source_type,
                source_url=source_url
            )
        except TypeError:
            # 回退到基础方法
            return self.store.insert_documents(chunks, embeddings)
    
    def _generate_batch_summary(self, results: List[EnhancedSyncResult], dir_path: str):
        """生成批量处理汇总报告"""
        successful = [r for r in results if not r.skipped or r.reason != "error"]
        failed = [r for r in results if r.skipped and "error" in r.reason]
        skipped_unchanged = [r for r in results if r.skipped and r.reason == "unchanged"]
        
        summary = {
            "batch_id": hashlib.md5(str(time.time()).encode()).hexdigest()[:12],
            "timestamp": datetime.now().isoformat(),
            "directory": dir_path,
            "total_files": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "skipped_unchanged": len(skipped_unchanged),
            "quality_stats": {
                "avg_doc_quality": sum(r.quality_score for r in successful) / max(len(successful), 1),
                "total_chunks": sum(r.chunk_count for r in successful),
                "avg_chunks_per_doc": sum(r.chunk_count for r in successful) / max(len(successful), 1)
            } if successful else {},
            "failed_files": [
                {"doc_id": r.doc_id, "reason": r.reason} 
                for r in failed
            ]
        }
        
        # 保存汇总报告
        summary_file = self.quality_report_dir / f"batch_summary_{summary['batch_id']}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[KB-Enhanced] 批量处理汇总: {summary_file}")
        
        # 输出统计信息
        logger.info(f"[KB-Enhanced] 批量处理完成: 成功={len(successful)}, 失败={len(failed)}, 跳过={len(skipped_unchanged)}")