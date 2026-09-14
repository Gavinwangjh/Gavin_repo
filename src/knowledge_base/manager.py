# src/knowledge_base/manager.py
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.document_loader.loader import DocumentLoader, Document, DocumentChunk
from src.embedding.embedder import EmbeddingManager
from src.vector_store.milvus_store import MilvusVectorStore
from src.utils.helpers import PerformanceTimer


@dataclass
class SyncResult:
    """一次同步的结果摘要"""
    doc_id: str
    mode: str
    inserted: int
    deleted: int
    skipped: bool
    reason: str = ""
    doc_hash: str = ""
    version: int = 1


class KnowledgeBaseManager:
    """
    知识库管理器（增量更新 + 版本号 + 内容哈希去重）

    目标：
    - 上传同一个文件重复跑不会重复插入（skip）
    - 文件改了：删除旧 doc_id 的块，再插入新块（保持一致）
    - 兼容 EmbeddingManager.embed_documents() 返回：
        - EmbeddingResult(embeddings=...)
        - 或直接返回 List[List[float]]
        - 或 dict: {"embeddings": ...}
    - 兼容 MilvusVectorStore.insert_documents() 参数差异：
        - insert_documents(chunks, embeddings)
        - 或 insert_documents(chunks, embeddings, doc_version=..., updated_at=...)
    """

    def __init__(
        self,
        loader: DocumentLoader,
        embedder: EmbeddingManager,
        store: MilvusVectorStore,
        state_dir: str = "data/kb_state",
    ):
        self.loader = loader
        self.embedder = embedder
        self.store = store
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[KB] state_dir={self.state_dir.resolve()}")

    # ----------------------------
    # Public APIs
    # ----------------------------
    def sync_file(self, file_path: str, mode: str = "incremental") -> SyncResult:
        """
        同步单文件到知识库
        mode:
          - incremental: 增量（推荐）
          - full: 全量重建该 doc（删后插）
        """
        file_path = str(file_path)
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"file not found: {file_path}")

        with PerformanceTimer("KB同步"):
            doc: Document = self.loader.load_document(file_path)
            chunks: List[DocumentChunk] = self.loader.chunk_document(doc)

            if not chunks:
                return SyncResult(
                    doc_id=doc.doc_id,
                    mode=mode,
                    inserted=0,
                    deleted=0,
                    skipped=True,
                    reason="no chunks",
                )

            # 计算 doc hash（用 content）
            doc_hash = self._hash_text(doc.content)
            prev = self._load_state(doc.doc_id)

            # 版本号策略：同 doc_id 下内容变更就 +1
            version = int(prev.get("version", 1)) if prev else 1
            if prev and prev.get("doc_hash") and prev["doc_hash"] != doc_hash:
                version += 1

            # incremental: doc_hash 没变 -> 直接跳过
            if mode == "incremental" and prev and prev.get("doc_hash") == doc_hash:
                logger.info(f"[KB] skip (unchanged): doc_id={doc.doc_id}, version={version}, file={path.name}")
                return SyncResult(
                    doc_id=doc.doc_id,
                    mode=mode,
                    inserted=0,
                    deleted=0,
                    skipped=True,
                    reason="unchanged",
                    doc_hash=doc_hash,
                    version=version,
                )

            # full 或内容变更：删旧块再插（以 doc_id 为单位保持一致）
            deleted = 0
            if prev:
                # 只有在确实变更/全量模式下才删除
                if mode == "full" or prev.get("doc_hash") != doc_hash:
                    ok = self.store.delete_by_doc_id(doc.doc_id)
                    deleted = int(prev.get("chunk_count", 0)) if ok else 0
                    logger.info(f"[KB] delete old: doc_id={doc.doc_id}, deleted~={deleted}, ok={ok}")

            # 写入 chunk 级别的 metadata：版本、更新时间、chunk_hash
            updated_at = int(time.time())
            for c in chunks:
                c.metadata = dict(c.metadata or {})
                c.metadata["doc_version"] = version
                c.metadata["updated_at"] = updated_at
                c.metadata["doc_hash"] = doc_hash
                if "chunk_hash" not in c.metadata:
                    c.metadata["chunk_hash"] = self._hash_text(c.content)

            # 生成 embedding & 插入
            chunks = self._prepare_chunks_for_embedding(chunks)
            texts = [c.content for c in chunks]
            embeddings = self._embed_prepared_texts(texts)

            insert_result = self._insert_documents_compat(chunks, embeddings, version, updated_at)
            if not getattr(insert_result, "success", True):
                err = getattr(insert_result, "error", "unknown insert error")
                raise RuntimeError(f"[KB] milvus insert failed: {err}")

            inserted = int(getattr(insert_result, "insert_count", len(chunks)))

            # 保存本地状态（用于增量判断）
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
                },
            )

            logger.info(f"[KB] {mode} sync: doc_id={doc.doc_id}, chunks={len(chunks)}, inserted={inserted}, version={version}")
            return SyncResult(
                doc_id=doc.doc_id,
                mode=mode,
                inserted=inserted,
                deleted=deleted,
                skipped=False,
                reason="synced",
                doc_hash=doc_hash,
                version=version,
            )

    def sync_dir(self, dir_path: str, recursive: bool = False, mode: str = "incremental") -> List[SyncResult]:
        """同步目录（批量）"""
        dirp = Path(dir_path)
        if not dirp.exists() or not dirp.is_dir():
            raise FileNotFoundError(f"dir not found: {dir_path}")

        supported = getattr(self.loader, "SUPPORTED_FORMATS", {".pdf", ".txt", ".md", ".docx"})
        pattern = "**/*" if recursive else "*"

        results: List[SyncResult] = []
        for fp in dirp.glob(pattern):
            if not fp.is_file():
                continue
            if fp.suffix.lower() not in supported:
                continue
            try:
                results.append(self.sync_file(str(fp), mode=mode))
            except Exception as e:
                logger.exception(f"[KB] sync failed: {fp.name}: {e}")
                results.append(SyncResult(doc_id="", mode=mode, inserted=0, deleted=0, skipped=True, reason=f"error: {e}"))
        return results

    def clear_state(self):
        """仅清空本地 state（不删 Milvus 数据）"""
        if self.state_dir.exists():
            for p in self.state_dir.glob("*.json"):
                try:
                    p.unlink()
                except Exception:
                    pass
        logger.warning("[KB] local state cleared (milvus data unchanged)")

    # ----------------------------
    # Internals
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
        except Exception:
            return {}

    def _save_state(self, doc_id: str, state: Dict[str, Any]):
        p = self._state_path(doc_id)
        p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.md5((text or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _extract_embeddings(emb_ret: Any) -> List[List[float]]:
        """
        兼容：
        - EmbeddingResult.embeddings
        - dict["embeddings"]
        - 直接就是 List[List[float]]
        """
        if emb_ret is None:
            raise RuntimeError("embedding returned None")

        # EmbeddingResult
        if hasattr(emb_ret, "embeddings"):
            embeddings = getattr(emb_ret, "embeddings")
            if embeddings is None:
                err = getattr(emb_ret, "error", "embedding embeddings is None")
                raise RuntimeError(f"embedding failed: {err}")
            return embeddings

        # dict
        if isinstance(emb_ret, dict) and "embeddings" in emb_ret:
            embeddings = emb_ret["embeddings"]
            if embeddings is None:
                raise RuntimeError("embedding failed: dict embeddings is None")
            return embeddings

        # list
        if isinstance(emb_ret, list):
            # 粗校验：二维 list
            if len(emb_ret) == 0:
                return []
            if isinstance(emb_ret[0], list):
                return emb_ret  # type: ignore
            raise RuntimeError("embedding returned list but not List[List[float]]")

        raise RuntimeError(f"unsupported embedding return type: {type(emb_ret)}")

    def _prepare_chunks_for_embedding(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Split oversized chunks before embedding so chunks and vectors stay aligned.
        """
        embedder = getattr(self.embedder, "embedder", None)
        text_processor = getattr(self.embedder, "text_processor", None)
        max_input_tokens = getattr(embedder, "max_input_tokens", None)

        if not embedder or not text_processor or not isinstance(max_input_tokens, int):
            return chunks

        prepared: List[DocumentChunk] = []
        max_tokens = max(1, min(max_input_tokens - 8, 320))
        tokenizer = getattr(embedder, "tokenizer", None)

        for chunk in chunks:
            cleaned = text_processor.clean_text(chunk.content)
            if not cleaned:
                continue

            parts = text_processor.split_long_text(
                cleaned,
                max_tokens=max_tokens,
                tokenizer=tokenizer,
            )

            for split_index, part in enumerate(parts):
                metadata = dict(chunk.metadata or {})
                metadata.update({
                    "original_chunk_index": chunk.chunk_index,
                    "split_index": split_index,
                    "split_count": len(parts),
                    "chunk_length": len(part),
                })

                prepared.append(DocumentChunk(
                    content=part,
                    metadata=metadata,
                    chunk_id="",
                    doc_id=chunk.doc_id,
                    chunk_index=len(prepared),
                ))

        return prepared

    def _embed_prepared_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed already split texts without another document-level split pass."""
        raw_embedder = getattr(self.embedder, "embedder", None)
        if raw_embedder and callable(getattr(raw_embedder, "embed_texts", None)):
            return self._extract_embeddings(raw_embedder.embed_texts(texts))
        return self._extract_embeddings(self.embedder.embed_documents(texts))

    def _insert_documents_compat(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        version: int,
        updated_at: int,
    ):
        """
        兼容 MilvusVectorStore.insert_documents 的不同签名：
        - insert_documents(chunks, embeddings)
        - insert_documents(chunks, embeddings, doc_version=..., updated_at=...)
        """
        try:
            return self.store.insert_documents(chunks, embeddings, doc_version=version, updated_at=updated_at)  # type: ignore
        except TypeError:
            # 你的 store 还是旧签名
            return self.store.insert_documents(chunks, embeddings)
