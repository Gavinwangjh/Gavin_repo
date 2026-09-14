from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import tempfile
import os
import json
import time
from pathlib import Path
import asyncio
from importlib.util import find_spec

from loguru import logger

from src.document_loader.loader import DocumentLoader, DocumentChunk
from src.embedding.embedder import EmbeddingManager
from src.vector_store.milvus_store import MilvusVectorStore
from src.retrieval.retriever import HybridRetriever
from src.generation.generator import RAGGenerator, GenerationResult, ChatMessage
from src.utils.helpers import get_config, PerformanceTimer


# Pydantic模型定义
class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    top_k: int = Field(5, description="检索文档数量", ge=1, le=20)
    method: str = Field("hybrid_rrf", description="检索方法", pattern="^(dense|sparse|hybrid|hybrid_rrf)$")
    stream: bool = Field(False, description="是否流式返回")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_time: float
    generation_time: float
    total_time: float
    method: str


class ChatRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    history: List[Dict[str, str]] = Field([], description="聊天历史")
    stream: bool = Field(False, description="是否流式返回")


class DocumentInfo(BaseModel):
    filename: str
    file_size: int
    file_type: str
    chunk_count: int
    doc_id: str
    upload_time: str


class UploadResponse(BaseModel):
    success: bool
    message: str
    document: Optional[DocumentInfo] = None
    chunk_count: int = 0


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, Any]
    timestamp: str


class StatsResponse(BaseModel):
    milvus_stats: Dict[str, Any]
    retriever_stats: Dict[str, Any]
    generator_stats: Dict[str, Any]


# 全局变量（生产环境建议使用依赖注入）
DEFAULT_APP_STATE = {
    "document_loader": None,
    "embedding_manager": None,
    "vector_store": None,
    "retriever": None,
    "generator": None,
    "initialized": False
}
app_state = DEFAULT_APP_STATE.copy()

UPLOADS_ENABLED = find_spec("multipart") is not None


def ensure_app_state() -> None:
    for key, value in DEFAULT_APP_STATE.items():
        app_state.setdefault(key, value)


def get_document_loader() -> DocumentLoader:
    """获取文档加载器"""
    ensure_app_state()
    if not app_state.get("initialized"):
        raise HTTPException(status_code=500, detail="系统未就绪")
    if app_state.get("document_loader") is None:
        chunk_size = get_config("document.chunk_size", 512)
        chunk_overlap = get_config("document.chunk_overlap", 50)
        app_state["document_loader"] = DocumentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    return app_state["document_loader"]


def get_embedding_manager() -> EmbeddingManager:
    """获取嵌入管理器"""
    ensure_app_state()
    if not app_state.get("initialized"):
        raise HTTPException(status_code=500, detail="系统未就绪")
    if app_state.get("embedding_manager") is None:
        app_state["embedding_manager"] = EmbeddingManager(provider="siliconflow")
    return app_state["embedding_manager"]


def get_vector_store() -> MilvusVectorStore:
    """获取向量存储"""
    ensure_app_state()
    if not app_state.get("initialized"):
        raise HTTPException(status_code=500, detail="系统未就绪")
    if app_state.get("vector_store") is None:
        app_state["vector_store"] = MilvusVectorStore()
        # 初始化连接
        if not app_state["vector_store"].initialize():
            raise HTTPException(status_code=500, detail="向量数据库连接失败")
    return app_state["vector_store"]


def get_retriever() -> HybridRetriever:
    """获取检索器"""
    ensure_app_state()
    if not app_state.get("initialized"):
        raise HTTPException(status_code=500, detail="系统未就绪")
    if app_state.get("retriever") is None:
        vector_store = get_vector_store()
        embedding_manager = get_embedding_manager()
        app_state["retriever"] = HybridRetriever(
            vector_store=vector_store,
            embedding_manager=embedding_manager
        )
    return app_state["retriever"]


def get_generator() -> RAGGenerator:
    """获取生成器"""
    ensure_app_state()
    if not app_state.get("initialized"):
        raise HTTPException(status_code=500, detail="系统未就绪")
    if app_state.get("generator") is None:
        app_state["generator"] = RAGGenerator()
    return app_state["generator"]


def prepare_chunks_for_embedding(
    chunks: List[DocumentChunk],
    embedding_manager: EmbeddingManager
) -> List[DocumentChunk]:
    """Split oversized chunks before embedding so chunks and vectors stay aligned."""
    embedder = getattr(embedding_manager, "embedder", None)
    text_processor = getattr(embedding_manager, "text_processor", None)
    max_input_tokens = getattr(embedder, "max_input_tokens", None)

    if not embedder or not text_processor or not isinstance(max_input_tokens, int):
        return chunks

    prepared: List[DocumentChunk] = []
    max_tokens = max(1, max_input_tokens - 8)
    tokenizer = getattr(embedder, "tokenizer", None)

    for chunk in chunks:
        cleaned = embedding_manager.text_processor.clean_text(chunk.content)
        if not cleaned:
            continue

        parts = embedding_manager.text_processor.split_long_text(
            cleaned,
            max_tokens=max_tokens,
            tokenizer=tokenizer
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


# FastAPI应用初始化
def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app_config = get_config("api", {})
    
    app = FastAPI(
        title=app_config.get("title", "LLM RAG API"),
        description=app_config.get("description", "智能问答系统API"),
        version=app_config.get("version", "1.0.0"),
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应限制具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    try:
        logger.info("API服务启动中...")
        
        # External services are initialized lazily by request dependencies.
        ensure_app_state()
        app_state["initialized"] = True
        logger.info("API服务启动完成")
        
    except Exception as e:
        logger.error(f"API服务启动失败: {str(e)}")
        app_state["initialized"] = False


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    try:
        ensure_app_state()
        if app_state.get("vector_store"):
            app_state["vector_store"].disconnect()
        logger.info("API服务已关闭")
    except Exception as e:
        logger.error(f"API服务关闭时出错: {str(e)}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径"""
    return {
        "message": "LLM RAG API",
        "version": "1.0.0",
        "app": "/app",
        "docs": "/docs"
    }


@app.get("/app", response_class=HTMLResponse)
async def rag_app():
    """RAG system web UI."""
    app_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(app_path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    from datetime import datetime
    
    ensure_app_state()
    status = "healthy" if app_state.get("initialized") else "unhealthy"
    components = {}
    
    try:
        if app_state.get("vector_store"):
            components["milvus"] = app_state["vector_store"].health_check()
        else:
            components["milvus"] = {"status": "not_initialized"}
        
        components["embedding"] = {"status": "ok"} if app_state.get("embedding_manager") else {"status": "not_initialized"}
        components["generator"] = {"status": "ok"} if app_state.get("generator") else {"status": "not_initialized"}
        
    except Exception as e:
        components["error"] = str(e)
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        components=components,
        timestamp=datetime.now().isoformat()
    )


if not UPLOADS_ENABLED:
    logger.warning("python-multipart is not installed; document upload endpoint is disabled")

    @app.post("/documents/upload", response_model=UploadResponse)
    async def upload_document_unavailable():
        raise HTTPException(
            status_code=503,
            detail="Document upload requires python-multipart. Install it with: pip install python-multipart"
        )


upload_document_route = (
    app.post("/documents/upload", response_model=UploadResponse)
    if UPLOADS_ENABLED
    else (lambda func: func)
)


@upload_document_route
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    loader: DocumentLoader = Depends(get_document_loader),
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
    vector_store: MilvusVectorStore = Depends(get_vector_store)
):
    """上传文档"""
    try:
        # 检查文件大小
        max_size = get_config("document.max_file_size", 10485760)  # 10MB
        
        # 检查文件类型
        file_ext = Path(file.filename).suffix.lower()
        supported_formats = get_config("document.supported_formats", [".pdf", ".txt", ".md"])
        
        if file_ext not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {file_ext}. 支持的格式: {supported_formats}"
            )
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            
            if len(content) > max_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件大小超限: {len(content)} bytes > {max_size} bytes"
                )
            
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # 加载并分块文档
            with PerformanceTimer("文档处理"):
                chunks = loader.load_and_chunk_document(tmp_file_path)
                
                if not chunks:
                    raise HTTPException(status_code=400, detail="文档内容为空或无法解析")
                
                # 生成嵌入
                indexed_chunks = prepare_chunks_for_embedding(chunks, embedding_manager)
                if not indexed_chunks:
                    raise HTTPException(status_code=400, detail="文档内容为空或无法解析")

                texts = [chunk.content for chunk in indexed_chunks]
                if callable(getattr(embedding_manager, "embed_documents", None)):
                    embedding_result = embedding_manager.embed_documents(texts)
                else:
                    embedding_result = embedding_manager.embedder.embed_texts(texts)

                if len(indexed_chunks) != len(embedding_result.embeddings):
                    raise HTTPException(
                        status_code=500,
                        detail="文档索引失败: chunk 数量与嵌入向量数量不一致"
                    )
                
                # 插入向量数据库
                insert_result = vector_store.insert_documents(indexed_chunks, embedding_result.embeddings)
                
                if not insert_result.success:
                    raise HTTPException(status_code=500, detail=f"文档索引失败: {insert_result.error}")
            
            # 更新稀疏检索索引（后台任务）
            retriever = get_retriever()
            background_tasks.add_task(retriever.build_sparse_index, True)
            
            # 构建响应
            doc_info = DocumentInfo(
                filename=file.filename,
                file_size=len(content),
                file_type=file_ext,
                chunk_count=len(indexed_chunks),
                doc_id=indexed_chunks[0].doc_id if indexed_chunks else "",
                upload_time=str(time.time())
            )
            
            return UploadResponse(
                success=True,
                message=f"文档上传成功: {file.filename}",
                document=doc_info,
                chunk_count=len(indexed_chunks)
            )
            
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest
):
    """查询文档"""
    try:
        retriever = get_retriever()
        generator = get_generator()
        with PerformanceTimer(f"问答查询: {request.question[:50]}..."):
            # 检索相关文档
            retrieval_result = retriever.search(
                query=request.question,
                top_k=request.top_k,
                method=request.method
            )
            
            # 生成回答
            generation_result = generator.generate_answer(
                question=request.question,
                retrieval_result=retrieval_result,
                stream=request.stream
            )
            
            # 构建响应
            sources = []
            for hit in generation_result.sources:
                sources.append({
                    "id": hit.id,
                    "content": hit.content,
                    "score": hit.score,
                    "metadata": hit.metadata,
                    "doc_id": hit.doc_id,
                    "chunk_index": hit.chunk_index
                })
            
            return QueryResponse(
                question=generation_result.question,
                answer=generation_result.answer,
                sources=sources,
                retrieval_time=retrieval_result.retrieval_time,
                generation_time=generation_result.generation_time,
                total_time=generation_result.total_time,
                method=request.method
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.post("/chat")
async def chat(
    request: ChatRequest
):
    """聊天对话（不使用RAG）"""
    try:
        generator = get_generator()
        # 转换聊天历史
        chat_history = []
        for msg in request.history:
            chat_history.append(ChatMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", "")
            ))
        
        # 生成回答
        answer = generator.chat(
            question=request.question,
            chat_history=chat_history
        )
        
        return {"answer": answer}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"聊天失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"聊天失败: {str(e)}")


@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    vector_store: MilvusVectorStore = Depends(get_vector_store)
):
    """删除文档"""
    try:
        success = vector_store.delete_by_doc_id(doc_id)
        
        if success:
            # 删除成功后重建稀疏索引以保持数据一致性
            try:
                retriever = get_retriever()
                retriever.build_sparse_index(force_rebuild=True)
                logger.info("删除文档后稀疏索引重建完成")
            except Exception as rebuild_e:
                logger.warning(f"稀疏索引重建失败: {str(rebuild_e)}")
            
            return {"message": f"文档删除成功: {doc_id}"}
        else:
            raise HTTPException(status_code=404, detail=f"文档不存在或删除失败: {doc_id}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats(
    vector_store: MilvusVectorStore = Depends(get_vector_store),
    retriever: HybridRetriever = Depends(get_retriever),
    generator: RAGGenerator = Depends(get_generator)
):
    """获取系统统计信息"""
    try:
        return StatsResponse(
            milvus_stats=vector_store.get_collection_stats(),
            retriever_stats=retriever.get_stats(),
            generator_stats=generator.get_stats()
        )
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


# 知识库管理API
@app.get("/knowledge/documents")
async def list_documents(
    vector_store: MilvusVectorStore = Depends(get_vector_store)
):
    """获取文档列表"""
    try:
        # 查询所有文档的doc_id和基础信息
        if not vector_store.collection:
            raise HTTPException(status_code=500, detail="集合未初始化")
        
        # 使用聚合查询获取文档列表
        docs = []
        seen_docs = set()
        
        # 简单查询所有记录，然后去重
        search_results = vector_store.collection.query(
            expr="doc_id != ''",
            output_fields=["doc_id", "content", "metadata", "chunk_index"],
            limit=10000
        )
        
        for result in search_results:
            doc_id = result.get('doc_id')
            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                
                # 解析metadata
                import json
                try:
                    metadata = json.loads(result.get('metadata', '{}'))
                except:
                    metadata = {}
                
                docs.append({
                    "doc_id": doc_id,
                    "filename": metadata.get('filename', '未知文件'),
                    "file_type": metadata.get('file_type', 'unknown'),
                    "file_size": metadata.get('file_size', 0),
                    "created_time": metadata.get('created_time', 0),
                    "modified_time": metadata.get('modified_time', 0),
                    "preview": result.get('content', '')[:200] + '...' if len(result.get('content', '')) > 200 else result.get('content', '')
                })
        
        return {"documents": docs, "total": len(docs)}
        
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@app.get("/knowledge/documents/{doc_id}")
async def get_document_detail(
    doc_id: str,
    vector_store: MilvusVectorStore = Depends(get_vector_store)
):
    """获取文档详细信息"""
    try:
        if not vector_store.collection:
            raise HTTPException(status_code=500, detail="集合未初始化")
        
        # 查询该文档的所有块
        chunks = vector_store.collection.query(
            expr=f'doc_id == "{doc_id}"',
            output_fields=["id", "content", "metadata", "chunk_index"],
            limit=1000
        )
        
        if not chunks:
            raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
        
        # 按chunk_index排序
        chunks.sort(key=lambda x: x.get('chunk_index', 0))
        
        # 解析第一个块的metadata作为文档metadata
        import json
        try:
            doc_metadata = json.loads(chunks[0].get('metadata', '{}'))
        except:
            doc_metadata = {}
        
        # 构建响应
        doc_info = {
            "doc_id": doc_id,
            "filename": doc_metadata.get('filename', '未知文件'),
            "file_type": doc_metadata.get('file_type', 'unknown'),
            "file_size": doc_metadata.get('file_size', 0),
            "created_time": doc_metadata.get('created_time', 0),
            "modified_time": doc_metadata.get('modified_time', 0),
            "chunk_count": len(chunks),
            "chunks": []
        }
        
        for chunk in chunks:
            try:
                chunk_metadata = json.loads(chunk.get('metadata', '{}'))
            except:
                chunk_metadata = {}
                
            doc_info["chunks"].append({
                "id": chunk.get('id'),
                "content": chunk.get('content', ''),
                "chunk_index": chunk.get('chunk_index', 0),
                "chunk_length": chunk_metadata.get('chunk_length', 0)
            })
        
        return doc_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文档详情失败: {str(e)}")


@app.get("/knowledge/search")
async def search_documents(
    q: str,
    limit: int = 10,
    vector_store: MilvusVectorStore = Depends(get_vector_store)
):
    """搜索文档内容"""
    try:
        if not vector_store.collection:
            raise HTTPException(status_code=500, detail="集合未初始化")
        
        # 简单的文本匹配搜索
        results = vector_store.collection.query(
            expr=f'content like "%{q}%"',
            output_fields=["id", "doc_id", "content", "metadata", "chunk_index"],
            limit=limit
        )
        
        search_results = []
        for result in results:
            try:
                import json
                metadata = json.loads(result.get('metadata', '{}'))
                
                search_results.append({
                    "id": result.get('id'),
                    "doc_id": result.get('doc_id'),
                    "filename": metadata.get('filename', '未知文件'),
                    "content": result.get('content', ''),
                    "chunk_index": result.get('chunk_index', 0),
                    "highlight": result.get('content', '').replace(q, f"**{q}**")
                })
            except:
                continue
        
        return {"results": search_results, "total": len(search_results), "query": q}
        
    except Exception as e:
        logger.error(f"搜索文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索文档失败: {str(e)}")


@app.get("/knowledge/stats")
async def get_knowledge_stats(
    vector_store: MilvusVectorStore = Depends(get_vector_store)
):
    """获取知识库统计信息"""
    try:
        if not vector_store.collection:
            raise HTTPException(status_code=500, detail="集合未初始化")
        
        # 获取基础统计
        collection_stats = vector_store.get_collection_stats()
        
        # 获取文档数量统计
        all_chunks = vector_store.collection.query(
            expr="doc_id != ''",
            output_fields=["doc_id", "metadata"],
            limit=10000
        )
        
        doc_count = len(set(chunk.get('doc_id') for chunk in all_chunks))
        
        # 文件类型统计
        file_types = {}
        for chunk in all_chunks:
            try:
                import json
                metadata = json.loads(chunk.get('metadata', '{}'))
                file_type = metadata.get('file_type', 'unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1
            except:
                continue
        
        return {
            "total_documents": doc_count,
            "total_chunks": collection_stats.get("entity_count", 0),
            "file_types": file_types,
            "collection_name": collection_stats.get("collection_name"),
            "dimension": collection_stats.get("dimension"),
            "index_type": collection_stats.get("index_type")
        }
        
    except Exception as e:
        logger.error(f"获取知识库统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取知识库统计失败: {str(e)}")


@app.post("/index/rebuild")
async def rebuild_index(
    background_tasks: BackgroundTasks,
    retriever: HybridRetriever = Depends(get_retriever)
):
    """重建索引"""
    try:
        # 后台重建稀疏索引
        background_tasks.add_task(retriever.build_sparse_index, True)
        
        return {"message": "索引重建任务已启动"}
        
    except Exception as e:
        logger.error(f"重建索引失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重建索引失败: {str(e)}")


# 流式响应端点（如果需要）
@app.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    generator: RAGGenerator = Depends(get_generator)
):
    """流式查询（实验性功能）"""
    async def generate_stream():
        try:
            # 检索
            retrieval_result = retriever.search(
                query=request.question,
                top_k=request.top_k,
                method=request.method
            )
            
            # 这里简化处理，实际需要实现真正的流式生成
            generation_result = generator.generate_answer(
                question=request.question,
                retrieval_result=retrieval_result,
                stream=False
            )
            
            # 模拟流式输出
            words = generation_result.answer.split()
            for word in words:
                yield f"data: {json.dumps({'content': word + ' '})}\n\n"
                await asyncio.sleep(0.1)
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # 配置日志
    from src.utils.helpers import Logger
    Logger.setup_logger(
        log_level=get_config("logging.level", "INFO"),
        log_format=get_config("logging.format"),
        log_file="logs/api.log"
    )
    
    # 启动服务
    host = get_config("api.host", "0.0.0.0")
    port = get_config("api.port", 8000)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        access_log=True
    )
