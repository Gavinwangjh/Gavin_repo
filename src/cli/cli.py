import click
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger

from src.document_loader.loader import DocumentLoader, DocumentChunk
from src.embedding.embedder import EmbeddingManager
from src.vector_store.milvus_store import MilvusVectorStore
from src.retrieval.retriever import HybridRetriever
from src.generation.generator import RAGGenerator
from src.utils.helpers import get_config, Logger, PerformanceTimer

from src.scrapers.nsfc_scraper import NSFCScraper
from src.scrapers.data_processor import process_nsfc_data

# 知识库管理器
from src.knowledge_base.manager import KnowledgeBaseManager, SyncResult


class RAGSystem:
    """RAG系统管理类"""

    def __init__(self):
        self.document_loader: Optional[DocumentLoader] = None
        self.embedding_manager: Optional[EmbeddingManager] = None
        self.vector_store: Optional[MilvusVectorStore] = None
        self.retriever: Optional[HybridRetriever] = None
        self.generator: Optional[RAGGenerator] = None
        self.kb_manager: Optional[KnowledgeBaseManager] = None
        self.initialized: bool = False

    def initialize(self, force_reconnect: bool = False) -> bool:
        """初始化系统组件"""
        try:
            if self.initialized and not force_reconnect:
                return True

            click.echo("🚀 初始化RAG系统...")

            # 1) 文档加载器
            chunk_size = get_config("document.chunk_size", 512)
            chunk_overlap = get_config("document.chunk_overlap", 50)
            self.document_loader = DocumentLoader(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            click.echo("✅ 文档加载器初始化完成")

            # 2) 嵌入管理器
            self.embedding_manager = EmbeddingManager(provider="siliconflow")
            click.echo("✅ 嵌入管理器初始化完成")

            # 3) 向量存储（Milvus）
            self.vector_store = MilvusVectorStore()
            if not self.vector_store.initialize():
                raise Exception("Milvus连接失败")
            click.echo("✅ Milvus向量存储初始化完成")

            # 4) 检索器 - 使用新的配置方式
            try:
                from src.retrieval.retriever import HybridRetriever, HybridRetrieverConfig
                
                # 创建配置对象
                config = HybridRetrieverConfig(
                    dense_weight=0.7,
                    sparse_weight=0.3,
                    initial_top_k=50,
                    final_top_k=5,
                    use_rrf=True,
                    use_structured_boost=True
                )
                
                # 使用新的调用方式
                self.retriever = HybridRetriever(
                    vector_store=self.vector_store,
                    embedding_manager=self.embedding_manager,
                    config=config  # ✅ 使用config参数
                )
                click.echo("✅ 增强版混合检索器初始化完成")
                
            except Exception as e:
                click.echo(f"⚠️ 检索器初始化异常: {str(e)}，创建简单版本")
                class SimpleRetriever:
                    def search(self, query, top_k=5, method="hybrid"):
                        from src.retrieval.retriever import RetrievalResult
                        return RetrievalResult(
                            query=query,
                            hits=[],
                            dense_hits=[],
                            sparse_hits=[],
                            total_hits=0,
                            retrieval_time=0.0,
                            method=method
                        )
                    def build_sparse_index(self, force_rebuild=False):
                        pass
                    def update_weights(self, *args, **kwargs):
                        pass
                    def get_stats(self):
                        return {}
                
                self.retriever = SimpleRetriever()
                click.echo("✅ 简单检索器初始化完成")

            # 5) 知识库管理器
            try:
                from src.knowledge_base.manager import KnowledgeBaseManager
                self.kb_manager = KnowledgeBaseManager(
                    loader=self.document_loader,
                    embedder=self.embedding_manager,
                    store=self.vector_store
                )
                click.echo("✅ 知识库管理器初始化完成")
            except ImportError as e:
                click.echo(f"⚠️ 无法导入知识库管理器: {str(e)}")
                self.kb_manager = None

            # 6) 生成器
            self.generator = RAGGenerator()
            click.echo("✅ RAG生成器初始化完成")

            self.initialized = True
            click.echo("🎉 系统初始化完成!")
            return True

        except Exception as e:
            click.echo(f"❌ 系统初始化失败: {str(e)}", err=True)
            logger.exception("RAG系统初始化异常")
            return False

    def health_check(self) -> Dict[str, object]:
        """健康检查"""
        if not self.initialized:
            return {"status": "未初始化"}

        health_info = {
            "系统状态": "正常" if self.initialized else "异常",
            "Milvus连接": "正常" if getattr(self.vector_store, "is_connected", False) else "异常",
        }

        try:
            stats = self.vector_store.get_collection_stats() if self.vector_store else {}
            health_info["文档数量"] = stats.get("entity_count", 0)
        except Exception as e:
            logger.warning(f"获取集合统计失败: {str(e)}")
            health_info["文档数量"] = "未知"

        return health_info


# 全局RAG系统实例
rag_system = RAGSystem()


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='配置文件路径')
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
@click.pass_context
def cli(ctx, config, verbose):
    """LLM RAG 智能问答系统命令行工具"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

    # 设置日志级别
    log_level = "DEBUG" if verbose else "INFO"
    Logger.setup_logger(log_level=log_level)

    if config:
        click.echo(f"使用配置文件: {config}")


@cli.command()
@click.pass_context
def init(ctx):
    """初始化系统"""
    success = rag_system.initialize(force_reconnect=True)
    ctx.exit(0 if success else 1)


@cli.command()
@click.pass_context
def health(ctx):
    """系统健康检查"""
    health_info = rag_system.health_check()
    
    click.echo("🔍 系统健康检查:")
    click.echo("=" * 50)
    for key, value in health_info.items():
        click.echo(f"  {key}: {value}")
    click.echo("=" * 50)


def _print_sync_result(res: SyncResult):
    """统一打印 SyncResult（避免字段不一致导致崩）"""
    click.echo(f"✅ doc_id={getattr(res, 'doc_id', '')}")
    inserted = getattr(res, 'inserted', 0)
    deleted = getattr(res, 'deleted', 0)
    skipped = getattr(res, 'skipped', False)
    reason = getattr(res, 'reason', '') or ''
    version = getattr(res, 'version', None)

    click.echo(f"  ➕ inserted: {inserted}")
    click.echo(f"  ➖ deleted:  {deleted}")
    if skipped:
        click.echo(f"  ⏭️ skipped:  {reason if reason else 'unchanged'}")
    else:
        if version is not None:
            click.echo(f"  🔖 version:  {version}")


def _upload_single_file(file_path: Path, mode: str = "incremental") -> bool:
    """上传单个文件（KB sync）"""
    click.echo(f"📤 上传文件: {file_path.name}  (mode={mode})")
    try:
        if rag_system.kb_manager is None:
            raise RuntimeError("知识库管理器未初始化")

        with PerformanceTimer("KB同步"):
            res: SyncResult = rag_system.kb_manager.sync_file(str(file_path), mode=mode)

        _print_sync_result(res)
        
        # 修正返回值逻辑：跳过(skipped)也被认为是操作成功
        return True

    except Exception as e:
        click.echo(f"❌ 文件失败 {file_path.name}: {str(e)}", err=True)
        logger.exception(f"文件上传异常: {file_path}")
        return False


def _upload_directory(dir_path: Path, recursive: bool, mode: str = "incremental") -> bool:
    """上传目录中的文档（KB sync 批量） - 修正版本"""
    click.echo(f"📁 处理目录: {dir_path} (mode={mode})")

    supported_formats = set(get_config("document.supported_formats", [".pdf", ".txt", ".md", ".docx"]))
    pattern = "**/*" if recursive else "*"

    files: List[Path] = []
    for fp in dir_path.glob(pattern):
        if fp.is_file() and fp.suffix.lower() in supported_formats:
            files.append(fp)

    if not files:
        click.echo("⚠️ 未找到支持的文档文件")
        return True

    click.echo(f"📄 找到 {len(files)} 个文件")

    success_count = 0
    skip_count = 0
    fail_count = 0
    
    with click.progressbar(files, label="处理文件", show_eta=True, show_percent=True) as bar:
        for fp in bar:
            try:
                success = _upload_single_file(fp, mode=mode)
                if success:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                click.echo(f"\n❌ 文件处理异常 {fp.name}: {str(e)}", err=True)
                fail_count += 1

    # 详细统计输出
    click.echo(f"\n📊 批量上传完成:")
    click.echo(f"  ✅ 成功/跳过: {success_count} 个")
    click.echo(f"  ❌ 失败: {fail_count} 个")
    click.echo(f"  📄 总计: {len(files)} 个文件")

    # ✅ 目录处理后重建稀疏索引（建议）
    if success_count > 0:
        try:
            click.echo("🔄 重建检索索引...")
            rag_system.retriever.build_sparse_index(force_rebuild=True)
            click.echo("✅ 索引重建完成")
        except Exception as e:
            click.echo(f"⚠️ 索引重建失败（不影响入库成功）: {str(e)}")
            logger.warning(f"索引重建失败: {str(e)}")

    # 修正：只要有文件成功处理，就返回True（允许跳过文件）
    return fail_count == 0


@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--recursive', '-r', is_flag=True, help='递归处理子目录')
@click.option('--mode', '-m', default='incremental', 
              type=click.Choice(['incremental', 'overwrite', 'skip']), 
              help='同步模式: incremental=增量, overwrite=覆盖, skip=跳过')
@click.pass_context
def upload(ctx, path, recursive, mode):
    """上传文档或目录到知识库"""
    if not rag_system.initialize():
        click.echo("❌ 系统未就绪", err=True)
        ctx.exit(1)
    
    try:
        from pathlib import Path
        
        path_obj = Path(path)
        
        if path_obj.is_file():
            # 上传单个文件
            success = _upload_single_file(path_obj, mode)
            if not success:
                ctx.exit(1)
        
        elif path_obj.is_dir():
            # 上传目录
            success = _upload_directory(path_obj, recursive, mode)
            if not success:
                ctx.exit(1)
        
        else:
            click.echo(f"❌ 路径不存在或无法访问: {path}", err=True)
            ctx.exit(1)
        
        click.echo("🎉 上传完成!")
        
    except Exception as e:
        click.echo(f"❌ 上传失败: {str(e)}", err=True)
        logger.exception("上传文档异常")
        ctx.exit(1)


def _output_simple_result(result):
    click.echo(f"\n💬 问题: {result.question}")
    click.echo(f"🤖 回答: {result.answer}")
    click.echo(f"⏱️  耗时: {result.total_time:.2f}秒")
    if result.sources:
        click.echo(f"📚 参考文档: {len(result.sources)} 个")


def _output_detailed_result(result):
    click.echo(f"\n" + "=" * 60)
    click.echo(f"💬 问题: {result.question}")
    click.echo(f"🤖 回答: {result.answer}")
    click.echo(f"⏱️  总耗时: {result.total_time:.2f}秒")
    click.echo(f"🔍 检索耗时: {result.retrieval_result.retrieval_time:.2f}秒")
    click.echo(f"🎯 生成耗时: {result.generation_time:.2f}秒")
    click.echo(f"📊 检索方法: {result.retrieval_result.method}")

    if result.sources:
        click.echo(f"\n📚 参考文档 ({len(result.sources)} 个):")
        for i, source in enumerate(result.sources, 1):
            click.echo(f"\n  📄 文档 {i} (相似度: {source.score:.3f})")
            filename = source.metadata.get('filename', '未知文件')
            click.echo(f"     📁 文件: {filename}")
            click.echo(f"     📝 内容: {source.content[:200]}...")
    else:
        click.echo("\n⚠️ 未找到相关文档")


def _output_json_result(result):
    output_data = {
        "question": result.question,
        "answer": result.answer,
        "total_time": result.total_time,
        "generation_time": result.generation_time,
        "retrieval_time": result.retrieval_result.retrieval_time if result.retrieval_result else 0,
        "method": result.retrieval_result.method if result.retrieval_result else "unknown",
        "sources": [
            {
                "id": source.id,
                "content": source.content,
                "score": source.score,
                "metadata": source.metadata,
                "doc_id": source.doc_id,
                "chunk_index": source.chunk_index
            }
            for source in result.sources
        ]
    }
    click.echo(json.dumps(output_data, ensure_ascii=False, indent=2))


@cli.command()
@click.argument('question')
@click.option('--top-k', '-k', default=5, show_default=True, help='检索文档数量')
@click.option('--method', '-m', default='hybrid_rrf', 
              type=click.Choice(['dense', 'sparse', 'hybrid', 'hybrid_rrf']),
              show_default=True, help='检索方法')
@click.option('--output', '-o', type=click.Choice(['simple', 'detailed', 'json']), 
              default='simple', show_default=True, help='输出格式')
@click.pass_context
def query(ctx, question, top_k, method, output):
    """查询问答"""
    if not rag_system.initialize():
        click.echo("❌ 系统未就绪", err=True)
        ctx.exit(1)

    try:
        click.echo(f"🔍 查询: {question}")
        click.echo(f"📊 方法: {method}")

        with PerformanceTimer("查询处理"):
            retrieval_result = rag_system.retriever.search(
                query=question,
                top_k=top_k,
                method=method
            )

            # 调试输出：查看检索结果
            click.echo(f"\n📊 检索结果统计:")
            click.echo(f"  总结果数: {retrieval_result.total_hits}")
            click.echo(f"  稠密结果: {len(retrieval_result.dense_hits)}")
            click.echo(f"  稀疏结果: {len(retrieval_result.sparse_hits)}")
            click.echo(f"  融合结果: {len(retrieval_result.hits)}")
            
            if retrieval_result.hits:
                click.echo(f"\n📄 融合结果前3个:")
                for i, hit in enumerate(retrieval_result.hits[:3], 1):
                    click.echo(f"  {i}. 分数: {hit.score:.4f}, 内容: {hit.content[:100]}...")

            generation_result = rag_system.generator.generate_answer(
                question=question,
                retrieval_result=retrieval_result
            )

        if output == 'json':
            _output_json_result(generation_result)
        elif output == 'detailed':
            _output_detailed_result(generation_result)
        else:
            _output_simple_result(generation_result)

    except Exception as e:
        click.echo(f"❌ 查询失败: {str(e)}", err=True)
        logger.exception("查询处理异常")
        ctx.exit(1)

@cli.command()
@click.pass_context
def rebuild_index(ctx):
    """重建检索索引（稀疏 TF-IDF）"""
    if not rag_system.initialize():
        click.echo("❌ 系统未就绪", err=True)
        ctx.exit(1)

    try:
        click.echo("🔄 开始重建检索索引...")
        with PerformanceTimer("索引重建"):
            rag_system.retriever.build_sparse_index(force_rebuild=True)
        click.echo("✅ 索引重建完成")
    except Exception as e:
        click.echo(f"❌ 索引重建失败: {str(e)}", err=True)
        ctx.exit(1)


@cli.command()
@click.argument('doc_id')
@click.pass_context
def delete(ctx, doc_id):
    """删除文档（按 doc_id 删除 Milvus 记录）"""
    if not rag_system.initialize():
        click.echo("❌ 系统未就绪", err=True)
        ctx.exit(1)

    try:
        success = rag_system.vector_store.delete_by_doc_id(doc_id)

        if success:
            click.echo(f"✅ 文档删除成功: {doc_id}")
            try:
                click.echo("🔄 重建检索索引...")
                rag_system.retriever.build_sparse_index(force_rebuild=True)
                click.echo("✅ 索引重建完成")
            except Exception as e:
                click.echo(f"⚠️ 索引重建失败（不影响删除成功）: {str(e)}")
        else:
            click.echo(f"❌ 文档删除失败或不存在: {doc_id}", err=True)
            ctx.exit(1)

    except Exception as e:
        click.echo(f"❌ 删除文档失败: {str(e)}", err=True)
        ctx.exit(1)


@cli.command()
@click.pass_context
def chat(ctx):
    """交互式聊天"""
    if not rag_system.initialize():
        click.echo("❌ 系统未就绪", err=True)
        ctx.exit(1)

    click.echo("💬 进入交互式聊天模式")
    click.echo("💡 输入 'quit', 'exit' 或 Ctrl+C 退出")
    click.echo("💡 输入 'help' 查看帮助")
    click.echo("-" * 50)

    while True:
        try:
            question = click.prompt("🤔 您的问题", type=str)

            if question.lower() in ['quit', 'exit']:
                break
            if question.lower() == 'help':
                click.echo("直接输入问题即可。")
                continue
            if not question.strip():
                continue

            try:
                with PerformanceTimer("查询处理", verbose=False):
                    retrieval_result = rag_system.retriever.search(
                        query=question,
                        top_k=5,
                        method='hybrid'
                    )
                    generation_result = rag_system.generator.generate_answer(
                        question=question,
                        retrieval_result=retrieval_result
                    )

                click.echo(f"\n🤖 {generation_result.answer}")
                if generation_result.sources:
                    click.echo(f"📚 参考了 {len(generation_result.sources)} 个文档片段")
                else:
                    click.echo("⚠️ 未找到相关文档")
                click.echo(f"⏱️ 耗时: {generation_result.total_time:.2f}秒\n")

            except Exception as e:
                click.echo(f"❌ 查询出错: {str(e)}\n")

        except (KeyboardInterrupt, EOFError):
            break

    click.echo("\n👋 再见!")


@cli.command()
@click.pass_context
def list_docs(ctx):
    """列出所有文档统计信息"""
    if not rag_system.initialize():
        click.echo("❌ 系统未就绪", err=True)
        ctx.exit(1)
    
    try:
        stats = rag_system.vector_store.get_collection_stats()
        count = stats.get("entity_count", 0)
        
        click.echo("📊 文档统计信息:")
        click.echo("=" * 50)
        click.echo(f"📚 文档总数: {count}")
        click.echo(f"🏷️  集合名称: {stats.get('collection_name', '未知')}")
        click.echo(f"📐 向量维度: {stats.get('dimension', '未知')}")
        click.echo(f"🔍 索引类型: {stats.get('index_type', '未知')}")
        click.echo(f"📏 度量类型: {stats.get('metric_type', '未知')}")
        click.echo("=" * 50)
        
    except Exception as e:
        click.echo(f"❌ 获取文档统计失败: {str(e)}")
        ctx.exit(1)


def main():
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n👋 用户中断，退出程序")
        sys.exit(0)
    except Exception as e:
        click.echo(f"❌ 程序异常: {str(e)}", err=True)
        logger.exception("CLI主程序异常")
        sys.exit(1)


if __name__ == "__main__":
    main()