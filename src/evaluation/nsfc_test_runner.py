# -*- coding: utf-8 -*-
"""
NSFC 自包含评测运行器（不依赖 FastAPI） - v4
目标：
- 区分三种情况：真无缓存 / 仅retriever缓存 / 仅脚本缓存
- 通过 monkey-patch/替换缓存容器，强制让 HybridRetriever 不走内部缓存（尽量不改其源码）
- 生成格式化的报告文件
"""

from __future__ import annotations

import sys
import time
import random
import statistics
import importlib
import pkgutil
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Optional
from datetime import datetime


def _add_project_root_to_syspath() -> Path:
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    print(f"📁 项目根目录: {project_root}")
    print(f"📂 当前目录: {current_file.parent}")
    return project_root


PROJECT_ROOT = _add_project_root_to_syspath()

from src.utils.helpers import Logger  # noqa: E402
from loguru import logger  # noqa: E402

from src.vector_store.milvus_store import MilvusVectorStore  # noqa: E402
from src.retrieval.retriever import HybridRetriever  # noqa: E402


def _find_embedding_manager_class() -> Type[Any]:
    pkg = importlib.import_module("src")
    for modinfo in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        modname = modinfo.name
        if modname.startswith("src.api."):
            continue
        try:
            m = importlib.import_module(modname)
        except Exception:
            continue
        for attr_name in dir(m):
            if "EmbeddingManager" in attr_name:
                obj = getattr(m, attr_name, None)
                if isinstance(obj, type):
                    return obj
    raise ImportError("未在 src 中找到名为 *EmbeddingManager* 的类")


def _list_text_files(docs_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in ("*.txt", "*.md"):
        files.extend(docs_dir.rglob(ext))
    return sorted(files)


def _build_queries_from_docs(docs_dir: Path, num_queries: int, seed: int = 20260120) -> List[str]:
    files = _list_text_files(docs_dir)
    if not files:
        raise FileNotFoundError(f"{docs_dir} 下未找到 .txt/.md 文档")

    sentences: List[str] = []
    for fp in files[:80]:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if 12 <= len(line) <= 120:
                sentences.append(line)

    if not sentences:
        sentences = [f.stem for f in files]

    rnd = random.Random(seed)
    rnd.shuffle(sentences)
    return sentences[:num_queries]


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    if f == c:
        return xs_sorted[f]
    return xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)


def _repeat_summary(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"first": 0.0, "rest_avg": 0.0, "rest_p50": 0.0, "rest_p95": 0.0}
    first = latencies[0]
    rest = latencies[1:] if len(latencies) > 1 else []
    return {
        "first": first,
        "rest_avg": statistics.mean(rest) if rest else 0.0,
        "rest_p50": _percentile(rest, 0.5) if rest else 0.0,
        "rest_p95": _percentile(rest, 0.95) if rest else 0.0,
    }


class _AlwaysMissDict(dict):
    """一个永远 miss 的 dict，用于让 retriever 内部缓存形同虚设。"""
    def __contains__(self, key):  # type: ignore[override]
        return False
    def get(self, key, default=None):  # type: ignore[override]
        return default


class NSFCStandaloneEvaluator:
    def __init__(self, docs_dir: str):
        self.docs_dir = (PROJECT_ROOT / docs_dir).resolve()
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"文档目录不存在: {self.docs_dir}")

        self.vector_store = MilvusVectorStore()
        try:
            self.vector_store.initialize(force_recreate=False)
        except Exception as e:
            logger.warning(f"Milvus initialize 调用失败（若你已自动初始化可忽略）: {e}")

        EmbeddingManagerCls = _find_embedding_manager_class()
        self.embedding_manager = EmbeddingManagerCls()

        self.retriever = HybridRetriever(self.vector_store, self.embedding_manager)

        self._cache: Dict[Tuple[str, str, int], Any] = {}
        self._cache_enabled: bool = True
        self.cache_hits: int = 0
        self.cache_total: int = 0

        # 用于临时恢复 retriever 的缓存字段
        self._retriever_cache_backup: Dict[str, Any] = {}

    def _reset_cache_stats(self):
        self.cache_hits = 0
        self.cache_total = 0

    def _disable_retriever_cache_temporarily(self):
        """
        尽量"广覆盖"地禁用 HybridRetriever 内部缓存：
        - 把常见缓存字段替换为 AlwaysMissDict / None
        - 保留原值以便恢复
        """
        cand_names = [
            "cache", "_cache", "search_cache", "_search_cache",
            "retrieval_cache", "_retrieval_cache",
            "result_cache", "_result_cache",
            "query_cache", "_query_cache",
        ]
        for name in cand_names:
            if hasattr(self.retriever, name):
                self._retriever_cache_backup[name] = getattr(self.retriever, name)
                try:
                    setattr(self.retriever, name, _AlwaysMissDict())
                except Exception:
                    try:
                        setattr(self.retriever, name, None)
                    except Exception:
                        pass

    def _restore_retriever_cache(self):
        for k, v in self._retriever_cache_backup.items():
            try:
                setattr(self.retriever, k, v)
            except Exception:
                pass
        self._retriever_cache_backup.clear()

    def _retrieve(self, method: str, query: str, top_k: int) -> Tuple[Any, bool]:
        q = query.strip()
        key = (method, q, top_k)

        self.cache_total += 1
        if self._cache_enabled and key in self._cache:
            self.cache_hits += 1
            return self._cache[key], True

        result = self.retriever.search(query=q, method=method, top_k=top_k)

        if self._cache_enabled:
            self._cache[key] = result

        return result, False

    def warmup_sparse_index(self):
        try:
            self.retriever.build_sparse_index(force_rebuild=False)
            logger.info("✅ 稀疏索引预热完成（如果已构建会很快）")
        except Exception as e:
            logger.warning(f"稀疏索引预热失败（可忽略，但 hybrid 的首次耗时会被污染）: {e}")

    def run_once(
        self,
        name: str,
        methods: List[str],
        top_k: int,
        enable_cache: bool,
        queries: List[str],
        clear_retriever_cache: bool = True,
        force_no_retriever_cache: bool = False,
        verbose_each: bool = False,
    ) -> Dict[str, Any]:
        self._reset_cache_stats()
        self._cache_enabled = enable_cache
        self._cache.clear()

        if clear_retriever_cache:
            try:
                self.retriever.clear_cache()
            except Exception:
                pass

        if force_no_retriever_cache:
            self._disable_retriever_cache_temporarily()

        try:
            metrics: Dict[str, Dict[str, Any]] = {}
            per_method_latencies: Dict[str, List[float]] = {}

            for method in methods:
                latencies: List[float] = []
                hit_latencies: List[float] = []
                miss_latencies: List[float] = []

                for i, q in enumerate(queries, start=1):
                    t0 = time.perf_counter()
                    _, hit = self._retrieve(method, q, top_k)
                    t1 = time.perf_counter()
                    dt = t1 - t0

                    latencies.append(dt)
                    if hit:
                        hit_latencies.append(dt)
                    else:
                        miss_latencies.append(dt)

                    if verbose_each:
                        print(f"  - {name} [{method}] #{i:02d} {'HIT' if hit else 'MISS'} dt={dt:.6f}s")

                per_method_latencies[method] = latencies

                metrics[method] = {
                    "avg": statistics.mean(latencies) if latencies else 0.0,
                    "p50": _percentile(latencies, 0.50),
                    "p95": _percentile(latencies, 0.95),
                    "n": len(latencies),
                    "miss_avg": statistics.mean(miss_latencies) if miss_latencies else 0.0,
                    "hit_avg": statistics.mean(hit_latencies) if hit_latencies else 0.0,
                    "hit_n": len(hit_latencies),
                    "miss_n": len(miss_latencies),
                }
                if name.startswith("repeat_"):
                    metrics[method]["repeat"] = _repeat_summary(latencies)

            hit_rate = (self.cache_hits / self.cache_total) if self.cache_total else 0.0
            return {
                "name": name,
                "methods": methods,
                "top_k": top_k,
                "metrics": metrics,
                "latencies": per_method_latencies,
                "cache": {
                    "enabled": enable_cache,
                    "hits": self.cache_hits,
                    "total": self.cache_total,
                    "hit_rate": hit_rate,
                },
                "force_no_retriever_cache": force_no_retriever_cache,
                "timestamp": datetime.now().isoformat(),
                "queries_count": len(queries),
                "queries_sample": queries[:3] if queries else [],
            }
        finally:
            if force_no_retriever_cache:
                self._restore_retriever_cache()


def _print_report(title: str, report: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for m, v in report["metrics"].items():
        line = (
            f"[{m}] avg={v['avg']:.4f}s p50={v['p50']:.4f}s p95={v['p95']:.4f}s n={v['n']} "
            f"| miss_avg={v['miss_avg']:.4f}s (n={v['miss_n']}) hit_avg={v['hit_avg']:.6f}s (n={v['hit_n']})"
        )
        print(line)

        if "repeat" in v:
            r = v["repeat"]
            print(
                f"      repeat_summary: first={r['first']:.4f}s rest_avg={r['rest_avg']:.6f}s "
                f"rest_p50={r['rest_p50']:.6f}s rest_p95={r['rest_p95']:.6f}s"
            )

    c = report["cache"]
    print(f"\n脚本缓存: enabled={c['enabled']} hit_rate={c['hit_rate']:.1%} hits={c['hits']} / {c['total']}")
    print(f"force_no_retriever_cache={report.get('force_no_retriever_cache', False)}")
    print("=" * 70)


def save_json_report(reports: List[Dict[str, Any]], filename: str = None) -> str:
    """保存报告为 JSON 文件"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nsfc_evaluation_report_{timestamp}.json"
    
    report_data = {
        "generated_at": datetime.now().isoformat(),
        "reports": reports,
        "summary": {
            "total_experiments": len(reports),
            "methods_tested": list(set(m for r in reports for m in r["methods"])),
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ JSON 报告已保存: {filename}")
    return filename


def save_markdown_report(reports: List[Dict[str, Any]], filename: str = None) -> str:
    """保存报告为 Markdown 文件"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nsfc_evaluation_report_{timestamp}.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# NSFC 检索系统性能评测报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 测试概览\n\n")
        f.write(f"- 实验总数: {len(reports)}\n")
        all_methods = set()
        for r in reports:
            all_methods.update(r["methods"])
        f.write(f"- 测试方法: {', '.join(sorted(all_methods))}\n")
        f.write(f"- 测试数据: {reports[0].get('queries_count', 'N/A')} 条查询\n\n")
        
        f.write("## 详细结果\n\n")
        
        for i, report in enumerate(reports, 1):
            f.write(f"### 实验 {i}: {report['name']}\n\n")
            f.write(f"- **测试配置**:\n")
            f.write(f"  - 方法: {', '.join(report['methods'])}\n")
            f.write(f"  - Top-K: {report['top_k']}\n")
            f.write(f"  - 查询数量: {report.get('queries_count', 'N/A')}\n")
            f.write(f"  - 脚本缓存: {'启用' if report['cache']['enabled'] else '禁用'}\n")
            f.write(f"  - 强制禁用 Retriever 缓存: {'是' if report.get('force_no_retriever_cache', False) else '否'}\n\n")
            
            f.write("- **性能指标**:\n\n")
            f.write("| 方法 | 平均耗时(s) | P50(s) | P95(s) | 总次数 | 命中次数 | 命中率 |\n")
            f.write("|------|-------------|--------|--------|--------|----------|--------|\n")
            
            for method, metrics in report["metrics"].items():
                cache_info = report["cache"]
                hit_rate = cache_info["hit_rate"]
                hit_count = cache_info["hits"] if "hits" in cache_info else 0
                total_count = cache_info["total"] if "total" in cache_info else metrics["n"]
                
                f.write(f"| {method} | {metrics['avg']:.4f} | {metrics['p50']:.4f} | {metrics['p95']:.4f} | "
                       f"{total_count} | {hit_count} | {hit_rate:.1%} |\n")
            
            f.write("\n")
            
            if "repeat" in next(iter(report["metrics"].values())):
                f.write("- **重复查询分析**:\n")
                for method, metrics in report["metrics"].items():
                    if "repeat" in metrics:
                        r = metrics["repeat"]
                        f.write(f"  - {method}: 首次={r['first']:.4f}s, 后续平均={r['rest_avg']:.6f}s\n")
            
            f.write("\n")
    
    print(f"✅ Markdown 报告已保存: {filename}")
    return filename


def save_csv_summary(reports: List[Dict[str, Any]], filename: str = None) -> str:
    """保存汇总数据为 CSV 文件"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nsfc_evaluation_summary_{timestamp}.csv"
    
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            '实验名称', '方法', 'Top-K', '脚本缓存', '禁用Retriever缓存',
            '平均耗时(s)', 'P50(s)', 'P95(s)', '总次数', '命中次数', '命中率',
            '未命中平均(s)', '命中平均(s)', '测试时间'
        ])
        
        for report in reports:
            for method, metrics in report["metrics"].items():
                cache_info = report["cache"]
                writer.writerow([
                    report['name'],
                    method,
                    report['top_k'],
                    '是' if report['cache']['enabled'] else '否',
                    '是' if report.get('force_no_retriever_cache', False) else '否',
                    f"{metrics['avg']:.6f}",
                    f"{metrics['p50']:.6f}",
                    f"{metrics['p95']:.6f}",
                    cache_info['total'],
                    cache_info['hits'],
                    f"{cache_info['hit_rate']:.4f}",
                    f"{metrics['miss_avg']:.6f}",
                    f"{metrics['hit_avg']:.6f}",
                    report.get('timestamp', '')
                ])
    
    print(f"✅ CSV 汇总已保存: {filename}")
    return filename


def main() -> int:
    Logger.setup_logger(log_level="INFO")
    
    print("🚀 NSFC 检索系统性能评测开始")
    print("=" * 50)

    evaluator = NSFCStandaloneEvaluator(docs_dir="data/nsfc_test")
    evaluator.warmup_sparse_index()

    base_queries = _build_queries_from_docs(evaluator.docs_dir, num_queries=15, seed=20260120)
    
    all_reports = []

    report1 = evaluator.run_once(
        name="nsfc_basic",
        methods=["dense", "hybrid", "hybrid_rrf"],
        top_k=5,
        enable_cache=True,
        queries=base_queries,
        clear_retriever_cache=True,
    )
    all_reports.append(report1)
    _print_report("🧪 NSFC实验1: 基础检索性能（稀疏索引已预热）", report1)

    report2_no = evaluator.run_once(
        name="nsfc_no_cache",
        methods=["hybrid_rrf"],
        top_k=5,
        enable_cache=False,
        queries=base_queries[:10],
        clear_retriever_cache=True,
    )
    all_reports.append(report2_no)
    _print_report("🧪 NSFC实验2A: 无脚本缓存（固定queries）", report2_no)

    report2_yes = evaluator.run_once(
        name="nsfc_with_cache",
        methods=["hybrid_rrf"],
        top_k=5,
        enable_cache=True,
        queries=base_queries[:10],
        clear_retriever_cache=True,
    )
    all_reports.append(report2_yes)
    _print_report("🧪 NSFC实验2B: 有脚本缓存（固定queries）", report2_yes)

    repeat_q = base_queries[0]
    repeat_queries = [repeat_q] * 10

    report3A = evaluator.run_once(
        name="repeat_no_script_cache",
        methods=["hybrid_rrf"],
        top_k=5,
        enable_cache=False,
        queries=repeat_queries,
        clear_retriever_cache=True,
        force_no_retriever_cache=False,
        verbose_each=True,
    )
    all_reports.append(report3A)
    _print_report("🧪 NSFC实验3A: 重复Query（无脚本缓存；允许retriever内部缓存）", report3A)

    report3B = evaluator.run_once(
        name="repeat_with_script_cache",
        methods=["hybrid_rrf"],
        top_k=5,
        enable_cache=True,
        queries=repeat_queries,
        clear_retriever_cache=True,
        force_no_retriever_cache=False,
        verbose_each=True,
    )
    all_reports.append(report3B)
    _print_report("🧪 NSFC实验3B: 重复Query（有脚本缓存；允许retriever内部缓存）", report3B)

    report3C = evaluator.run_once(
        name="repeat_true_no_cache",
        methods=["hybrid_rrf"],
        top_k=5,
        enable_cache=False,
        queries=repeat_queries,
        clear_retriever_cache=True,
        force_no_retriever_cache=True,
        verbose_each=True,
    )
    all_reports.append(report3C)
    _print_report("🧪 NSFC实验3C: 重复Query（真无缓存：禁脚本缓存 + 禁retriever内部缓存）", report3C)

    report3D = evaluator.run_once(
        name="repeat_script_cache_only",
        methods=["hybrid_rrf"],
        top_k=5,
        enable_cache=True,
        queries=repeat_queries,
        clear_retriever_cache=True,
        force_no_retriever_cache=True,
        verbose_each=True,
    )
    all_reports.append(report3D)
    _print_report("🧪 NSFC实验3D: 重复Query（只有脚本缓存：禁retriever内部缓存）", report3D)

    # 保存报告文件
    print("\n" + "=" * 70)
    print("📊 生成报告文件...")
    print("=" * 70)
    
    json_file = save_json_report(all_reports)
    md_file = save_markdown_report(all_reports)
    csv_file = save_csv_summary(all_reports)
    
    print(f"\n📁 生成的文件:")
    print(f"  - JSON 详细报告: {json_file}")
    print(f"  - Markdown 报告: {md_file}")
    print(f"  - CSV 数据汇总: {csv_file}")

    print("\n🎯 NSFC测试完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())