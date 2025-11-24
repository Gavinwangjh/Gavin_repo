# corpus_reader.py
"""
跨论文问答 / 综述生成模块

依赖：
- data/index/faiss.index
- data/index/docs.json
- data/index/metadata.json

这些由 paper_indexer.py 生成：
    python paper_indexer.py
"""

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llm_api import call_llm

INDEX_DIR = "data/index"

# 和之前保持一致的 embedding 模型
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def load_corpus():
    """加载 FAISS 索引 + 文本片段 docs + 论文元数据 metadata"""
    index_path = os.path.join(INDEX_DIR, "faiss.index")
    docs_path = os.path.join(INDEX_DIR, "docs.json")
    meta_path = os.path.join(INDEX_DIR, "metadata.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"未找到索引文件: {index_path}，请先运行 paper_indexer.py")

    index = faiss.read_index(index_path)

    with open(docs_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 建一个 paper_id -> filename 的映射，方便展示
    pid2fname = {m["paper_id"]: m["filename"] for m in metadata}

    return index, docs, pid2fname


def search_corpus(question: str, index, docs, top_k: int = 10):
    """
    在“所有论文的所有段落”上做语义检索，返回 top_k 个最相关段落
    返回结果格式：[ {paper_id, page, text, score}, ... ]
    """
    q_vec = embed_model.encode([question], convert_to_numpy=True)
    D, I = index.search(q_vec, top_k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        d = docs[int(idx)]
        d = d.copy()
        d["score"] = float(dist)
        results.append(d)

    return results


def build_corpus_prompt(question: str, hits, pid2fname, max_chars: int = 4000):
    """
    把多个论文片段 + 页码 + 文件名 拼成一个大 prompt，喂给 LLM 用。
    max_chars 用来控制上下文长度，防止太长。
    """
    context_lines = []
    used_evidence = []

    total_len = 0
    for h in hits:
        paper_id = h["paper_id"]
        page = h["page"]
        text = h["text"]
        fname = pid2fname.get(paper_id, paper_id)

        snippet = f"【论文：{fname} | 第{page}页】\n{text}\n"
        if total_len + len(snippet) > max_chars:
            break

        context_lines.append(snippet)
        total_len += len(snippet)

        used_evidence.append({
            "paper_id": paper_id,
            "filename": fname,
            "page": page,
            "text": text
        })

    context = "\n".join(context_lines)

    prompt = f"""
你是一名科研论文综述助手。现在有多篇论文的原文片段，请你基于这些内容回答一个科研问题。

下面是检索到的论文原文片段（包含论文名与页码）：
------------------------
{context}
------------------------

要求：
1. 回答必须只基于以上片段的内容，不要编造论文中没有的结论。
2. 尽量用条理清晰的方式总结（可以用分点）。
3. 如果不同论文有不同观点，可以指出差异。
4. 最后给出一句简短总结。

问题：{question}
"""
    return prompt, used_evidence


def corpus_rag_query(question: str, top_k: int = 10, max_chars: int = 4000):
    """
    对“整个论文语料库”进行 RAG 问答：
    输入：问题（中文或英文）
    输出：
        answer: LLM 生成的总结
        evidence: 用到的论文片段列表（含 paper_id / filename / page / text）
    """
    index, docs, pid2fname = load_corpus()

    hits = search_corpus(question, index, docs, top_k=top_k)
    if not hits:
        return "没有检索到相关内容，请换一个问题试试。", []

    prompt, evidence = build_corpus_prompt(question, hits, pid2fname, max_chars=max_chars)

    answer = call_llm([
        {"role": "system", "content": "你是一个严谨的科研论文综述助手。"},
        {"role": "user", "content": prompt}
    ])

    return answer.strip(), evidence


# 方便命令行直接用
if __name__ == "__main__":
    print("🧠 跨论文科研问答助手（Corpus RAG）")
    print("提示：请确保已经运行过 paper_indexer.py 构建索引。\n")

    while True:
        q = input("请输入你的科研问题（或输入 q 退出）：").strip()
        if not q or q.lower() == "q":
            break

        print("\n🔍 正在检索语料库并生成回答...\n")
        answer, evidence = corpus_rag_query(q, top_k=12, max_chars=4000)

        print("====== 模型回答 ======\n")
        print(answer)
        print("\n====== 证据来源（论文+页码）======\n")
        for i, e in enumerate(evidence, 1):
            print(f"[{i}] 论文: {e['filename']} | 第 {e['page']} 页")
        print("\n" + "=" * 50 + "\n")
