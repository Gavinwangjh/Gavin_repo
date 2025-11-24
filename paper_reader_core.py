# paper_reader_core.py
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llm_api import call_llm

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def parse_pdf_single(path):
    
    docs = []
    with pdfplumber.open(path) as pdf:
        for page_id, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for para in text.split("\n\n"):
                para = para.strip()
                if len(para) < 50:
                    continue
                docs.append({"page": page_id, "text": para})
    return docs

def build_index_single(docs):
    texts = [d["text"] for d in docs]
    emb = embed_model.encode(texts, convert_to_numpy=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    return index, emb, texts

def search(query, index, docs, k=6):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    return [docs[int(i)] for i in I[0]]

def build_prompt(question, retrieved):
    context = ""
    pages = []
    for r in retrieved:
        context += f"[第{r['page']}页] {r['text']}\n"
        pages.append(r["page"])
    prompt = f"""
你是一名科研助手，任务是基于论文原文给出可靠总结。

下面是论文中的原始段落（带页码）：
-------------------------
{context}
-------------------------

请仅基于以上内容回答，不要编造信息。

问题：{question}
"""
    return prompt, sorted(set(pages))

def generate_guide_for_pdf(pdf_path: str):
    docs = parse_pdf_single(pdf_path)
    index, emb, texts = build_index_single(docs)

    QUESTIONS = {
        "summary": "请用不超过三句话总结这篇论文的核心内容。",
        "problem": "这篇论文主要解决了什么研究问题？",
        "method": "这篇论文采用了什么方法或模型？",
        "conclusion": "这篇论文的主要结论是什么？是否提到了局限性？",
    }

    results = {}

    for key, q in QUESTIONS.items():
        retrieved = search(q, index, docs, k=6)
        prompt, pages = build_prompt(q, retrieved)
        answer = call_llm([
            {"role": "system", "content": "你是一个严谨的科研论文阅读助手。"},
            {"role": "user", "content": prompt}
        ])
        results[key] = {
            "answer": answer.strip(),
            "pages": pages
        }

    return results

if __name__ == "__main__":
    pdf_path = "data/pdfs/A Comparative Analysis of Recurrent and Attention Architectures for Isolated Sign Language Recognition.pdf"  # 换成你自己的路径
    guide = generate_guide_for_pdf(pdf_path)
    for k, v in guide.items():
        print("\n====", k.upper(), "====")
        print("引用页码:", v["pages"])
        print(v["answer"])
