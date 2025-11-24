# paper_indexer.py
import os
import json
import pdfplumber
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

PDF_DIR = "data/pdfs"
INDEX_DIR = "data/index"

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def parse_single_pdf(path, paper_id):
    """
    返回：[{paper_id, page, text}, ...]
    """
    docs = []
    with pdfplumber.open(path) as pdf:
        for page_id, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for para in text.split("\n\n"):
                para = para.strip()
                if len(para) < 50:
                    continue
                docs.append({
                    "paper_id": paper_id,
                    "page": page_id,
                    "text": para
                })
    return docs

def build_corpus():
    all_docs = []
    metadata = []
    paper_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    for idx, fname in enumerate(tqdm(paper_files, desc="解析PDF")):
        paper_id = f"paper_{idx}"
        path = os.path.join(PDF_DIR, fname)
        docs = parse_single_pdf(path, paper_id)
        all_docs.extend(docs)
        metadata.append({
            "paper_id": paper_id,
            "filename": fname,
            "num_chunks": len(docs)
        })

    return all_docs, metadata

def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)

    docs, metadata = build_corpus()
    texts = [d["text"] for d in docs]

    print("编码为向量...")
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 保存 index + embeddings + docs 元数据
    
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), embeddings)

    with open(os.path.join(INDEX_DIR, "docs.json"), "w", encoding="utf-8") as f:

        json.dump(docs, f, ensure_ascii=False, indent=2)

    with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"完成：共 {len(docs)} 段文本，{len(metadata)} 篇论文。")

if __name__ == "__main__":
    build_index()
