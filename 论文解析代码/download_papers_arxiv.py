# download_papers_arxiv.py
import os
import re
import arxiv

OUT_DIR = "data/pdfs"
os.makedirs(OUT_DIR, exist_ok=True)

def safe_filename(name: str) -> str:
    """清理非法路径字符"""
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = name.replace("\n", " ").strip()
    return name[:120]

def download_papers(query="cat:cs.CL", max_results=30):
    print(f"\n 搜索中: {query}")
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    for paper in search.results():
        # arXiv 原版 PDF URL
        pdf_url = paper.pdf_url.strip()

        filename = safe_filename(paper.title) + ".pdf"
        filepath = os.path.join(OUT_DIR, filename)

        if os.path.exists(filepath):
            print(f" 已存在: {filename}")
            continue

        print(f"⬇ 正在下载: {paper.title}")
        try:
            paper.download_pdf(filename=filepath)  
        except Exception as e:
            print("下载失败:", e)
            continue

    print("\n下载完成！\n")

if __name__ == "__main__":
    download_papers(query="cat:cs.CL", max_results=50)      # NLP
    download_papers(query="cat:cs.LG", max_results=50)      # ML
    download_papers(query="retrieval augmented generation", max_results=30)  # RAG 专题
    download_papers(query="transformer", max_results=20)    # tansformer
