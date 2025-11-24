本项目实现自动下载论文 · PDF 解析 · 向量索引 · 多文档检索 · LLM 总结/问答 · Streamlit Web App
本项目构建了一个完整的论文阅读和检索增强生成（RAG）系统，支持：
批量下载 arXiv 论文
自动解析 PDF、按段落切分
构建 FAISS 向量数据库
基于 Sentence-Transformer 的多文档语义检索
使用 DeepSeek API 进行总结与问答
单文档导读（摘要、方法、问题、结论）
跨论文检索问答（Corpus RAG）
Web UI（Streamlit）


project/
│
├── llm_api.py                 # DeepSeek API 调用
├── download_papers_arxiv.py   # 批量下载 arXiv 论文
├── paper_indexer.py           # 构建 FAISS 全库索引
├── paper_reader_core.py       # 单论文 PDF 导读
├── corpus_reader.py           # 跨论文检索增强问答
├── app_streamlit.py           # Web UI
│
├── data/
│   ├── pdfs/                  # 下载的论文 PDF
│   ├── single_tmp/            # Web 临时文件
│   └── index/
│       ├── faiss.index
│       ├── embeddings.npy
│       ├── docs.json
│       └── metadata.json
