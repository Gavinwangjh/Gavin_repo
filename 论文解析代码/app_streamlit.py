# app_streamlit.py
import os
import streamlit as st
from paper_reader_core import generate_guide_for_pdf

TMP_DIR = "data/single_tmp"
os.makedirs(TMP_DIR, exist_ok=True)

st.set_page_config(page_title="论文导读助手", layout="wide")

st.title("论文导读助手(NLP + RAG Demo)")
st.write("上传一篇学术论文 PDF,我会帮你生成：摘要、研究问题、方法与结论，并附上引用页码。")

uploaded_file = st.file_uploader("上传论文 PDF", type=["pdf"])

if uploaded_file is not None:
    tmp_path = os.path.join(TMP_DIR, "upload.pdf")
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("正在解析和生成导读，请稍等...")
    guide = generate_guide_for_pdf(tmp_path)

    for section, info in guide.items():
        st.subheader(section.upper())
        st.write("引用页码: ", info["pages"])
        st.write(info["answer"])
