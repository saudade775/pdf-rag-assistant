from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatZhipuAI

st.title("PDF 个人知识库问答器")

uploaded_files = st.file_uploader(
    "上传 PDF 文件（可以多选）",
    type=["pdf"],
    accept_multiple_files=True
)


# 关键：用 cache_resource 缓存，避免每次交互都重新跑
@st.cache_resource
def build_retriever(file_contents):
    # file_contents 是一个元组，里面是每个文件的字节内容
    docs = []
    for content in file_contents:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())

    # 切片
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 嵌入 + 存库
    api_key = os.environ.get("ZHIPU_API_KEY")
    embeddings = ZhipuAIEmbeddings(api_key=api_key, model="embedding-2")
    db = Chroma.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever


if uploaded_files:
    # 把每个文件读成字节，拼成一个元组，作为缓存依据
    file_contents = tuple(f.getvalue() for f in uploaded_files)

    with st.spinner("正在处理 PDF，请稍候..."):
        retriever = build_retriever(file_contents)

    st.success(f"已就绪，共处理 {len(uploaded_files)} 个文件。")

    # 聊天输入
    user_question = st.chat_input("问点什么...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        results = retriever.invoke(user_question)
        context = "\n\n".join([r.page_content for r in results])
        prompt = f"根据以下内容回答问题：\n{context}\n\n问题：{user_question}\n如果找不到答案，就说'根据现有资料无法回答'。"

        api_key = os.environ.get("ZHIPU_API_KEY")
        chat_model = ChatZhipuAI(api_key=api_key, model="glm-3-turbo")
        response = chat_model.invoke(prompt)
        ai_answer = response.content

        with st.chat_message("assistant"):
            st.write(ai_answer)

        with st.expander("查看参考来源"):
            for i, r in enumerate(results):
                st.write(f"**片段 {i + 1}**：{r.page_content[:100]}...")