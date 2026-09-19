####网页地址
https://pdf-rag-assistant-tvfxdtdxkj8boxbzjjqigd.streamlit.app/

# pdf-rag-assistant
基于 LangChain + Chroma + 智谱AI 的 PDF 知识库问答器
# PDF 个人知识库问答器

基于 LangChain + Chroma + 智谱AI 的 PDF 知识库问答系统。

用户粘贴 PDF 路径，程序自动读取、切片、嵌入，然后回答用户提问，并显示参考来源。

## 功能

- 用户交互式输入 PDF 路径（支持单个文件、多个文件、文件夹）
- 自动读取 PDF 内容，切片、嵌入、存入向量数据库
- 用户提问后，检索最相关的 3 块内容
- 调用大模型生成回答
- 显示回答参考了哪几块内容
- 自动检查 API Key 和文件路径，出错时给出友好提示

## 技术栈

- Python
- LangChain
- Chroma（向量数据库）
- 智谱AI（嵌入 + 对话）

## 如何运行

### 1. 安装依赖

```bash
pip install langchain langchain-community langchain-chroma pypdf zhipuai
