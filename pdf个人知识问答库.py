# 1. 加载文件
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ========== 第二步：让用户输入 PDF 路径 ==========
print("请粘贴你的 PDF 文件路径，或者一个包含 PDF 的文件夹路径。")
print("如果有多个文件，用英文逗号分隔。")
print("例如：")
print("  D:/note1.pdf")
print("  D:/my_knowledge")
print("  D:/note1.pdf, D:/note2.pdf, D:/my_knowledge")
print("-" * 50)

user_input = input("请输入路径：").strip()

# 把用户输入按逗号拆成多个路径
input_paths = [p.strip() for p in user_input.split(",") if p.strip()]

if not input_paths:
    print("=" * 50)
    print("错误：你没有输入任何路径。")
    print("=" * 50)
    exit()

# 收集所有 PDF 文件
all_pdf_files = []

for path in input_paths:
    if not os.path.exists(path):
        print(f"提示：路径不存在，已跳过：{path}")
        continue

    if os.path.isfile(path):
        # 如果是一个文件，直接判断是不是 PDF
        if path.lower().endswith(".pdf"):
            all_pdf_files.append(path)
        else:
            print(f"提示：这不是 PDF 文件，已跳过：{path}")

    elif os.path.isdir(path):

        pdf_files = [f for f in os.listdir(path) if f.lower().endswith(".pdf")]

        for pdf_file in pdf_files:
            all_pdf_files.append(os.path.join(path, pdf_file))
if not all_pdf_files:
    print("=" * 50)
    print("错误：没有找到任何 PDF 文件。")
    print("请确认你输入的路径里包含 PDF 文件。")
    print("=" * 50)
    exit()

print(f"\n找到 {len(all_pdf_files)} 个 PDF 文件，开始处理...")

# 逐个加载所有 PDF
docs = []
for file_path in all_pdf_files:
    print(f"正在读取：{file_path}")
    loader = PyPDFLoader(file_path)
    docs.extend(loader.load())

print(f"共加载 {len(docs)} 页内容。")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 嵌入 + 存库
import os
from langchain_community.embeddings import ZhipuAIEmbeddings

# 第一步：先获取 Key，检查是否存在
api_key = os.environ.get("ZHIPU_API_KEY")

if not api_key:
    print("=" * 50)
    print("错误：没有找到 ZHIPU_API_KEY 环境变量。")
    print("请先按以下步骤配置你的 API Key：")
    print("1. 去 https://open.bigmodel.cn/apikey/platform 注册并生成 Key")
    print("2. 在运行环境里设置环境变量 ZHIPU_API_KEY=你的Key")
    print("3. 重新运行程序")
    print("=" * 50)
    exit()

# 第二步：确认 Key 存在后，再传给嵌入模型
embeddings = ZhipuAIEmbeddings(
    api_key=api_key,
    model="embedding-2"
)

db = Chroma.from_documents(chunks, embeddings, persist_directory="D:/note1")

# 检索+检索结果
retriever = db.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke("讲了什么？")

for i, r in enumerate(results):
    print(f"--- 第{i+1}块 ---")
    print(r.page_content[:200])

# ai回答
context = "\n\n".join([r.page_content for r in results])
# 构造提示词
user_question = "讲了什么？"
prompt = f"根据以下内容回答问题：\n{context}\n\n问题：{user_question}\n如果找不到答案，就说“根据现有资料无法回答”。"

# 调用智谱对话api
from langchain_community.chat_models import ChatZhipuAI

chat_model = ChatZhipuAI(
    api_key=os.environ.get("ZHIPU_API_KEY"),
    model="glm-3-turbo"
)

# 发出提示词并获取回答
response = chat_model.invoke(prompt)
ai_answer = response.content

# 打印回答及参考
print("\n--- AI的回答 ---")
print(ai_answer)

print("\n--- 参考来源 ---")
for i, result in enumerate(results):
    content_preview = result.page_content[:100]
    meta = result.metadata
    print(f"\n【参考片段 {i + 1}】")
    print(f"内容摘要：{content_preview}...")
    print(f"元数据：{meta}")
    print("-" * 50)