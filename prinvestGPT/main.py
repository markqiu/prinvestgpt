import gradio as gr
import os
import time
from langchain.chat_models import ErnieBotChat
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import ErnieEmbeddings
import yaml
import random
import pandas as pd
import PyPDF2
from tqdm import tqdm
from langchain.vectorstores import Milvus
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from duckduckgo_search import DDGS
import itertools


###llm
def ddg_search(tosearch):
    web_content = ""
    count = 1
    with DDGS(timeout=10) as ddgs:
        answer = itertools.islice(ddgs.text(f"{tosearch}", region="cn-zh"), 5)  #
        for result in answer:
            web_content += f"{count}. {result['body']}"
            count += 1
        # instant = itertools.islice(ddgs.answers(f"kingsoft"), 5)#,region="cn-zh"
        # for result in instant:
        #     web_content += f"{count}. {result['text']}\n"
    return web_content


with open("../new_cof.yaml", "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
llm_model = config["MODELS"]["llm_model"]
embedding_model = config["MODELS"]["embedding_model"]
ernie_client_id = config["API"]["ernie_client_id"]
ernie_client_secret = config["API"]["ernie_client_secret"]
openai_api_key = config["API"]["openai_api_key"]
MILVUS_HOST = config["MILVUS"]["host"]
MILVUS_PORT = config["MILVUS"]["port"]


def pre_embeding_file(chatbot):
    message = "预热知识库中，请耐心等待完成......"
    return chatbot + [[message, None]]


def applydata_(chatbot):
    message = "载入知识库成功"
    return chatbot + [[message, None]]


def is_use_database(chatbot, use_database):
    if use_database == "是":
        message = "使用知识库中...."
    else:
        message = "取消使用知识库"
    return chatbot + [[message, None]]


def apply_model_setting(model_name, embedding_model_name, chatbot):
    message = f"载入语言模型{model_name}，embedding模型{embedding_model_name}"
    return chatbot + [[message, None]]


def init_model(llm_model_name, embedding_model_name, temperature, max_tokens):
    llm_model = ErnieBotChat(
        ernie_client_id=ernie_client_id,
        ernie_client_secret=ernie_client_secret,
        temperature=temperature,
    )

    embedding_model = ErnieEmbeddings(
        ernie_client_id=ernie_client_id,
        ernie_client_secret=ernie_client_secret, )
    return llm_model, embedding_model


def general_template(history=False):
    general_template = f"""这下面是文心酱AI与人类的对话. The AI is talkative and provides lots of specific details from its context. 如果AI不知道问题的答案，AI会诚实地说"我不知道"，而不是编造一个答案。AI在回答问题会注意自己的身份和角度。
----
Current conversation:"""
    if history:
        general_template += """
{history}"""
        general_template += """
Human: {input}
AI: """
    else:
        general_template += """
已知内容：
'''{context}'''
"""
        general_template += """
Human: {question}
AI: """
    return general_template


def init_base_chain(llm_model, history=None, user_question=None):
    template = general_template(history=True)
    chain = ConversationChain(llm=llm_model,
                              verbose=True,
                              memory=history,
                              )
    chain.prompt.template = template
    try:
        output = chain.run(user_question)
    except Exception as e:
        raise e
    return output


def init_base_embedding_chain(llm_model, embedding_model, knowledge_database, user_question):
    if knowledge_database:
        template = general_template()
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        vectorDB = Milvus(
            embedding_model,
            connection_args={
                "host": MILVUS_HOST,
                "port": MILVUS_PORT
            },
            collection_name=knowledge_database,
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            chain_type="stuff",
            retriever=vectorDB.as_retriever(),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, "verbose": True}
        )
        try:
            output = qa_chain.run(user_question)
        except Exception as e:
            raise e
        return output


def sheet_to_string(sheet, sheet_name=None):
    result = []
    for index, row in sheet.iterrows():
        row_string = ""
        for column in sheet.columns:
            row_string += f"{column}: {row[column]}, "
        row_string = row_string.rstrip(", ")
        row_string += "."
        result.append(row_string)
    return result


def excel_to_string(file_path):
    # 读取Excel文件中的所有工作表
    excel_file = pd.read_excel(file_path, engine='openpyxl', sheet_name=None)

    # 初始化结果字符串
    result = []

    # 遍历每一个工作表
    for sheet_name, sheet_data in excel_file.items():
        # 处理当前工作表并添加到结果字符串
        result += sheet_to_string(sheet_data, sheet_name=sheet_name)

    return result


def get_documents(file_src):
    from langchain.schema import Document
    from langchain.text_splitter import TokenTextSplitter
    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=30)

    documents = []
    for file in file_src:
        filepath = file.name
        filename = os.path.basename(filepath)
        file_type = os.path.splitext(filename)[1]
        try:
            if file_type == ".pdf":
                pdftext = ""
                with open(filepath, "rb") as pdfFileObj:
                    pdfReader = PyPDF2.PdfReader(pdfFileObj)
                    for page in tqdm(pdfReader.pages):
                        pdftext += page.extract_text()
                texts = [Document(page_content=pdftext,
                                  metadata={"source": filepath})]
            elif file_type == ".docx":
                from langchain.document_loaders import UnstructuredWordDocumentLoader
                loader = UnstructuredWordDocumentLoader(filepath)
                texts = loader.load()
            elif file_type == ".pptx":
                from langchain.document_loaders import UnstructuredPowerPointLoader
                loader = UnstructuredPowerPointLoader(filepath)
                texts = loader.load()
            elif file_type == ".epub":
                from langchain.document_loaders import UnstructuredEPubLoader
                loader = UnstructuredEPubLoader(filepath)
                texts = loader.load()
            elif file_type == ".xlsx":
                text_list = excel_to_string(filepath)
                texts = []
                for elem in text_list:
                    texts.append(Document(page_content=elem,
                                          metadata={"source": filepath}))
            else:
                from langchain.document_loaders import TextLoader
                loader = TextLoader(filepath, "utf8")
                texts = loader.load()
        except Exception as e:
            raise e

        texts = text_splitter.split_documents(texts)
        documents.extend(texts)
    return documents


def load_embedding_chain_file(fileobj=None, embedding_model=None):
    if embedding_model is None:
        llm_model, embedding_model = init_model(llm_model_name=None, embedding_model_name=None, temperature=0.7, max_tokens=2000)
    if fileobj:
        filepath = fileobj.name
        print(filepath)
        bookname = f"temp{random.randint(0, 100000)}"
        docs = get_documents([fileobj])
        vectorDB = Milvus.from_documents(
            docs,
            embedding_model,
            connection_args={
                "host": MILVUS_HOST,
                "port": MILVUS_PORT
            },
            collection_name=bookname,
            drop_old=True  # 是否删除旧的collection
        )
        return vectorDB, bookname


def load_embedding_chain_url(url=None, embedding_model=None):
    if embedding_model is None:
        llm_model, embedding_model = init_model(llm_model_name=None, embedding_model_name=None, temperature=0.7, max_tokens=2000)
    if url:
        filepath = url
        print(filepath)
        bookname = f"temp{random.randint(0, 100000)}"
        docs = get_documents([url])
        vectorDB = Milvus.from_documents(
            docs,
            embedding_model,
            connection_args={
                "host": MILVUS_HOST,
                "port": MILVUS_PORT
            },
            collection_name=bookname,
            drop_old=True  # 是否删除旧的collection
        )
        return vectorDB, bookname


###gradio
block = gr.Blocks(css="footer {visibility: hidden}", title="文言一心助手")
with block:
    history = ConversationBufferMemory()
    history_state = gr.State(history)  # 历史记录的状态
    llm_model_state = gr.State()  # llm模型的状态
    embedding_model_state = gr.State()  # embedding模型的状态
    milvus_books = None
    milvus_books_state = gr.State(milvus_books)  # milvus_books的状态
    trash = gr.State()  # 垃圾桶

    with gr.Row():
        # 设置行

        with gr.Column(scale=1):
            with gr.Accordion("模型配置", open=False):
                llm_model_name = gr.Dropdown(
                    choices=llm_model, value=llm_model[0], label="语言模型", multiselect=False, interactive=True
                )
                embedding_model_name = gr.Dropdown(
                    choices=embedding_model, value=embedding_model[0], label="embedding模型", multiselect=False, interactive=True
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="temperature",
                    interactive=True,
                )
                max_tokens = gr.Slider(
                    minimum=1,
                    maximum=16384,
                    value=1000,
                    step=1,
                    label="max_tokens",
                    interactive=True,
                )
                modle_settings = gr.Button("应用")

            use_database = gr.Radio(["是", "否"],
                                    label="是否使用知识库",
                                    value="否")

            with gr.Accordion("知识库选项", open=False):
                with gr.Tab("上传"):
                    file = gr.File(label='上传知识库文件',
                                   file_types=['.txt', '.md', '.docx', '.pdf', '.pptx', '.epub', '.xlsx'])
                    init_dataset_upload = gr.Button("应用")
                with gr.Tab("链接载入"):
                    knowledge_url_box = gr.Textbox(
                        label="url载入知识库",
                        placeholder="请粘贴你的知识库url",
                        show_label=True,
                        lines=1
                    )
                    init_dataset_url = gr.Button("应用")

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="文心酱")
            with gr.Row():
                message = gr.Textbox(
                    label="在此处填写你的问题",
                    placeholder="我有很多问题想问你......",
                    lines=1,
                )
            with gr.Row():
                submit = gr.Button("发送", variant="primary")
                # 刷新
                clear = gr.Button("刷新", variant="secondary")


            def clear_():
                chatbot = []
                history_state = ConversationBufferMemory()
                return "", chatbot, history_state


            def user(user_message, history):
                return "", history + [[user_message, None]]


            def bot(user_message,
                    chatbot=None,
                    history_state=ConversationBufferMemory(),
                    temperature=None,
                    max_tokens=None,
                    llm_model=None,
                    embedding_model=None,
                    llm_model_name=None,
                    embedding_model_name=None,
                    use_database=None,
                    milvus_books_state=None):
                try:
                    user_message = chatbot[-1][0]
                    if llm_model is None or embedding_model is None:
                        llm_model, embedding_model = init_model(llm_model_name, embedding_model_name, temperature, max_tokens)
                    if use_database == "否":
                        output = init_base_chain(llm_model, history=history_state, user_question=user_message)
                    else:
                        output = init_base_embedding_chain(llm_model, embedding_model, milvus_books_state, user_question=user_message)
                except Exception as e:
                    raise e
                chatbot[-1][1] = ""
                for character in output:
                    chatbot[-1][1] += character
                    time.sleep(0.03)
                    yield chatbot
    # 是否使用知识库
    use_database.change(is_use_database, inputs=[chatbot, use_database], outputs=[chatbot])
    # 模型配置
    modle_settings.click(init_model, inputs=[llm_model_name, embedding_model_name, temperature, max_tokens],
                         outputs=[llm_model_state, embedding_model_state]).then(
        apply_model_setting, inputs=[llm_model_name, embedding_model_name, chatbot], outputs=[chatbot]
    )
    # 知识库选项
    init_dataset_upload.click(pre_embeding_file, inputs=[chatbot], outputs=[chatbot]).then(load_embedding_chain_file, inputs=[file, embedding_model_state],
                                                                                           outputs=[trash, milvus_books_state]).then(applydata_,
                                                                                                                                     inputs=[chatbot],
                                                                                                                                     outputs=[chatbot])
    init_dataset_url.click(pre_embeding_file, inputs=[chatbot], outputs=[chatbot]).then(load_embedding_chain_url,
                                                                                        inputs=[knowledge_url_box, embedding_model_state],
                                                                                        outputs=[trash, milvus_books_state]).then(applydata_, inputs=[chatbot],
                                                                                                                                  outputs=[chatbot])

    # 刷新按钮
    clear.click(clear_, inputs=[], outputs=[message, chatbot, history_state])
    # send按钮
    submit.click(user, [message, chatbot], [message, chatbot], queue=False).then(
        bot,
        [message, chatbot, history_state, temperature, max_tokens, llm_model_state, embedding_model_state, llm_model_name, embedding_model_name, use_database,
         milvus_books_state], [chatbot]
    )
    # 回车
    message.submit(user, [message, chatbot], [message, chatbot], queue=False).then(
        bot,
        [message, chatbot, history_state, temperature, max_tokens, llm_model_state, embedding_model_state, llm_model_name, embedding_model_name, use_database,
         milvus_books_state], [chatbot]
    )

if __name__ == "__main__":
    # 启动参数
    block.queue(concurrency_count=config['block']['concurrency_count']).launch(
        debug=config['block']['debug'],
        server_name=config['block']['server_name'],
        server_port=config['block']['server_port'],
    )
