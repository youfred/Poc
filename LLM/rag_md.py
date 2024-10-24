import streamlit as st
import tiktoken
from loguru import logger
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from io import BytesIO
from langchain.memory import StreamlitChatMessageHistory

# 로고 관련 이미지
Hyundai_logo = "LLM/images/Hyundai_logo.png"
horizontal_logo = "LLM/images/Hyundai_logo_horizen.png"

def main():
    st.set_page_config(
        page_title="Hyundai Motor Company - Motor Vehicle Law",
        page_icon=Hyundai_logo
    )

    st.title("_:blue[Hyundai Motor]_ - Motor Vehicle Law Data :blue[QA Chatbot] :scales:")
    st.markdown("Hyundai Motor Company & Handong Global University")
    
    # 사이드바
    st.image(
        horizontal_logo,
        use_column_width=True
    )
    st.sidebar.markdown("법률 문서를 업로드하세요. OpenAI API 키를 입력하고 '처리' 버튼을 눌러주세요!")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("파일 업로드", type=['pdf', 'docx', 'pptx', 'md'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("처리")

    if process:
        if not openai_api_key:
            st.info("계속하려면 OpenAI API 키를 입력하세요.")
            st.stop()
        files_text = get_text(uploaded_files)  # 문서 텍스트 가져오기
        text_chunks = get_text_chunks(files_text)  # 텍스트 청크로 분할
        vectorstore = get_vectorstore(text_chunks)  # 벡터스토어 생성

        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 법률 문서에 대해 질문이 있으면 자유롭게 물어보세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # 채팅 로직
    if query := st.chat_input("질문을 입력하세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            if chain is not None:
                with st.spinner("생각 중..."):
                    result = chain({"question": query})
                    response = result['answer']
                    source_documents = result['source_documents']

                    st.markdown(response)
                    with st.expander("참고 문서 확인"):
                        for doc in source_documents:
                            source_info = doc.metadata.get('source', '소스 정보가 없습니다.')
                            st.markdown(source_info)

                    st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name
        # BytesIO를 사용하여 메모리에 저장
        file_content = doc.getvalue()
        
        logger.info(f"Uploaded {file_name}")
        
        # PDF 처리
        if file_name.endswith('.pdf'):
            loader = PyPDFLoader(BytesIO(file_content))
            documents = loader.load_and_split()

        # DOCX 처리
        elif file_name.endswith('.docx'):
            loader = Docx2txtLoader(BytesIO(file_content))
            documents = loader.load_and_split()

        # PPTX 처리
        elif file_name.endswith('.pptx'):
            loader = UnstructuredPowerPointLoader(BytesIO(file_content))
            documents = loader.load_and_split()

        # MD 처리 (마크다운 파일)
        elif file_name.endswith('.md'):
            markdown_content = file_content.decode('utf-8')
            documents = [{"page_content": markdown_content, "metadata": {"source": file_name}}]  # 메타데이터 포함

        doc_list.extend(documents)
    
    return doc_list

def get_text_chunks(texts):
    # 마크다운 헤더에 따라 분할
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=["#", "##", "###"])
    
    # 입력 형식 확인
    if not isinstance(texts, list):
        logger.error("Input to get_text_chunks is not a list.")
        return []
    
    # 각 텍스트가 'page_content'를 포함한 딕셔너리인지 확인
    for text in texts:
        if not isinstance(text, dict) or 'page_content' not in text:
            logger.error(f"Invalid document format: {text}")
            return []

    header_chunks = header_splitter.split_documents(texts)

    # 이후 RecursiveCharacterTextSplitter를 사용하여 청크 생성
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    
    final_chunks = []
    for doc in header_chunks:
        if isinstance(doc, dict) and 'page_content' in doc:
            chunks = text_splitter.split_documents([doc])  # 문서별로 청크 분할
            final_chunks.extend(chunks)
        else:
            logger.warning("Unexpected document format: {}".format(doc))

    return final_chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

if __name__ == '__main__':
    main()
