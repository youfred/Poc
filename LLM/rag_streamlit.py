import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.callbacks import get_openai_callback

# 이미지 경로
Hyundai_logo = "LLM/images/Hyundai_logo.png"
horizontal_logo = "LLM/images/Hyundai_logo_horizen.png"

def main():
    st.set_page_config(
        page_title="Hyundai Motor Company - Motor Vehicle Law ",
        page_icon=Hyundai_logo
    )

    st.title("_:blue[Hyundai Motor]_ - Motor Vehicle Law Data :blue[QA Chatbot] :scales:")
    st.markdown("Hyundai Motor Company & Handong Global University")

    # 로고 출력
    st.image(horizontal_logo, use_column_width=True)
    st.sidebar.image(Hyundai_logo, use_column_width=True)
    st.sidebar.markdown("법률 문서를 사이드바에 업로드하고 OpenAI API 키를 입력한 후 'Process'를 누르세요!")

    # 상태 초기화
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    with st.sidebar:
        process = st.button("Start Chatting")

    # 채팅 시작 시 벡터 스토어 로드
    if process:
        vectorstore = load_vectorstore('LLM/db/faiss')
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.processComplete = True

    # 기본 메시지 설정
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "assistant", "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}
        ]

    # 메시지 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 채팅 입력 및 처리
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            if chain is not None:
                with st.spinner("Thinking..."):
                    result = chain({"question": query})
                    response = result['answer']
                    source_documents = result['source_documents']

                    st.markdown(f"**답변:** {response}")
                    with st.expander("참고 문서 확인"):
                        for doc in source_documents:
                            st.markdown(f"문서: {doc.metadata['source']}", help=doc.page_content)

                    st.session_state.messages.append({"role": "assistant", "content": response})

def load_vectorstore(db_path):
    # 임베딩 모델 로드
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # FAISS 벡터 스토어 로드
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def get_conversation_chain(vectorstore):
    # 인터넷을 통해 Llama 모델 다운로드
    model_name = "akjindal53244/Llama-3.1-Storm-8B"
    
    # 토크나이저와 모델을 Hugging Face 허브에서 다운로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 모델 파이프라인 생성
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0
    )

    # HuggingFacePipeline을 사용해 LLM 생성
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # ConversationalRetrievalChain 생성
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
