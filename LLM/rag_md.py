import streamlit as st
from io import BytesIO
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit 앱 설정
st.title("Document Upload and Splitter")

def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name
        file_content = doc.getvalue()

        logger.info(f"Uploaded {file_name}")

        if file_name.endswith('.pdf'):
            loader = PyPDFLoader(BytesIO(file_content))
            documents = loader.load_and_split()
            for document in documents:
                document.metadata = {"source": file_name}
        
        elif file_name.endswith('.docx'):
            loader = Docx2txtLoader(BytesIO(file_content))
            documents = loader.load_and_split()
            for document in documents:
                document.metadata = {"source": file_name}

        elif file_name.endswith('.pptx'):
            loader = UnstructuredPowerPointLoader(BytesIO(file_content))
            documents = loader.load_and_split()
            for document in documents:
                document.metadata = {"source": file_name}

        elif file_name.endswith('.md'):
            markdown_content = file_content.decode('utf-8')
            documents = [{"page_content": markdown_content, "metadata": {"source": file_name}}]

        doc_list.extend(documents)

    return doc_list

def split_markdown_by_headers(markdown_document):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # Markdown 헤더를 기준으로 텍스트를 분할하는 MarkdownHeaderTextSplitter 객체를 생성합니다.
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # markdown_document를 헤더를 기준으로 분할하여 md_header_splits에 저장합니다.
    md_header_splits = markdown_splitter.split_text(markdown_document)

    return md_header_splits

def split_text_recursively(text):
    # 재귀적 텍스트 분할기를 생성합니다.
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 청크 크기
        chunk_overlap=200  # 청크 오버랩
    )
    # 텍스트를 재귀적으로 분할합니다.
    text_chunks = recursive_splitter.split_text(text)
    return text_chunks

def main():
    uploaded_files = st.file_uploader("파일 업로드", accept_multiple_files=True)

    if uploaded_files:
        # 업로드된 파일로부터 텍스트를 가져옵니다.
        documents = get_text(uploaded_files)

        for document in documents:
            # 각 문서의 페이지 내용을 가져옵니다.
            page_content = document.page_content
            
            # 문서가 마크다운 형식일 경우 헤더로 분할
            if document.metadata.get("source", "").endswith('.md'):
                md_splits = split_markdown_by_headers(page_content)
                for split in md_splits:
                    st.markdown(split)

            # 문서 내용을 재귀적으로 분할
            recursive_chunks = split_text_recursively(page_content)
            st.write("재귀적 분할 결과:")
            for chunk in recursive_chunks:
                st.markdown(chunk)

            # 문서 출처 표시
            st.markdown(f"소스: {document.metadata['source']}")

if __name__ == "__main__":
    main()
