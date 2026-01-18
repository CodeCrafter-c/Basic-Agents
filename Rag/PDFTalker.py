from langchain_community.document_loaders import PyPDFLoader
from  pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = Path(__file__).resolve().parent /"nodejs.pdf"      
loader=PyPDFLoader(file_path=pdf_path)

docs=loader.load()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs=text_splitter.split_documents(documents=docs)
print(len(split_docs))