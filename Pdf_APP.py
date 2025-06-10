from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
import streamlit as st
import io
from dotenv import load_dotenv
load_dotenv()
import os,tempfile

st.title("PDFBot: Content retriever Tool 📄")
st.sidebar.title("Uploaded documents")

st.subheader("Upload PDF Files")
pdf1 = st.file_uploader("Upload First PDF", type="pdf", key="pdf1")
pdf2 = st.file_uploader("Upload Second PDF (optional)", type="pdf", key="pdf2")
pdf3 = st.file_uploader("Upload Third PDF (optional)", type="pdf", key="pdf3")

pdf_paths=[
    pdf1,pdf2,pdf3]



def document_loader(pdf_paths):
    full_doc=[]
    for uploaded_file in pdf_paths:
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            loader = PyMuPDFLoader(file_path=tmp_file_path)
            data = loader.load()
            full_doc.extend(data)
            os.unlink(tmp_file_path)
    return full_doc

def text_splitter(pdf_paths):
    text=document_loader(pdf_paths)
    if not text:
        return
    splitter=RecursiveCharacterTextSplitter(separators=["\n\n","\n","." ],chunk_size=700,chunk_overlap=100)

    docs=splitter.split_documents(text)
    return docs

def create_embeddings(pdf_paths):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    chunks=text_splitter(pdf_paths)
    if not chunks:
        return None
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)
    vectorstore.save_local("faiss_index")
    return vectorstore

llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.7)
process_pdf_clicked = st.sidebar.button("Process selected documents")
main_placeholder = st.empty()


if process_pdf_clicked:
    if any(pdf_paths):
        with main_placeholder:
            st.text("Text Splitter...... Started...✅✅✅")
            time.sleep(1)
            vectorstore=create_embeddings(pdf_paths)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                time.sleep(1)
            else:
                st.error("No valid PDFs uploaded or processing failed")
    else:
        st.error("please upload at least one pdf file")

vectorstore=st.session_state.get("vectorstore",None)
if not vectorstore and os.path.exists("faiss_index"):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
    st.session_state.vectorstore = vectorstore

query = main_placeholder.text_input("Question: ")
if query:
    if vectorstore:
        with st.spinner("Processing...."):
            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorstore.as_retriever())
            result=chain({"question":query},return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)

# with main_placeholder:
#     query = st.text_input("Question: ", placeholder="e.g., What is feature engineering?", key="query_input")
#     if query:
#         if vectorstore:
#             with st.spinner("Processing...."):
#                 chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorstore.as_retriever())
#                 result=chain({"question":query},return_only_outputs=True)
#                 st.session_state.result=result
#         else:
#             st.error("Please upload and process PDFs first.")
#     else:
#         st.write("Enter a question to get an answer.")
#
# if "result" in st.session_state:
#     result = st.session_state.result
#     st.header("Answer")
#     st.write(result.get("answer", "No answer found."))
#     st.subheader("Sources:")
#     sources = result.get("sources", "")
#     if sources:
#         sources_list = sources.split("\n")
#         for source in sources_list:
#             st.write(source)
#     else:
#         st.write("No sources provided. ")




