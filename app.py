import streamlit as st
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

import os
import shutil
import tempfile

# --- Functions from your code ---

def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(extracted_data)

def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def get_recent_chat_history(x, k=3):
    history = memory.load_memory_variables(x)["chat_history"]
    return history[-k:] if len(history) > k else history

# --- UI ---

st.set_page_config(page_title="MediChat", layout="wide")
st.title("👨‍⚕️ Medical Chatbot (RAG with Local Llama)")

# --- Upload PDFs ---
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing..."):
        # Save uploaded files to a temp folder
        temp_dir = tempfile.mkdtemp()
        for file in uploaded_files:
            with open(os.path.join(temp_dir, file.name), "wb") as f:
                f.write(file.read())

        # Load and process PDFs
        extracted_data = load_pdf_file(temp_dir)
        text_chunks = text_split(extracted_data)
        texts = [doc.page_content for doc in text_chunks]

        # Vectorstore
        persist_directory = "./chroma_index"
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        embeddings = download_hugging_face_embeddings()
        vectorstore = Chroma.from_texts(texts=texts, embedding=embeddings,
                                        persist_directory=persist_directory)
        retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k':2})

        # Prompt
        prompt_template = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}"
        )

        prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

        # Local LLM
        llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                            model_type="llama",
                            config={'max_new_tokens': 256, 'temperature': 0.8})

        # Memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question")

        # Chain
        doc_chain = create_stuff_documents_chain(llm, prompt)

        retrieval_chain = (
            RunnableParallel({
                "context": RunnableLambda(lambda x: retriever.invoke(x["question"])),
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(lambda x: get_recent_chat_history(x))
            }) | doc_chain
        )

        # Store in session state
        st.session_state['retrieval_chain'] = retrieval_chain
        st.session_state['memory'] = memory

        st.success("PDFs processed. You can now ask questions!")

# --- Chat Interface ---
if 'retrieval_chain' in st.session_state:
    user_question = st.text_input("Ask a question about the PDFs:", key="question_input")

    if user_question:
        with st.spinner("Thinking..."):
            inputs = {"question": user_question}
            response = st.session_state['retrieval_chain'].invoke(inputs)
            st.session_state['memory'].save_context(inputs, {"output": response})

            st.markdown(f"**Answer:** {response}")
