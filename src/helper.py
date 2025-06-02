from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers 
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory 
from langchain_core.runnables import RunnableParallel, RunnablePassthrough,  RunnableLambda
 


#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

#Download the Embeddings from Hugging Face
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

def get_recent_chat_history(x, k=3):
    history = memory.load_memory_variables(x)["chat_history"]
    return history[-k:] if len(history) > k else history


# 6. Wrap in memory.save_context for chat tracking
def ask_question(user_input):
    inputs = {"question": user_input}
    response = retrieval_chain.invoke(inputs)
    memory.save_context(inputs, {"output": response})
    return response
