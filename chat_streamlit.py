# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:20:11 2024

@author: Rise Networks
"""

import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from langchain_chroma import Chroma
#from transformers import pipeline
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import os

# File Uploading
uploaded_file = st.sidebar.file_uploader(
    "Upload document", ["pdf", "docx", "txt"]
)

if uploaded_file != None:
    st.sidebar.success("Document Uploaded", icon="✔️")
else:
    st.sidebar.warning("Upload a document", icon="⚠️")
    
# Creating the chat box
container = st.container(border=True, height=300,)
with container:
    st.header("Chat with Document 🤖")
    
question = st.chat_input("Ask a question")


# creating a function for the type of document uploaded
def load_file(document):
    document = uploaded_file
    if uploaded_file and uploaded_file.name[-3:] == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix="pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            tempfile_path = temp_file.name
       #doc = Py
    elif uploaded_file and uploaded_file.name[-3] == "txt":
        with tempfile.NamedTemporaryFile(delete=False, suffix="txt") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            tempfile_path = temp_file.name
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix="docx") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            tempfile_path = temp_file.name

    return tempfile_path

# creating the rag function
def rag():
    
    if uploaded_file:
        # Initializing the document
        document_path = load_file(uploaded_file)
    
        # creating conditions to affirm the file format or type
        if document_path[-3:] == "pdf":
            doc = PyPDFLoader(document_path)
        elif  document_path[-3:] == "txt":
            doc = TextLoader(document_path)
        else:
            doc = Docx2txtLoader(document_path)
    
        # Loading the documents
        doc = doc.load()
    
        # Converting the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150
        )
        chunks = text_splitter.split_documents(doc)
        
        # create embeddings using GPT4All Embeddings or HuggingFace Embeddings
        #embeddings = HuggingFaceEmbeddings()
        #model_name = "sentence-transformers/all-mpnet-base-v2"
        #model_kwargs = {'device': 'cpu'}
        #encode_kwargs = {'normalize_embeddings': False}
        #hf = HuggingFaceEmbeddings(
            #model_name=model_name,
            #model_kwargs=model_kwargs,
            #encode_kwargs=encode_kwargs
        #)
        
        #gpt4all_kwargs = {'allow_download': False}
        #embeddings = GPT4AllEmbeddings(
        #model_name=r"C:\Users\Rise Networks\gpt4all\resources\nomic-embed-text-v1.5.f16.gguf",
        #gpt4all_kwargs=gpt4all_kwargs
        #)
        
        embeddings = FastEmbedEmbeddings(model_name = "BAAI/bge-base-en-v1.5")
    
        # Creating a vector database
        db = Chroma.from_documents(embedding=embeddings,documents=chunks)
    
        # creating retriever
        retriever = db.as_retriever()
        
        
        # Creating the LLM
        groq_llm = ChatGroq(
            model=  "llama3-70b-8192",
            temperature = 0,
            max_tokens = 1024,
            max_retries = 2,
            api_key = os.environ["GROQ_API_KEY"]
        )
    
        # creating the retrieval qa
        qa = RetrievalQA.from_chain_type(
            llm = groq_llm,
            chain_type = "stuff",
            retriever = retriever,
            return_source_document = True
        )
    
        return qa
    
    
# define app
def app():
     
    # Document uploading 
    if uploaded_file:
        result = load_file(uploaded_file)
        st.write(result)
    else:
        st.warning("Upload a document", icon="❗")
        
    # RAG
    final_rag = rag()
        
    
    # If the question has been asked then do this:
    if question:
        answer = final_rag.invoke(question)
        with container:
            st.text_area("Answer", answer["result"])
            
            st.divider()
    else:
        pass
        
    
if __name__=="__main__":
    app()