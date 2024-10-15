# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:02:13 2024

@author: Rise Networks
"""
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForDocumentQuestionAnswering
from transformers import pipeline
from langchain_community.llms import huggingface_pipeline
from langchain.chains import RetrievalQA

# Setting the background color
st.markdown("""
<style>
body {
  background-color: #f0f2f6;
}
</style>
""", unsafe_allow_html=True
)


def app():
    
    # Upload a document symbol
    container = st.sidebar.container()
        
    # Import a side bar where you can upload the document [pdf, docx]    
    uploaded_file = st.sidebar.file_uploader(
        "",
        type=["pdf", "docx", "txt"], 
    )
        
    # Initialize doc
    doc = None
    
    # Confirm that a document has been uploaded!!!
    if uploaded_file == None:
        # Prompt them to upload
        with container:
            st.sidebar.warning("Upload a document", icon="⚠️")
    else:
            
        if uploaded_file != None and uploaded_file.name[-3:] == "pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf",) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name  # Get the temp file path
            
            # Displaying the file path or filename
            with container:
                st.sidebar.success("Document Uploaded", icon="✅")
            
            # Load the pdf document using PyPDFLoader
            loader = PyPDFLoader(temp_file_path)
            
            # Load it properly
            doc = loader.load()
            
            def split_document(doc):
                # Create an instance of the text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = 1500,
                    chunk_overlap = 150,
                )
            
                # Applying the text splitter instance
                chunk = text_splitter.split_documents(doc)
            
                # return the smaller chunks of the document
                return chunk
            
            # Displaying the first item in the document
            #st.write(
                #split_document(doc)[0]
            #)
            
            # create embeddings using GPT4All Embeddingss
            def embedding():
                
                # Create an instance of the embedding
                embeddings = GPT4AllEmbeddings()
                
                # return the embeddings
                return embeddings
            
            # create a vector database using Chroma
            def vector_database():
                
                # Instantiate the vector database
                db = Chroma.from_documents(
                    embedding= embedding(), 
                    documents = split_document(doc)
                )
                
                # return the database
                return db
            
            # creating a retriever
            def retriever():
                
                # loading the database
                db = vector_database()
                
                # creating the retriever
                retriever = db.as_retriever()
                
                # return the retriever
                return retriever
            
            # creating a transformer pipeline
            def pipeline():
                
                # Instantating our model
                model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
                
                # Instantiating the tokenizer from transformers
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
               
                # creating the transformer's pipeline
                pipe = pipeline(
                    task = "question-answering",
                    model = model,
                    tokenizer = tokenizer,
                    max_length = 512,
                    repetition_penalty = 1.15,
                )
                
                # creating Huggingface pipeline
                huggingface_llm = huggingface_pipeline(pipe)
                
                # return the huggingface_pipeline
                return huggingface_llm
            
            # creating the RetrievalQA
            def retrieval_qa():
                
                # creating the retrievalqa
                retrieval = RetrievalQA.from_chain_type(
                    llm = pipeline(),
                    chain_type = "stuff",
                    retriever = retriever(),
                    return_source_documents = True
                )
                
                # return the retrievalqa
                return retrieval
        
        
        elif uploaded_file != None and uploaded_file.name[-3:] == "txt":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt",) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name  # Get the temp file path
            
            # Displaying the file path or file name
            with container:
                st.sidebar.success("Document Uploaded", icon="✅")
            
            # Load the pdf document using PyPDFLoader
            loader = TextLoader(temp_file_path)
            
            # Load it properly
            doc = loader.load()
            
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx",) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name  # Get the temp file path
            
            # Displaying the file path or file name
            with container:
                st.sidebar.success("Document Uploaded", icon="✅")
            
            # Load the pdf document using PyPDFLoader
            loader = Docx2txtLoader(temp_file_path)
            
            # Load it properly
            doc = loader.load()
            
            
    # create the question widget
    question_widget = st.chat_input("Ask your question")
        
    if question_widget:
        # Load the retrieval_qa
        qa = retrieval_qa()
            
        # Pass the asked question into the retrievalqa
        ans = qa(question_widget)
            
        # return the annswer
        st.write(ans["result"])
    else:
        st.warning("Ask your question", icon="❗")
  
        
        
 
# run the app function    
if __name__=="__main__":
    app()