# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:02:13 2024

@author: Rise Networks
"""
#__import__('pysqlite3')
#import sys

#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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
from langchain.llms import HuggingFacePipeline
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

    # create the question widget
    question_widget = st.chat_input("Ask your question")
        
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
            
            
            # Create an instance of the text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1500,
                chunk_overlap = 150,
            )
            
            # Applying the text splitter instance
            chunk = text_splitter.split_documents(doc)
            
                
            # Create an instance of the embedding
            embeddings = GPT4AllEmbeddings()
            
                
            # Instantiate the vector database
            db = Chroma.from_documents(
                embedding= embeddings, 
                documents = chunk
            )
                
                            
            # creating the retriever
            retriever = db.as_retriever()
            
                
            # Instantating our model
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
                
            # Instantiating the tokenizer from transformers
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
               
            # creating the transformer's pipeline
            pipe = pipeline(
                "text2text-generation",
                model,
                tokenizer = tokenizer,
                max_length = 512,
                repetition_penalty = 1.15,
            )
                
            # creating Huggingface pipeline
            huggingface_llm = HuggingFacePipeline(
                pipeline=pipe
            )
                
            # creating the retrievalqa
            retrieval = RetrievalQA.from_chain_type(
                llm = huggingface_llm,
                chain_type = "stuff",
                retriever = retriever,
                return_source_documents = True
            )
        
        
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


        
        
 
# run the app function    
if __name__=="__main__":
    app()
