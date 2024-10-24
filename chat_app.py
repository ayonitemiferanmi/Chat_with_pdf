# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:02:13 2024

@author: Rise Networks
"""
__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import tempfile
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from langchain_chroma import Chroma
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForDocumentQuestionAnswering
#from transformers import pipeline
#from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

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

    # create the question widget
    #question_widget = st.chat_input("Ask your question")
    
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
                emdeddings_1 = HuggingFaceEmbeddings()
                embeddings_2 = FastEmbedEmbeddings()
                
                # return the embeddings
                return embeddings_1
            
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
            def create_pipeline():
                
                # Instantating our model
                #model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
                
                # Instantiating the tokenizer from transformers
                #tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
               
                # creating the transformer's pipeline
                #pipe = pipeline(
                 #   "text2text-generation",
                  #  model,
                   # tokenizer = tokenizer,
                   # max_length = 512,
                   # repetition_penalty = 1.15,
                #)
                
                # creating Huggingface pipeline
                #huggingface_llm = HuggingFacePipeline(
                  #pipeline=pipe
                #)
              
                # Creating the new llm
                groq_llm = ChatGroq(
                  model = "llama3-70b-8192",
                  temperature = 0,
                  max_tokens = 1024,
                  max_retries = 2,
                  api_key = os.environ["GROQ_API_KEY"]
                )
                
                # return the huggingface_pipeline
                return groq_llm
            
            # creating the RetrievalQA
            def retrieval_qa():
                
                # creating the retrievalqa
                retrieval = RetrievalQA.from_chain_type(
                    llm = create_pipeline(),
                    chain_type = "stuff",
                    retriever = retriever(),
                    return_source_documents = True
                )
                
                # return the retrievalqa
                return retrieval
        qa = retrieval_qa()
        if qa:
          st.write("Trueee")
        
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


    # get the output of the question_answer function
  #output = answer_question()
  #st.write(output)
        
        
 
# run the app function    
if __name__=="__main__":
    app()
