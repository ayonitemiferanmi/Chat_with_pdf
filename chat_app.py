# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:02:13 2024

@author: Rise Networks
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain_chroma.vectorstores import Chroma
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
        
    # Import a side bar where you can upload the document [pdf, docx]    
    uploaded_file = st.sidebar.file_uploader(
        "Upload a document", 
        type=["pdf", "docx", "txt"], 
    )
    
    # Confirm that a document has been uploaded!!!
    if uploaded_file == None:
        # Prompt them to upload
        st.warning("Upload a document", icon="⚠️")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf",) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name  # Get the temp file path

        # Load the uploaded file using PyPDFLoader
        def load_document():
        
            # Create an instance of the document loader
            loader = PyPDFLoader(temp_file_path)
        
            # return the loaded document
            return loader.load()
    
        # Split the documents into chunks
        def split_document():
        
            # Load the document
            document = load_document()
        
            # Create an instance of the text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1500,
                chunk_overlap = 150
            )
        
            # creating the chunks or dividing the documents into smaller chunks
            chunks = text_splitter.split_documents(document)
        
            # return the smaller chunks
            return chunks
    
        # Create embeddingssss
        def create_embeddings():
        
            # create an instance of the embeddings
            embedding = GPT4AllEmbeddings()
                
            # return the embedded documents
            return embedding
    
        # Creating a vector database
        def vector_database():
        
            # Creating the vector database
            db = Chroma.from_documents(
                documents = split_document(),
                embedding = create_embeddings()
            )
        
            # returning the created database
            return db
    
        # Querying our database
        def query_db():
        
            # creating a question
            question = "When did Nicodemus meet Jesus?"
        
            # using similarity search to affirm
            database = vector_database()
        
            # Provide the answer
            answer = database.similarity_search(question)
        
            # return the answer
            return answer
    
        # Display the answer to confirm what we have
        st.write(
            query_db()
        )
        
        # Creating a retriever
        def retriever():
            
            # instantiating the retriever
            retriever =  vector_database().as_retriever()
            
            # return the retriever
            return retriever
        
        # Creating the LLM using a pipeline
        def create_llm():
            
            # Instantiating our tokenizer
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
            
            # Instantiating our model
            model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large')
            
            # creating our pipeline
            pipe = pipeline(
                "question-answering",
                model = model,
                tokenizer = tokenizer,
                max_length = 512,
                repetition_penalty = 1.15
            )
            
            # creating the llm using huggingface pipeline
            llm = huggingface_pipeline(
                pipe,
            )
            
            # return the llm
            return llm
        
        # Creating the QA
        def QA():
            
            # create an instance of the qa
            qa = RetrievalQA.from_chain_type(
                llm = create_llm(),
                chain_type = "stuff",
                retriever = retriever(),
                return_source_documents = True
            )
            
            # return the qa
            return qa
        
        
 
# run the app function    
if __name__=="__main__":
    app()
