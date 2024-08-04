import streamlit as st
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI

# Load FAISS index and documents
faiss_index_path = "faiss_index.index"
documents_path = "documents.pkl"

index = faiss.read_index(faiss_index_path)
with open(documents_path, "rb") as f:
    documents = pickle.load(f)

# Recreate the FAISS vector store
embeddings = OpenAIEmbeddings()
vectorStore_openAI = FAISS(index=index, embedding_function=embeddings.embed_query)

# Initialize the language model
llm = OpenAI(model_name="text-davinci-003", temperature=0)

# Create the retriever from the FAISS vector store
retriever = vectorStore_openAI.as_retriever()

# Create the RetrievalQAWithSourcesChain
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

# Streamlit app
st.title("Ardent Blue Chatbot")

query = st.text_input("Ask a question about Ardent Blue:")

if query:
    result = chain.invoke({"question": query}, return_only_outputs=True)
    st.write(result)
