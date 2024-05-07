# Code for this project is inspired from (https://twitter.com/Saboo_Shubham_/status/1785858499672670472?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1785858499672670472%7Ctwgr%5E5da60a5b68b8bc90341ad8c9f510e38a9fb13e88%7Ctwcon%5Es1_&ref_url=https%3A%2F%2Fanalyticsindiamag.com%2F10-wild-use-cases-for-llama-3%2F)
# To run this code - in terminal run (streamlit run RAG_with_llama3_webpage_chatbot.py )
import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

st.title("Chat with Webpage")
st.caption("Web application to chat with any webpage using local llama3 and RAG")

webpage_url = st.text_input("Enter the URL of the webpage: ", type = "default")

# load webpage url using webbaseloader --> split the document using TextSplitter --> load llama3 model --> load Chroma vectorstore to store embeddings
if webpage_url:
    loader = WebBaseLoader(webpage_url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 10)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model = "llama3")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    # call llama3 model  (RAG - Generation part)
    def ollama_llm(question, context):
        formatted_prompt = f"Question: {question}\n\nContext: {context}"
        response = ollama.chat(model = "llama3", messages = [{'role': 'user', 'content': formatted_prompt}])
        return response['message']['content']

    # RAG - Retriever part
    retriever = vectorstore.as_retriever()

    # combine all the documents to get the content of the webpage
    def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG - Chain (Retriever + Generator)
    def rag_chain(question):
        retrieved_docs = retriever.invoke(question)
        formatted_content = combine_docs(retrieved_docs)
        return ollama_llm(question, formatted_content)
    
st.success(f"Loaded webpage successfully")

prompt = st.text_input("Ask any question about the webpage")
prompt_with_url = prompt + " " + webpage_url

if prompt:
    result = rag_chain(prompt_with_url)
    st.write(result)

