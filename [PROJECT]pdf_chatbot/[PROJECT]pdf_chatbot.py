# This repository reads any pdf document and uses local llama3 and RAG to chat with the document and get answers.
# Run this code in terminal using - python pdf_chatbot.py

import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama


def process_pdf(pdf_bytes):
  """
  Processes uploaded PDF and returns necessary data structures.

  Args:
      pdf_bytes (bytes): Bytes of the uploaded PDF file.

  Returns:
      tuple: Tuple containing text splitter, vectorstore, and retriever objects.
  """
  if pdf_bytes is None:
    return None, None, None

  loader = PyMuPDFLoader(pdf_bytes)
  data = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) # 800, 400
  chunks = text_splitter.split_documents(data)

  embeddings = OllamaEmbeddings(model="llama3")
  vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
  retriever = vectorstore.as_retriever()

  return text_splitter, vectorstore, retriever


def combine_docs(docs):
    """
    Combines page content of retrieved documents into a single string.

    Args:
    docs (list): List of retrieved documents.

    Returns:
    str: Combined content of retrieved documents.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def ollama_llm(question, context):
    """
    Sends a prompt to Ollama for question answering.

    Args:
        question (str): User's question about the PDF.
        context (str): Combined content of retrieved documents.

    Returns:
        answer and retrieved documents.
    """
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model="llama3", messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']
        


def rag_chain(question, text_splitter, vectorstore, retriever):
  """
  Retrieves relevant documents from the vector store and uses Ollama for answering.

  Args:
      question (str): User's question about the PDF.
      text_splitter (object): Text splitter object.
      vectorstore (object): Chroma vector store object.
      retriever (object): Retriever object from the vector store.

  Returns:
      dict: Response dictionary containing answer and retrieved documents.
  """

  retrieved_docs = retriever.invoke(question)
  formatted_content = combine_docs(retrieved_docs)
  return ollama_llm(question, formatted_content)


def ask_question(pdf_bytes, question):
  """
  Asks a question about the PDF and retrieves an answer using RAG.

  Args:
      pdf_bytes (bytes): Bytes of the uploaded PDF file (can be None).
      question (str): User's question about the PDF.

  Returns:
      dict (or None): Response dictionary containing answer and citations 
                        if PDF is uploaded, otherwise None.
  """
  text_splitter, vectorstore, retriever = process_pdf(pdf_bytes)
  if text_splitter is None:
    return None  # No PDF uploaded

  result = rag_chain(question, text_splitter, vectorstore, retriever)
  return {result}

#   citations = [doc.metadata["source"] for doc in result["source_documents"]]
  
#   return {"answer": result["result"], "citations": ", ".join(citations)}


interface = gr.Interface(
    fn=ask_question,
    inputs=[gr.File(label="Upload PDF (optional)"), gr.Textbox(label="Ask a question")],
    outputs="text",
    title="Ask questions about your PDF",
    description="Use local Llama3 to answer your questions about the uploaded PDF document.",
)

interface.launch()
