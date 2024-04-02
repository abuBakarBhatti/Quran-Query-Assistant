from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os 
import openai


# Retrieve the OpenAI API key from the environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")
# Set the API key for the OpenAI module
openai.api_key = openai_api_key

#hugging face API key
huggingface_api_key = os.environ.get("HUGGINGFACE_HUB_TOKEN")


#reading the raw text file
with open("Quran.txt", encoding="utf-8") as f:
  Quran = f.read()

semantic_chunker = SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-large"), breakpoint_threshold_type="percentile")
semantic_chunks = semantic_chunker.create_documents([Quran])

# huggingface embedding model
embedding_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=huggingface_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
)

# setting up the vector store
semantic_chunk_vectorstore = FAISS.from_documents(semantic_chunks, embedding=embedding_model)

#retriever
semantic_chunk_retriever = semantic_chunk_vectorstore.as_retriever(search_kwargs={"k" : 1})

rag_template = """\
Use the following context to answer the user's query, also try to give reference. If you cannot answer, please respond with 'I'm sorry, I don't know.' and if the user say hello or greeting/how are you something like that you can respond politely and ask him to ask questions regarding Quran.

User's Query:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)

#generatin part
base_model = ChatOpenAI()


semantic_rag_chain = (
    {"context" : semantic_chunk_retriever, "question" : RunnablePassthrough()}
    | rag_prompt
    | base_model
    | StrOutputParser()
)

def answerable(question):
  response = semantic_rag_chain.invoke(question)
  return response