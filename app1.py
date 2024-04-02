# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI
# import os 
# import openai

# import chainlit as cl

# from langchain_core.prompts import ChatPromptTemplate  # Adjust imports as needed
# # ... other imports from rag.py

# #from rag import answerable  # Import your answerable function

# @cl.on_message
# async def main(message: cl.Message):
#     try:
#         # Integrate RAG setup code from rag.py
#         openai_api_key = os.environ.get("OPENAI_API_KEY")
#         openai.api_key = openai_api_key

#         huggingface_api_key = os.environ.get("HUGGINGFACE_HUB_TOKEN")

#         embedding_model = HuggingFaceInferenceAPIEmbeddings(
#             api_key=huggingface_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
#         )

#         semantic_chunker = SemanticChunker(
#             OpenAIEmbeddings(model="text-embedding-3-large"), breakpoint_threshold_type="percentile"
#         )

#         with open("Quran.txt", encoding="utf-8") as f:
#             Quran = f.read()
#         semantic_chunks = semantic_chunker.create_documents([Quran])

#         semantic_chunk_vectorstore = FAISS.from_documents(semantic_chunks, embedding=embedding_model)
#         semantic_chunk_retriever = semantic_chunk_vectorstore.as_retriever(search_kwargs={"k": 1})

#         rag_template = """\
# Use the following context to answer the user's query, also try to give reference. If you cannot answer, please respond with 'I'm sorry, I don't know.'.

# User's Query:
# {question}

# Context:
# {context}
# """
#         rag_prompt = ChatPromptTemplate.from_template(rag_template)

#         base_model = ChatOpenAI()
#         semantic_rag_chain = (
#             {"context": semantic_chunk_retriever, "question": RunnablePassthrough()}
#             | rag_prompt
#             | base_model
#             | StrOutputParser()
#         )
#         def answerable(question):
#              response = semantic_rag_chain.invoke(question)
#              return response

#         response = answerable(message.content)  # Call your RAG function
#         await cl.Message(content=response)

#     except Exception as e:
#         await cl.Message(content="An error occurred: {}".format(str(e)))

# if __name__ == "__main__":
#     cl.run()


# from openai import AsyncOpenAI

# import chainlit as cl

# client = AsyncOpenAI()

# # Instrument the OpenAI client
# cl.instrument_openai()

# settings = {
#     "model": "gpt-3.5-turbo",
#     "temperature": 0,
#     # ... more settings
# }

# @cl.on_message
# async def on_message(message: cl.Message):
#     response = await client.chat.completions.create(
#         messages=[
#             {
#                 "content": "You are a helpful bot, you always reply in Spanish",
#                 "role": "system"
#             },
#             {
#                 "content": input,
#                 "role": "user"
#             }
#         ],
#         **settings
#     )
#     await cl.Message(content=response.choices[0].message.content).send()