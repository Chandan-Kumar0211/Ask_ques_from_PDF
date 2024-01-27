import os
import getpass
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
# from langchain.llms import GooglePalm
from langchain_community.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain_community.llms.fireworks import Fireworks
from langchain.chains.question_answering import load_qa_chain


load_dotenv()

if "FIREWORKS_API_KEY" not in os.environ:
    os.environ["FIREWORKS_API_KEY"] = getpass.getpass("Fireworks API Key:")


pdf_file_path = '48lawsofpower.pdf'
pdf_reader = PdfReader(pdf_file_path)


def get_response(user_query):
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )

    # creating chunks of texts
    chunks = text_splitter.split_text(text=text)

    # initializing HuggingFaceInstructEmbeddings
    instructor_embeddings = HuggingFaceInstructEmbeddings()

    # creating embeddings of chunks and storing it in vector database (FAISS)
    vector_db = FAISS.from_texts(chunks, instructor_embeddings)


    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    docs = retriever.get_relevant_documents(user_query)

    # llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0)
    # Initialize a Fireworks model
    llm = Fireworks(
        model="accounts/fireworks/models/llama-v2-13b-chat",
        model_kwargs={"temperature": 0.1, "max_tokens": 256},
    )

    prompt = PromptTemplate.from_template(
        "Given the following information: {document}, Can you answer this question: {question}?"
    )

    # chain = load_qa_chain(llm=llm, chain_type='stuff')
    chain = prompt | llm

    response = chain.invoke({"document":docs, "question":user_query})

    return response

