from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

# Use environment variables for the API keys so they are secure when deploying
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

embeddings = download_hugging_face_embeddings()

persist_directory = "chroma_db"

docsearch = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_API_BASE"],
    model="openrouter/free",
    temperature=0.4, 
    max_tokens=500
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

from langchain_core.prompts import PromptTemplate
document_prompt = PromptTemplate(
    input_variables=["page_content", "page"],
    template="Page {page}:\n{page_content}"
)

question_answer_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    answer = response["answer"].replace("\n", "<br>")
    return str(answer)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=False)
