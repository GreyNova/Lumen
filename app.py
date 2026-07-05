import os
import time

import sentence_transformers  # noqa: F401  (kept for parity with original project)
from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

load_dotenv()

app = Flask(__name__)

# ---- OpenRouter configuration -------------------------------------------------
# The OpenRouter key is read from the .env file (OPENAI_API_KEY) and we point the
# OpenAI client at the OpenRouter endpoint, using the free Qwen model.
OPENROUTER_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen/qwen3-next-80b-a3b-instruct:free")

# ---- Embeddings + vector store ------------------------------------------------
embeddings = download_hugging_face_embeddings()

persist_directory = "chroma_db"
docsearch = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
)

# Retrieve a few more chunks so the model has enough context for citations.
retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 4}
)

# ---- LLM ----------------------------------------------------------------------
llm = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    model=MODEL_NAME,
    temperature=0.4,
    max_tokens=600,
    max_retries=6,  # built-in retry on transient upstream errors (e.g. 429)
)

# ---- Prompt chain -------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Inject the source page number into each context chunk so the model can cite it.
document_prompt = PromptTemplate(
    input_variables=["page_content", "page"],
    template="[Source page {page}]\n{page_content}",
)

question_answer_chain = create_stuff_documents_chain(
    llm, prompt, document_prompt=document_prompt
)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


def _answer_question(msg: str) -> str:
    """Invoke the RAG chain with a manual retry/backoff on top of the LLM's
    own retries. Free-tier OpenRouter models are often rate-limited (HTTP 429),
    so we wait and try again instead of showing an error to the user."""
    last_err = None
    for attempt in range(4):
        try:
            response = rag_chain.invoke({"input": msg})
            return response["answer"]
        except Exception as e:  # noqa: BLE001 - surface a friendly message
            last_err = e
            # Backoff: 5s, 10s, 20s
            if attempt < 3:
                time.sleep(5 * (2 ** attempt))
    return (
        "Sorry, I could not reach the model right now (the free tier may be "
        f"rate-limited). Please try again in a minute. Error: {last_err}"
    )


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    answer = _answer_question(msg)
    return answer.replace("\n", "<br>")


if __name__ == "__main__":
    # threaded=True keeps the UI responsive even when a chat request is busy
    # (the free-tier model can be slow / rate-limited).
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
