from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
from openai import OpenAI
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not GITHUB_TOKEN or not PINECONE_API_KEY:
    raise EnvironmentError("Required API keys are not set.")

# Set environment variables (for other packages)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize OpenAI client (GitHub-hosted)
client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=GITHUB_TOKEN,
)

# LangChain embedding + vector store setup
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# System prompt template
system_prompt = (
    "You are a Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "{context}"
)

# Create retrieval chain with LangChain
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def get_context_based_answer(query):
    from langchain_core.documents import Document

    # Retrieve relevant documents
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs if isinstance(doc, Document))

    # Inject context into system prompt
    messages = [
        {"role": "system", "content": system_prompt.format(context=context)},
        {"role": "user", "content": query}
    ]

    # Query GitHub-hosted model
    response = client.chat.completions.create(
        messages=messages,
        temperature=0.7,
        top_p=1.0,
        max_tokens=1000,
        model="openai/gpt-4o"
    )

    return response.choices[0].message.content

# Flask setup
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    try:
        answer = get_context_based_answer(msg)
        print("Assistant:", answer)
        return str(answer)
    except Exception as e:
        print("❌ Error:", e)
        return "Error processing your request."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
