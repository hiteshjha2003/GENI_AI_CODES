import getpass
import os
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from tqdm import tqdm
from pathlib import Path
import time
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import faiss
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_text_splitters import CharacterTextSplitter



# Load the .env file
load_dotenv(find_dotenv())

# Get the API key from the environment
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    google_api_key = getpass.getpass("Provide your Google API key here")
    os.environ["GOOGLE_API_KEY"] = google_api_key



# Initialize the embedding
embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Read in the text data and prepare them into vectorstore
ps = list(Path("./data/").glob('**/*.txt'))
print(ps)
data = []
sources = []
for p in ps:
    with open(p, encoding="utf-8") as f:
        content = f.read().strip()
        if content:  # Only include non-empty documents
            data.append(content)
            sources.append(p)

documents = [d for d in data if d]
print(len(data), len(documents), data[0] if data else "No data available")

# Speed test: check how fast (in seconds) processing 1 document vs. a batch of 10 documents
if documents:
    # Check document length and content
    first_document = documents[0]
    print("First document length:", len(first_document))
    print("First document content:", first_document[:500])  # Print first 500 characters

    print("Single Document Embedding: ")
    try:
        s = time.perf_counter()
        q_embedding = embedder.embed_documents([first_document])
        elapsed = time.perf_counter() - s
        print("\033[1m" + f"Executed in {elapsed:0.2f} seconds." + "\033[0m")
        print("Shape:", (len(q_embedding),))
    except Exception as e:
        print("Error during single document embedding:", e)

    print("\nBatch Document Embedding: ")
    try:
        s = time.perf_counter()
        d_embeddings = embedder.embed_documents(documents[:10])
        elapsed = time.perf_counter() - s
        print("\033[1m" + f"Executed in {elapsed:0.2f} seconds." + "\033[0m")
        print("Shape:", len(d_embeddings[0]) if d_embeddings else "No embeddings")
    except Exception as e:
        print("Error during batch document embedding:", e)

# Process the documents into FAISS vectorstore and save it to disk
text_splitter = CharacterTextSplitter(chunk_size=400, separator=" ")
docs = []
metadatas = []

for i, d in enumerate(documents):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

store = FAISS.from_texts(docs, embedder, metadatas=metadatas)
store.save_local('./toy_data/google_embedding')

# Load the vectorstore back
store = FAISS.load_local("./toy_data/google_embedding", embedder, allow_dangerous_deserialization=True)

# Wrap the restored vectorstore into a retriever
retriever = store.as_retriever()

# Define the prompt template
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Set up the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# Ask a question using the QA chain
question = "Tell me about the sweden ?"
result = qa_chain({"query": question})

# Print the result
print(result["result"])
