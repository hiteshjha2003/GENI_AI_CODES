import getpass
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from tqdm import tqdm
from pathlib import Path
import time
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import faiss
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

# Ensure the NVIDIA_API_KEY is set correctly
nvapi_key = os.getenv("NVIDIA_API_KEY")
if nvapi_key and nvapi_key.startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
    nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
    assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key

# Initialize the LLM
llm = ChatNVIDIA(model="ai-mixtral-8x7b-instruct", nvidia_api_key=nvapi_key, max_tokens=1024)

result = llm.invoke("Write a ballad about LangChain.")
print(result.content)

# Initialize the embedding
embedder = NVIDIAEmbeddings(model="ai-embed-qa-4")

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
store.save_local('./toy_data/nv_embedding')

# Load the vectorstore back
store = FAISS.load_local("./toy_data/nv_embedding", embedder)

# Wrap the restored vectorstore into a retriever and ask our question
retriever = store.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer solely based on the following context:\n<Documents>\n{context}\n</Documents>",
        ),
        ("user", "{question}"),
    ]
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke("Tell me about Sweden.")
print(result)
