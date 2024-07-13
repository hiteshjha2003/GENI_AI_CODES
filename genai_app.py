import gradio as gr
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import docx
import openai
import faiss
import numpy as np
import glob
import os

# OpenAI API key
openai.api_key = "XXXXXXXXXXXXXXXX"

# Function to extract information from web link
def extract_info_from_link(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract title and meta description as basic information
    title = soup.title.string if soup.title else "No title"
    description = ""
    if soup.find("meta", {"name": "description"}):
        description = soup.find("meta", {"name": "description"}).get("content")
    
    return f"Title: {title}\nDescription: {description}"

# Function to read PDF file
def read_pdf(file):
    text = ""
    with fitz.open(file) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to read Docx file
def read_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to index content using FAISS
def index_content(text):
    sentences = text.split('. ')
    embeddings = [openai.Embedding.create(model="text-embedding-ada-002", input=sentence)['data'][0]['embedding'] for sentence in sentences]
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype(np.float32))
    return index, sentences

# Function to answer questions based on indexed content
def answer_question(question, index, sentences):
    question_embedding = openai.Embedding.create(model="text-embedding-ada-002", input=question)['data'][0]['embedding']
    D, I = index.search(np.array([question_embedding]).astype(np.float32), k=5)
    context = " ".join([sentences[i] for i in I[0]])
    response = openai.Completion.create(
        model="gpt-4",
        prompt=f"Context: {context}\n\nQuestion: {question}\nAnswer:",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function to read and index multiple PDF files from a directory
def index_pdfs_in_directory(directory):
    all_text = ""
    for file_path in glob.glob(os.path.join(directory, "*.pdf")):
        all_text += read_pdf(file_path) + " "
    return index_content(all_text)

# Gradio functions
def process_file(file):
    if file.name.endswith(".pdf"):
        text = read_pdf(file.name)
    elif file.name.endswith(".docx"):
        text = read_docx(file.name)
    else:
        return "Unsupported file format"
    
    index, sentences = index_content(text)
    return text, index, sentences

def answer_from_file(file, question):
    text, index, sentences = process_file(file)
    return answer_question(question, index, sentences)

def process_directory(directory, question):
    index, sentences = index_pdfs_in_directory(directory)
    return answer_question(question, index, sentences)




# Gradio Interfaces
link_interface = gr.Interface(
    fn=extract_info_from_link,
    inputs=gr.Textbox(lines=1, placeholder="Enter URL here", label="Web Link"),
    outputs=gr.Textbox(label="Link Information"),
    title="Extract Information from Web Link"
)

file_interface = gr.Interface(
    fn=process_file,
    inputs=gr.File(file_types=[".pdf", ".docx"], label="Upload File"),
    outputs=gr.Textbox(label="File Content"),
    title="Process File"
)

question_file_interface = gr.Interface(
    fn=answer_from_file,
    inputs=[gr.File(file_types=[".pdf", ".docx"], label="Upload File"), gr.Textbox(lines=1, placeholder="Ask a question about the file content", label="Question")],
    outputs=gr.Textbox(label="Answer"),
    title="Answer Question from File"
)

directory_interface = gr.Interface(
    fn=process_directory,
    inputs=[gr.Textbox(lines=1, placeholder="Enter directory path for PDFs", label="PDF Directory"), gr.Textbox(lines=1, placeholder="Ask a question about the directory content", label="Question")],
    outputs=gr.Textbox(label="Directory Response"),
    title="Process PDF Directory"
)

# Combining the Interfaces into Tabs
iface = gr.TabbedInterface(
    interface_list=[link_interface, file_interface, question_file_interface, directory_interface],
    tab_names=["Web Link", "Process File", "Question from File", "PDF Directory"]
)

iface.launch()
