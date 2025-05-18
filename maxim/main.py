import argparse
import openai
import re
import pickle
import os
import numpy as np
import faiss
import fitz
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
import tiktoken
import shutil
import requests
import base64
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

nltk.data.path.append('/app/nltk_data')  # Add this line before any nltk.data.find calls
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading 'punkt_tab' resource for NLTK...")
    nltk.download('punkt_tab', download_dir='/app/nltk_data', quiet=True)
    print("'punkt_tab' downloaded successfully.")

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

VISION_API_KEY = os.getenv("VISION_API_KEY")
if not VISION_API_KEY:
    raise ValueError("VISION_API_KEY not found in environment variables.")

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tokenizer = tiktoken.get_encoding("cl100k_base")
conversation_history = []

SYSTEM_PROMPT = """
You are a Google Ads expert assistant. Answer user questions based on the provided knowledge base excerpts and image analysis (labels and extracted text).
Include direct, verbatim quotes from the knowledge base when available.
If appropriate, clearly provide practical suggestions or recommendations in bullet points.
If a question does not explicitly mention Google Ads but reasonably relates to Google Ads campaigns, assume the user intends a Google Ads-related answer and respond accordingly using the provided knowledge base.
If the knowledge base does not contain information relevant to the image analysis or query, explicitly respond:
"I don't have enough information in my knowledge base to answer that."
"""

PDF_STORAGE_DIR = os.path.join(os.path.dirname(__file__), "pdf_knowledge_base")
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from PDF: {pdf_path}")
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None
    return text

def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())

def chunk_text(text, max_words=500, overlap_words=100):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    word_count = 0
    for sentence in sentences:
        words = sentence.split()
        word_count += len(words)
        if word_count <= max_words:
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_words:] + [sentence]
            word_count = sum(len(s.split()) for s in current_chunk)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def truncate_text(text, max_tokens):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        return tokenizer.decode(tokens[:max_tokens])
    return text

def get_embeddings(texts, batch_size=10):
    embeddings = []
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start:batch_start + batch_size]
        response = client.embeddings.create(model="text-embedding-ada-002", input=batch)
        embeddings.extend([item.embedding for item in response.data])
    return np.array(embeddings, dtype=np.float32)

def build_faiss_index(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    index = faiss.IndexFlatL2(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)
    return index

def preprocess_and_index(raw_text, output_dir="models/embeddings"):
    cleaned_text = clean_text(raw_text)
    new_chunks = chunk_text(cleaned_text)
    if not new_chunks:
        return [], None, None
    embeddings = get_embeddings(new_chunks)
    if embeddings.size == 0:
        return [], None, None
    index = build_faiss_index(embeddings)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(new_chunks, f)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    return new_chunks, embeddings, index

def load_knowledge_base(output_dir="models/embeddings"):
    try:
        with open(os.path.join(output_dir, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)
        embeddings = np.load(os.path.join(output_dir, "embeddings.npy"))
        index = faiss.read_index(os.path.join(output_dir, "faiss_index.bin"))
        return chunks, embeddings, index
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return None, None, None

def upload_pdf(new_pdf_path, pdf_storage_dir=PDF_STORAGE_DIR, output_dir="models/embeddings"):
    print(f"Processing PDF upload: {new_pdf_path}")
    if not os.path.exists(new_pdf_path):
        print(f"File {new_pdf_path} does not exist")
        return "Error: PDF file does not exist."
    destination = os.path.join(pdf_storage_dir, os.path.basename(new_pdf_path))
    shutil.copy(new_pdf_path, destination)
    raw_text = extract_text_from_pdf(new_pdf_path)
    if raw_text is None:
        return "Error processing PDF."
    new_chunks = chunk_text(clean_text(raw_text))
    if not new_chunks:
        return "No content extracted from PDF."
    new_embeddings = get_embeddings(new_chunks)
    if new_embeddings.size == 0:
        return "Error generating embeddings for PDF."
    existing_chunks, existing_embeddings, index = load_knowledge_base(output_dir)
    if existing_chunks is None or index is None:
        existing_chunks = new_chunks
        existing_embeddings = new_embeddings
        index = build_faiss_index(new_embeddings)
    else:
        existing_chunks.extend(new_chunks)
        combined_embeddings = np.concatenate((existing_embeddings, new_embeddings), axis=0)
        index = build_faiss_index(combined_embeddings)
        existing_embeddings = combined_embeddings
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(existing_chunks, f)
    np.save(os.path.join(output_dir, "embeddings.npy"), existing_embeddings)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    return "PDF uploaded and indexed successfully."

def extract_image_details(image_path):
    print(f"Processing image: {image_path}")
    if not os.path.exists(image_path):
        print(f"Image file {image_path} does not exist")
        return None
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        url = f"https://vision.googleapis.com/v1/images:annotate?key={VISION_API_KEY}"
        payload = {
            "requests": [{
                "image": {"content": encoded_image},
                "features": [
                    {"type": "LABEL_DETECTION", "maxResults": 5},
                    {"type": "TEXT_DETECTION", "maxResults": 1}
                ]
            }]
        }
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code != 200:
            print(f"Vision API error: {response.text}")
            return None
        result = response.json()
        response_data = result.get('responses', [{}])[0]
        labels = [label.get('description', '') for label in response_data.get('labelAnnotations', [])]
        text = response_data.get('textAnnotations', [{}])[0].get('description', '')
        combined_info = f"Labels: {', '.join(labels)}. " if labels else ""
        combined_info += f"Extracted Text: {text}" if text else ""
        return combined_info.strip() or None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def query_faiss(query, chunks, index, k=5):
    query_embedding = get_embeddings([query])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    distances, indices = index.search(np.array([query_embedding]), k)
    if distances[0][0] > 0.7:  # Threshold for relevance
        return []
    return [chunks[idx] for idx in indices[0]]

def generate_response(query, retrieved_chunks):
    global conversation_history
    conversation_history.append({"role": "user", "content": query})
    if not retrieved_chunks:
        response = "I don't have enough information in my knowledge base to answer that."
        conversation_history.append({"role": "assistant", "content": response})
        return response
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history[-4:])
    kb_context = "\n---\n".join([truncate_text(chunk, 200) for chunk in retrieved_chunks])
    kb_message = (
        f"Query: {query}\n\n"
        f"Knowledge Base Excerpts:\n{kb_context}\n\n"
        "Provide a detailed Google Ads-related response based on the query and knowledge base excerpts."
    )
    messages.append({"role": "assistant", "content": kb_message})
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=600,
        temperature=0.6
    )
    assistant_response = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": assistant_response})
    return assistant_response

def process_user_input(text_input, image_input, chunks, index):
    combined_query = ""
    if image_input and os.path.exists(image_input):
        image_info = extract_image_details(image_input)
        if image_info is None:
            return "Error processing the image. Please try again with a different image."
        combined_query += image_info
    if text_input:
        combined_query += " " + text_input
    combined_query = combined_query.strip()
    if not combined_query:
        return "I'm sorry, but without an image to analyze or text input, I can't provide an explanation. Please provide an image or text for me to assist you further."
    if "explain the image" in text_input.lower() and not image_input:
        return "I'm sorry, but I can't provide an explanation of the image as no image was provided in your request. Please provide an image for me to analyze and explain."
    retrieved_chunks = query_faiss(combined_query, chunks, index)
    return generate_response(combined_query, retrieved_chunks)

def main():
    parser = argparse.ArgumentParser(description="Google Ads Chatbot")
    parser.add_argument("--upload", type=str, help="Path to PDF to upload")
    parser.add_argument("--image", type=str, help="Path to image to analyze")
    parser.add_argument("query", type=str, nargs='?', default="", help="Text query")
    args = parser.parse_args()

    model_dir = os.path.join(os.path.dirname(__file__), "models", "embeddings")
    if args.upload:
        response = upload_pdf(args.upload, pdf_storage_dir=PDF_STORAGE_DIR, output_dir=model_dir)
        print(f"AI Response: {response}")
        return

    chunks, embeddings, index = load_knowledge_base(model_dir)
    if chunks is None or index is None:
        print("Knowledge base is empty")
        response = "Knowledge base is empty. Please upload a PDF to initialize it."
        print(f"AI Response: {response}")
        return

    response = process_user_input(args.query, args.image, chunks, index)
    print(f"AI Response: {response}")

if __name__ == "__main__":
    main()