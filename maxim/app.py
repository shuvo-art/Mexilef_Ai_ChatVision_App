# import re
# import pandas as pd
# import pickle
# import os
# import logging
# import faiss
# import numpy as np
# import openai
# from google.cloud import vision
# import io
# from spellchecker import SpellChecker
# from dotenv import load_dotenv
# from PIL import Image, ImageEnhance

# # Load environment variables
# load_dotenv()

# # Set your OpenAI API key (via environment variable)
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Setup basic logging
# logging.basicConfig(level=logging.INFO)

# # Initialize spell checker
# spell = SpellChecker()

# # --- Memory for Conversation History ---
# conversation_history = []

# def add_to_history(role, message):
#     """Adds a message to the conversation history."""
#     conversation_history.append({"role": role, "content": message})

# def get_history():
#     """Returns the conversation history as a formatted string."""
#     return "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])

# # --- OpenAI Embedding Functions ---
# def get_openai_embedding(text):
#     """Generates embeddings using OpenAI's text-embedding-ada-002 model."""
#     try:
#         response = openai.Embedding.create(
#             input=[text],
#             model="text-embedding-ada-002"
#         )
#         return response['data'][0]['embedding']
#     except Exception as e:
#         logging.error(f"Error generating OpenAI embedding: {str(e)}")
#         return None

# # --- Indexing Functions ---
# def index_document(chunks, output_dir="models/embeddings"):
#     """Indexes the document chunks using FAISS and OpenAI embeddings."""
#     embeddings = []
#     for chunk in chunks:
#         embedding = get_openai_embedding(chunk)
#         if embedding:
#             embeddings.append(embedding)
    
#     embeddings = np.array(embeddings).astype("float32")
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dimension)
#     index.add(embeddings)
    
#     faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
#     return index

# # --- Query Handling Functions ---
# def truncate_context(context, max_tokens=4000):
#     """Truncates the context to fit within the token limit."""
#     tokens = context.split()
#     if len(tokens) > max_tokens:
#         return " ".join(tokens[:max_tokens])
#     return context

# def load_models_and_index(model_dir="models/embeddings"):
#     """Loads the FAISS index and chunks."""
#     index = faiss.read_index(os.path.join(model_dir, "faiss_index.bin"))
#     with open(os.path.join(model_dir, "chunks.pkl"), "rb") as f:
#         chunks = pickle.load(f)
#     return index, chunks

# def correct_spelling(query):
#     """Corrects spelling errors in the query."""
#     return " ".join([spell.correction(word) for word in query.split()])

# def process_query(query, index, chunks, k=3):
#     """Processes a user query and returns an answer using OpenAI embeddings."""
#     try:
#         query = correct_spelling(query)
#         query_embedding = np.array([get_openai_embedding(query)]).astype("float32")
#         distances, indices = index.search(query_embedding, k)
#         retrieved_chunks = [chunks[i] for i in indices[0]]
        
#         if distances[0][0] < 0.5:  # Low similarity score
#             logging.warning(f"Low confidence in query: {query}. Distances: {distances}")
#             return f"I'm not very confident in my answer. Would you like to refine your query? I found the following related sections:\n\n" + "\n".join(retrieved_chunks)
        
#         context = "\n".join(retrieved_chunks)
#         context = truncate_context(context)
        
#         # Add conversation history to the context
#         history_context = get_history()
#         full_context = f"Conversation History:\n{history_context}\n\nRelevant Context:\n{context}"
        
#         # Use GPT-4 to answer the question
#         prompt = f"Answer this Google Ads question using only the provided context. Do not use external knowledge:\nQuery: {query}\nContext: {full_context}"
#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[{"role": "system", "content": "You are a Google Ads expert. Answer queries based solely on the provided context."},
#                       {"role": "user", "content": prompt}],
#             max_tokens=150,
#             temperature=0.5
#         )
#         answer = response.choices[0].message["content"].strip()
        
#         # Add the query and answer to the conversation history
#         add_to_history("user", query)
#         add_to_history("assistant", answer)
        
#         return answer
    
#     except openai.error.OpenAIError as e:
#         logging.error(f"OpenAI API error: {str(e)}")
#         return "An error occurred while processing your query. Please try again later."
#     except Exception as e:
#         logging.error(f"Unexpected error processing query: {query}. Error: {str(e)}")
#         return "An unexpected error occurred. Please try again later."

# # Image Analysis Functions
# def validate_file(file_path, file_type="image"):
#     """Validates the file path and type."""
#     if not os.path.exists(file_path):
#         return f"Error: File '{file_path}' not found."
    
#     if file_type == "image" and not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
#         return "Error: Please upload a valid image file (PNG, JPG, JPEG, GIF)."
    
#     return None

# def preprocess_image(image_path):
#     """Preprocesses the image for better OCR accuracy."""
#     try:
#         image = Image.open(image_path)
#         # Enhance contrast
#         enhancer = ImageEnhance.Contrast(image)
#         image = enhancer.enhance(2.0)
#         # Resize for better OCR
#         image = image.resize((image.width * 2, image.height * 2))
#         return image
#     except Exception as e:
#         logging.error(f"Error preprocessing image: {str(e)}")
#         return None

# def extract_text_from_image(image_path):
#     """Extracts text from an image using Google Cloud Vision."""
#     try:
#         client = vision.ImageAnnotatorClient()
#         with io.open(image_path, "rb") as image_file:
#             content = image_file.read()
#         image = vision.Image(content=content)
#         response = client.text_detection(image=image)
#         if response.error.message:
#             raise Exception(f"Error in OCR processing: {response.error.message}")
#         return response.text_annotations[0].description if response.text_annotations else ""
#     except Exception as e:
#         logging.error(f"Error processing image: {image_path}. Error: {str(e)}")
#         return ""

# def parse_metrics(text):
#     """Parses metrics from the extracted text."""
#     metrics = {}
#     patterns = {
#         "CTR": r"CTR:?\s*(\d+\.?\d*)%",
#         "CPC": r"CPC:?\s*\$?(\d+\.?\d*)",
#         "Impressions": r"Impressions:?\s*(\d+)",
#         "Bid Strategy": r"Bid Strategy:?\s*([^\n]+)"
#     }
#     for metric, pattern in patterns.items():
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             metrics[metric] = match.group(1)
#     return metrics

# def formulate_query(metrics):
#     """Formulates a query based on the parsed metrics."""
#     if not metrics:
#         return "No metrics found in the screenshot. Please upload a valid Google Ads campaign screenshot."
#     query_parts = [f"{key}: {value}" for key, value in metrics.items()]
#     query = f"Campaign with {', '.join(query_parts)}. Provide optimization tips."
#     return query

# def process_image(image_path, index, chunks):
#     """Processes an image and returns an answer."""
#     extracted_text = extract_text_from_image(image_path)
#     metrics = parse_metrics(extracted_text)
#     query = formulate_query(metrics)
#     answer = process_query(query, index, chunks)
#     return answer

# # --- Knowledge Base Update Functions ---
# def update_knowledge_base(file_path, index, chunks, output_dir="models/embeddings"):
#     """Updates the knowledge base with new document chunks."""
#     new_chunks = preprocess_document(file_path, output_dir)
#     new_embeddings = np.array([get_openai_embedding(chunk) for chunk in new_chunks]).astype("float32")
#     index.add(new_embeddings)
    
#     chunks.extend(new_chunks)
    
#     faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
#     with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
#         pickle.dump(chunks, f)
#     return chunks

# # --- User Feedback Functions ---
# def collect_feedback(query, answer):
#     """Collects user feedback on the AI's response."""
#     print(f"Query: {query}")
#     print(f"Answer: {answer}")
#     feedback = input("Was this answer helpful? (yes/no): ").strip().lower()
#     if feedback == "no":
#         reason = input("What was the issue? (e.g., incorrect, irrelevant, incomplete): ").strip()
#         logging.info(f"User feedback for query '{query}': {reason}")

# # --- Main CLI Interface ---
# def main():
#     document_path = "data/google_ads_document.txt"
#     model_dir = "models/embeddings"
    
#     if not os.path.exists(document_path):
#         print("Error: Please place your Google Ads document at 'data/google_ads_document.txt'.")
#         return
    
#     if not os.path.exists(os.path.join(model_dir, "faiss_index.bin")) or not os.path.exists(os.path.join(model_dir, "chunks.pkl")):
#         print("Preprocessing and indexing document...")
#         chunks = preprocess_document(document_path)
#         index = index_document(chunks)
#     else:
#         index, chunks = load_models_and_index()
    
#     while True:
#         print("\nOptions:")
#         print("1. Ask a Google Ads question")
#         print("2. Analyze a screenshot")
#         print("3. Update knowledge base")
#         print("4. Exit")
#         choice = input("Enter your choice (1-4): ")
        
#         if choice == "1":
#             query = input("Enter your Google Ads question: ")
#             answer = process_query(query, index, chunks)
#             print(f"Answer: {answer}")
#             collect_feedback(query, answer)
        
#         elif choice == "2":
#             image_path = input("Enter the path to your screenshot: ")
#             validation_error = validate_file(image_path, file_type="image")
#             if validation_error:
#                 print(validation_error)
#                 continue
#             answer = process_image(image_path, index, chunks)
#             print(f"Answer: {answer}")
#             collect_feedback("Image Analysis", answer)
        
#         elif choice == "3":
#             new_doc_path = input("Enter the path to the new document: ")
#             if not os.path.exists(new_doc_path):
#                 print("Error: Document file not found.")
#                 continue
#             chunks = update_knowledge_base(new_doc_path, index, chunks)
#             print("Knowledge base updated successfully!")
        
#         elif choice == "4":
#             print("Exiting chatbot.")
#             break
        
#         else:
#             print("Invalid choice. Please try again.")

# if __name__ == "__main__":
#     main()






# import re
# import pickle
# import os
# import io
# import numpy as np
# import faiss
# import nltk
# from nltk.tokenize import sent_tokenize
# from openai import OpenAI
# from google.cloud import vision
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Initialize OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Initialize Google Cloud Vision client
# vision_client = vision.ImageAnnotatorClient()

# # System prompt for GPT-4
# SYSTEM_PROMPT = """
# You are an expert Google Ads assistant. Your role is to provide accurate, actionable, and concise text-based answers 
# to Google Ads-related questions. Use the provided document chunks or image data as your primary context to generate responses. 
# Do not generate images or answer unrelated questions. If a query is off-topic, respond with: "I can only assist with Google Ads questions." 
# Focus on optimization tips and maintain a professional, helpful tone.
# """

# # Global variable to store conversation history
# conversation_history = []

# # --- Text Preprocessing Functions ---
# def clean_text(text):
#     """Cleans the input text by removing extra spaces and empty lines."""
#     text = re.sub(r"\s+", " ", text.strip())
#     return text

# def chunk_text(text, max_words=1000, overlap_words=200):
#     """Splits text into larger chunks with overlap for better context retention."""
#     sentences = sent_tokenize(text)
#     chunks, current_chunk, word_count = [], [], 0
    
#     for sentence in sentences:
#         words = sentence.split()
#         if word_count + len(words) <= max_words:
#             current_chunk.append(sentence)
#             word_count += len(words)
#         else:
#             chunks.append(" ".join(current_chunk))
#             overlap_text = " ".join(current_chunk[-overlap_words:]) if len(current_chunk) > overlap_words else " ".join(current_chunk)
#             current_chunk = [overlap_text, sentence]
#             word_count = len(overlap_text.split()) + len(words)
    
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
    
#     return [chunk.strip() for chunk in chunks if chunk.strip()]

# def split_large_chunk(chunk, max_tokens=8192):
#     """Splits a large chunk into smaller sub-chunks that fit within the token limit."""
#     words = chunk.split()
#     sub_chunks = []
#     current_sub_chunk = []
#     current_token_count = 0
    
#     for word in words:
#         # Estimate token count (1 word ≈ 1.3 tokens on average)
#         word_tokens = int(len(word) * 1.3)
#         if current_token_count + word_tokens <= max_tokens:
#             current_sub_chunk.append(word)
#             current_token_count += word_tokens
#         else:
#             sub_chunks.append(" ".join(current_sub_chunk))
#             current_sub_chunk = [word]
#             current_token_count = word_tokens
    
#     if current_sub_chunk:
#         sub_chunks.append(" ".join(current_sub_chunk))
    
#     return sub_chunks

# # --- Embedding Functions ---
# def get_embeddings(texts):
#     """Generates embeddings using OpenAI's text-embedding-ada-002 model."""
#     response = client.embeddings.create(model="text-embedding-ada-002", input=texts)
#     return np.array([embedding.embedding for embedding in response.data], dtype=np.float32)

# # --- FAISS Indexing Functions ---
# def build_faiss_index(embeddings):
#     """Builds a FAISS index for fast similarity search."""
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)
#     return index

# def preprocess_and_index(raw_text, output_dir="models/embeddings"):
#     """Preprocesses the document and builds the FAISS index."""
#     cleaned_text = clean_text(raw_text)
#     chunks = chunk_text(cleaned_text, max_words=1000, overlap_words=200)
    
#     # Split large chunks to fit within token limits
#     final_chunks = []
#     for chunk in chunks:
#         if len(chunk.split()) * 1.3 > 8192:  # Check if chunk exceeds token limit
#             sub_chunks = split_large_chunk(chunk)
#             final_chunks.extend(sub_chunks)
#         else:
#             final_chunks.append(chunk)
    
#     embeddings = get_embeddings(final_chunks)
#     index = build_faiss_index(embeddings)
    
#     os.makedirs(output_dir, exist_ok=True)
#     with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
#         pickle.dump(final_chunks, f)
#     np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
#     faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    
#     return final_chunks, embeddings, index

# def load_knowledge_base(output_dir="models/embeddings"):
#     """Loads the existing knowledge base (chunks, embeddings, and FAISS index)."""
#     if not os.path.exists(output_dir):
#         return None, None, None
    
#     try:
#         with open(os.path.join(output_dir, "chunks.pkl"), "rb") as f:
#             chunks = pickle.load(f)
#         embeddings = np.load(os.path.join(output_dir, "embeddings.npy"))
#         index = faiss.read_index(os.path.join(output_dir, "faiss_index.bin"))
#         return chunks, embeddings, index
#     except Exception as e:
#         print(f"Error loading knowledge base: {e}")
#         return None, None, None

# def update_knowledge_base(new_pdf_path, output_dir="models/embeddings"):
#     """Updates the knowledge base with a new PDF."""
#     # Load existing knowledge base
#     chunks, embeddings, index = load_knowledge_base(output_dir)
    
#     # If no existing knowledge base, initialize with empty data
#     if chunks is None or embeddings is None or index is None:
#         chunks, embeddings, index = [], np.array([]), None
    
#     # Preprocess and index the new PDF
#     with open(new_pdf_path, "r", encoding="utf-8") as f:
#         raw_text = f.read()
#     new_chunks = chunk_text(clean_text(raw_text), max_words=1000, overlap_words=200)
    
#     # Split large chunks to fit within token limits
#     final_new_chunks = []
#     for chunk in new_chunks:
#         if len(chunk.split()) * 1.3 > 8192:  # Check if chunk exceeds token limit
#             sub_chunks = split_large_chunk(chunk)
#             final_new_chunks.extend(sub_chunks)
#         else:
#             final_new_chunks.append(chunk)
    
#     new_embeddings = get_embeddings(final_new_chunks)
    
#     # Combine the new chunks and embeddings with the existing ones
#     combined_chunks = chunks + final_new_chunks
#     combined_embeddings = np.vstack([embeddings, new_embeddings]) if embeddings.size > 0 else new_embeddings
    
#     # Rebuild the FAISS index with the combined embeddings
#     dimension = combined_embeddings.shape[1]
#     combined_index = faiss.IndexFlatL2(dimension)
#     combined_index.add(combined_embeddings)
    
#     # Save the updated index
#     os.makedirs(output_dir, exist_ok=True)
#     with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
#         pickle.dump(combined_chunks, f)
#     np.save(os.path.join(output_dir, "embeddings.npy"), combined_embeddings)
#     faiss.write_index(combined_index, os.path.join(output_dir, "faiss_index.bin"))
    
#     return combined_chunks, combined_embeddings, combined_index

# # --- Image Analysis Functions ---
# def analyze_image_to_query(image_path):
#     """Extracts text from an image using Google Cloud Vision."""
#     with io.open(image_path, "rb") as image_file:
#         content = image_file.read()
    
#     image = vision.Image(content=content)
#     response = vision_client.text_detection(image=image)
#     if response.error.message:
#         raise Exception(f"Vision API error: {response.error.message}")
    
#     text = response.text_annotations[0].description if response.text_annotations else ""
#     return f"Extracted text: {text}" if text else "No text detected in image."

# # --- Query Handling Functions ---
# def query_faiss(query, chunks, index, k=3):
#     """Queries the FAISS index for relevant chunks."""
#     query_embedding = get_embeddings([query])[0]
#     distances, indices = index.search(np.array([query_embedding]), k)
#     retrieved_chunks = [chunks[idx] for idx in indices[0]]
#     return retrieved_chunks

# def generate_response(query, retrieved_chunks):
#     """Generates a response using GPT-4, including conversation history."""
#     global conversation_history
    
#     # Add the user's query to the conversation history
#     conversation_history.append({"role": "user", "content": query})
    
#     # Prepare the context for GPT-4
#     context = f"Conversation History:\n"
#     for message in conversation_history:
#         context += f"{message['role']}: {message['content']}\n"
#     context += f"\nRetrieved Context:\n" + "\n".join(retrieved_chunks)
    
#     # Generate the response using GPT-4
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             *conversation_history,  # Include conversation history
#             {"role": "user", "content": context}
#         ],
#         max_tokens=300,
#         temperature=0.7
#     )
    
#     # Add the chatbot's response to the conversation history
#     chatbot_response = response.choices[0].message.content.strip()
#     conversation_history.append({"role": "assistant", "content": chatbot_response})
    
#     return chatbot_response

# # --- Input Detection and Processing ---
# def detect_input_type(user_input):
#     """Detects whether the input is text, an image path, or both."""
#     if os.path.exists(user_input) and user_input.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
#         return "image"
#     elif " " in user_input:  # Simple heuristic to detect text
#         return "text"
#     else:
#         return "unknown"

# def process_user_input(user_input, chunks, index):
#     """Processes user input (text, image, or both) and generates a response."""
#     input_type = detect_input_type(user_input)
    
#     if input_type == "text":
#         query = user_input
#         retrieved_chunks = query_faiss(query, chunks, index)
#         return generate_response(query, retrieved_chunks)
#     elif input_type == "image":
#         query = analyze_image_to_query(user_input)
#         retrieved_chunks = query_faiss(query, chunks, index)
#         return generate_response(query, retrieved_chunks)
#     else:
#         return "Unsupported input type. Please provide text or an image file path."

# # --- Main Loop ---
# def main():
#     # Initialize with one PDF
#     initial_pdf_path = "data/google_ads_document.txt"
#     model_dir = "models/embeddings"
    
#     if not os.path.exists(initial_pdf_path):
#         print(f"Error: Initial PDF file '{initial_pdf_path}' not found.")
#         return
    
#     # Preprocess and index the initial PDF
#     chunks, embeddings, index = preprocess_and_index(open(initial_pdf_path, "r", encoding="utf-8").read(), model_dir)
    
#     print("Chatbot is ready! You can provide text, an image path, or upload a new PDF.")
#     while True:
#         user_input = input("\nEnter your input (text, image path, or 'upload' to add a new PDF): ").strip()
        
#         if user_input.lower() in ["exit", "quit"]:
#             print("Exiting chatbot.")
#             break
#         elif user_input.lower() == "upload":
#             new_pdf_path = input("Enter the path to the new PDF: ").strip()
#             if not os.path.exists(new_pdf_path):
#                 print("Error: File not found.")
#                 continue
#             chunks, embeddings, index = update_knowledge_base(new_pdf_path, model_dir)
#             print("New PDF added to the knowledge base!")
#         else:
#             response = process_user_input(user_input, chunks, index)
#             print(f"\nResponse: {response}\n")

# if __name__ == "__main__":
#     main()

#####choltase valoi.....
# import openai
# import re
# import pickle
# import os
# import numpy as np
# import faiss
# import fitz  # PyMuPDF
# import nltk
# from nltk.tokenize import sent_tokenize
# from dotenv import load_dotenv
# import tiktoken
# import time

# nltk.download('punkt')
# load_dotenv()

# # Initialize OpenAI client
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# if not os.getenv("OPENAI_API_KEY"):
#     raise ValueError("OPENAI_API_KEY not found in environment variables.")

# tokenizer = tiktoken.get_encoding("cl100k_base")
# conversation_history = []

# SYSTEM_PROMPT = """
# You are an expert Google Ads assistant. Your role is to provide accurate, actionable, and concise text-based answers 
# to Google Ads-related questions. Use the provided document chunks as your primary context to generate responses. 
# Do not generate images or answer unrelated questions. If a query is off-topic, respond with: "I can only assist with Google Ads questions." 
# Focus on optimization tips and maintain a professional, helpful tone.
# """

# #  Text Preprocessing 
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     try:
#         with fitz.open(pdf_path) as doc:
#             for page in doc:
#                 text += page.get_text("text") + "\n"
#     except Exception as e:
#         print(f"Error extracting text from {pdf_path}: {e}")
#         return None
#     return text

# def clean_text(text):
#     text = re.sub(r"\s+", " ", text.strip())
#     return text

# def chunk_text(text, max_words=500, overlap_words=100):
#     sentences = sent_tokenize(text)
#     chunks, current_chunk, word_count = [], [], 0
    
#     for sentence in sentences:
#         words = sentence.split()
#         word_count += len(words)
#         if word_count <= max_words:
#             current_chunk.append(sentence)
#         else:
#             chunks.append(" ".join(current_chunk))
#             # Calculate overlap in terms of words, not sentences
#             total_words = sum(len(s.split()) for s in current_chunk)
#             overlap_idx = len(current_chunk) - 1
#             overlap_word_count = 0
#             while overlap_idx >= 0 and overlap_word_count < overlap_words:
#                 overlap_word_count += len(current_chunk[overlap_idx].split())
#                 overlap_idx -= 1
#             current_chunk = current_chunk[overlap_idx + 1:] + [sentence]
#             word_count = sum(len(s.split()) for s in current_chunk)
    
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
    
#     return [chunk.strip() for chunk in chunks if chunk.strip()]

# # --- Embedding Functions ---
# def count_tokens(text):
#     return len(tokenizer.encode(text))

# def get_embeddings(texts, max_tokens=8192, max_retries=5, initial_delay=1):
#     embeddings = []
#     for i, text in enumerate(texts):
#         if not text or not isinstance(text, str):
#             print(f"Skipping invalid chunk at index {i}: {text}")
#             continue
        
#         tokens = count_tokens(text)
#         if tokens > max_tokens:
#             print(f"Chunk at index {i} exceeds token limit ({tokens} > {max_tokens}): {text[:100]}...")
#             continue
        
#         retries = 0
#         while retries < max_retries:
#             try:
#                 response = client.embeddings.create(model="text-embedding-ada-002", input=text)
#                 embedding = response.data[0].embedding
#                 embeddings.append(embedding)
#                 break
#             except openai.RateLimitError as e:
#                 retries += 1
#                 delay = initial_delay * (2 ** (retries - 1))  # Exponential backoff
#                 print(f"Rate limit hit for chunk {i}. Retrying in {delay} seconds... ({retries}/{max_retries})")
#                 time.sleep(delay)
#             except Exception as e:
#                 print(f"Error embedding chunk at index {i}: {e}")
#                 break
#         else:
#             print(f"Failed to embed chunk {i} after {max_retries} retries.")
    
#     if not embeddings:
#         raise ValueError("No valid embeddings generated.")
    
#     return np.array(embeddings, dtype=np.float32)

# # --- FAISS Indexing ---
# def build_faiss_index(embeddings):
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)
#     return index

# def preprocess_and_index(raw_text, output_dir="models/embeddings"):
#     if not raw_text:
#         raise ValueError("Input text is empty or None.")
    
#     cleaned_text = clean_text(raw_text)
#     print(f"Cleaned text length: {len(cleaned_text)} characters")
    
#     chunks = chunk_text(cleaned_text, max_words=500, overlap_words=100)
#     print(f"Number of chunks: {len(chunks)}")
    
#     chunks = [chunk for chunk in chunks if chunk and isinstance(chunk, str)]
#     print(f"Number of valid chunks after filtering: {len(chunks)}")
#     if not chunks:
#         raise ValueError("No valid chunks found after preprocessing.")
    
#     print("Sample chunks:")
#     for i, chunk in enumerate(chunks[:3]):
#         print(f"Chunk {i+1}: {chunk[:100]}...")
    
#     embeddings = get_embeddings(chunks)
#     print(f"Embeddings shape: {embeddings.shape}")
    
#     index = build_faiss_index(embeddings)
#     print("FAISS index built successfully.")
    
#     os.makedirs(output_dir, exist_ok=True)
#     with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
#         pickle.dump(chunks, f)
#     np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
#     faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
#     print("Chunks, embeddings, and index saved successfully.")
    
#     return chunks, embeddings, index

# def load_knowledge_base(output_dir="models/embeddings"):
#     if not os.path.exists(output_dir):
#         return None, None, None
#     try:
#         with open(os.path.join(output_dir, "chunks.pkl"), "rb") as f:
#             chunks = pickle.load(f)
#         embeddings = np.load(os.path.join(output_dir, "embeddings.npy"))
#         index = faiss.read_index(os.path.join(output_dir, "faiss_index.bin"))
#         return chunks, embeddings, index
#     except Exception as e:
#         print(f"Error loading knowledge base: {e}")
#         return None, None, None

# # --- Query Handling ---
# def query_faiss(query, chunks, index, k=3):
#     query_embedding = get_embeddings([query])[0]
#     distances, indices = index.search(np.array([query_embedding]), k)
#     return [chunks[idx] for idx in indices[0]]

# def generate_response(query, retrieved_chunks):
#     global conversation_history
#     conversation_history.append({"role": "user", "content": query})
    
#     context = "Conversation History:\n" + "\n".join(f"{m['role']}: {m['content']}" for m in conversation_history)
#     context += f"\nRetrieved Context:\n" + "\n".join(retrieved_chunks)
    
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": context}
#         ],
#         max_tokens=300,
#         temperature=0.7
#     )
    
#     chatbot_response = response.choices[0].message.content.strip()
#     conversation_history.append({"role": "assistant", "content": chatbot_response})
#     return chatbot_response

# def process_user_input(user_input, chunks, index):
#     retrieved_chunks = query_faiss(user_input, chunks, index)
#     print("\nRetrieved Chunks:")
#     for i, chunk in enumerate(retrieved_chunks):
#         print(f"Chunk {i + 1}: {chunk[:100]}...")
#     return generate_response(user_input, retrieved_chunks)

# # --- Main Loop ---
# def main():
#     initial_pdf_path = "D:\\Downloads\\maxim\\Google ads knowledge base master.pdf"
#     model_dir = "models/embeddings"

#     if not os.path.exists(initial_pdf_path):
#         print(f"Error: Initial PDF file '{initial_pdf_path}' not found.")
#         return

#     raw_text = extract_text_from_pdf(initial_pdf_path)
#     if not raw_text:
#         print("Error: Could not extract text from the PDF.")
#         return

#     chunks, embeddings, index = preprocess_and_index(raw_text, model_dir)
#     if chunks is None or embeddings is None or index is None:
#         print("Error: Failed to preprocess and index the document.")
#         return

#     print("Chatbot is ready! You can provide text or upload a new PDF.")
#     while True:
#         user_input = input("\nEnter your input (text or 'upload' to add a new PDF): ").strip()
#         if user_input.lower() in ["exit", "quit"]:
#             print("Exiting chatbot.")
#             break
#         elif user_input.lower() == "upload":
#             new_pdf_path = input("Enter the path to the new PDF: ").strip()
#             if not os.path.exists(new_pdf_path):
#                 print("Error: File not found.")
#                 continue
#             new_text = extract_text_from_pdf(new_pdf_path)
#             if not new_text:
#                 print("Error: Could not extract text from the new PDF.")
#                 continue
#             chunks, embeddings, index = preprocess_and_index(new_text, model_dir)
#             if chunks is None or embeddings is None or index is None:
#                 print("Error: Failed to preprocess and index the new PDF.")
#                 continue
#             print("New PDF added to the knowledge base!")
#         else:
#             response = process_user_input(user_input, chunks, index)
#             print(f"\nResponse: {response}")

# if __name__ == "__main__":
#     main()



##### so far so good

# import openai
# import re
# import pickle
# import os
# import numpy as np
# import faiss
# import fitz  # PyMuPDF
# import nltk
# from nltk.tokenize import sent_tokenize
# from dotenv import load_dotenv
# import tiktoken
# import time
# import shutil

# nltk.download('punkt')
# load_dotenv()

# # Initialize OpenAI client
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# if not os.getenv("OPENAI_API_KEY"):
#     raise ValueError("OPENAI_API_KEY not found in environment variables.")

# tokenizer = tiktoken.get_encoding("cl100k_base")
# conversation_history = []

# SYSTEM_PROMPT = """
# You are an expert Google Ads assistant. Your role is to provide accurate, actionable, and concise text-based answers 
# to Google Ads-related questions. Use the provided document chunks as your primary context to generate responses. 
# Do not generate images or answer unrelated questions. If a query is off-topic, respond with: "I can only assist with Google Ads questions." 
# Focus on optimization tips and maintain a professional, helpful tone.
# """

# PDF_STORAGE_DIR = "data/pdf_knowledge_base"
# os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

# # --- Text Preprocessing ---
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     try:
#         with fitz.open(pdf_path) as doc:
#             for page in doc:
#                 text += page.get_text("text") + "\n"
#     except Exception as e:
#         print(f"Error extracting text from {pdf_path}: {e}")
#         return None
#     return text

# def clean_text(text):
#     text = re.sub(r"\s+", " ", text.strip())
#     return text

# def chunk_text(text, max_words=500, overlap_words=100):
#     sentences = sent_tokenize(text)
#     chunks, current_chunk, word_count = [], [], 0
    
#     for sentence in sentences:
#         words = sentence.split()
#         word_count += len(words)
#         if word_count <= max_words:
#             current_chunk.append(sentence)
#         else:
#             chunks.append(" ".join(current_chunk))
#             total_words = sum(len(s.split()) for s in current_chunk)
#             overlap_idx = len(current_chunk) - 1
#             overlap_word_count = 0
#             while overlap_idx >= 0 and overlap_word_count < overlap_words:
#                 overlap_word_count += len(current_chunk[overlap_idx].split())
#                 overlap_idx -= 1
#             current_chunk = current_chunk[overlap_idx + 1:] + [sentence]
#             word_count = sum(len(s.split()) for s in current_chunk)
    
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
    
#     return [chunk.strip() for chunk in chunks if chunk.strip()]

# # --- Embedding Functions ---
# def count_tokens(text):
#     return len(tokenizer.encode(text))

# def get_embeddings(texts, max_tokens=8192, max_retries=5, initial_delay=1):
#     embeddings = []
#     for i, text in enumerate(texts):
#         if not text or not isinstance(text, str):
#             print(f"Skipping invalid chunk at index {i}: {text}")
#             continue
        
#         tokens = count_tokens(text)
#         if tokens > max_tokens:
#             print(f"Chunk {i} exceeds token limit ({tokens} > {max_tokens}): {text[:100]}...")
#             continue
        
#         retries = 0
#         while retries < max_retries:
#             try:
#                 response = client.embeddings.create(model="text-embedding-ada-002", input=text)
#                 embedding = response.data[0].embedding
#                 embeddings.append(embedding)
#                 break
#             except openai.RateLimitError as e:
#                 retries += 1
#                 delay = initial_delay * (2 ** (retries - 1))
#                 print(f"Rate limit hit for chunk {i}. Retrying in {delay} seconds... ({retries}/{max_retries})")
#                 time.sleep(delay)
#             except Exception as e:
#                 print(f"Error embedding chunk at index {i}: {e}")
#                 break
#         else:
#             print(f"Failed to embed chunk {i} after {max_retries} retries.")
    
#     if not embeddings:
#         raise ValueError("No valid embeddings generated.")
    
#     return np.array(embeddings, dtype=np.float32)

# # --- FAISS Indexing ---
# def build_faiss_index(embeddings):
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings)
#     return index

# def preprocess_and_index(raw_text, output_dir="models/embeddings", existing_chunks=None, existing_embeddings=None):
#     if not raw_text:
#         raise ValueError("Input text is empty or None.")
    
#     cleaned_text = clean_text(raw_text)
#     print(f"Cleaned text length: {len(cleaned_text)} characters")
    
#     new_chunks = chunk_text(cleaned_text, max_words=500, overlap_words=100)
#     print(f"Number of new chunks: {len(new_chunks)}")
    
#     all_chunks = existing_chunks + new_chunks if existing_chunks else new_chunks
#     print(f"Total chunks after combining: {len(all_chunks)}")
    
#     all_chunks = [chunk for chunk in all_chunks if chunk and isinstance(chunk, str)]
#     print(f"Number of valid chunks after filtering: {len(all_chunks)}")
#     if not all_chunks:
#         raise ValueError("No valid chunks found after preprocessing.")
    
#     print("Sample chunks:")
#     for i, chunk in enumerate(all_chunks[:3]):
#         print(f"Chunk {i+1}: {chunk[:100]}...")
    
#     if existing_embeddings is not None and len(new_chunks) > 0:
#         new_embeddings = get_embeddings(new_chunks)
#         all_embeddings = np.vstack((existing_embeddings, new_embeddings))
#     else:
#         all_embeddings = get_embeddings(all_chunks)
#     print(f"Embeddings shape: {all_embeddings.shape}")
    
#     index = build_faiss_index(all_embeddings)
#     print("FAISS index built successfully.")
    
#     os.makedirs(output_dir, exist_ok=True)
#     with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
#         pickle.dump(all_chunks, f)
#     np.save(os.path.join(output_dir, "embeddings.npy"), all_embeddings)
#     faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
#     print("Chunks, embeddings, and index saved successfully.")
    
#     return all_chunks, all_embeddings, index

# def load_knowledge_base(output_dir="models/embeddings"):
#     if not os.path.exists(output_dir):
#         return None, None, None
#     try:
#         with open(os.path.join(output_dir, "chunks.pkl"), "rb") as f:
#             chunks = pickle.load(f)
#         embeddings = np.load(os.path.join(output_dir, "embeddings.npy"))
#         index = faiss.read_index(os.path.join(output_dir, "faiss_index.bin"))
#         return chunks, embeddings, index
#     except Exception as e:
#         print(f"Error loading knowledge base: {e}")
#         return None, None, None

# # --- Query Handling ---
# def query_faiss(query, chunks, index, k=3):
#     query_embedding = get_embeddings([query])[0]
#     distances, indices = index.search(np.array([query_embedding]), k)
#     return [chunks[idx] for idx in indices[0]]

# def generate_response(query, retrieved_chunks, max_context_tokens=4000, max_retries=3, initial_delay=1):
#     global conversation_history
#     conversation_history.append({"role": "user", "content": query})
    
#     # Build context with truncation
#     history_text = "\n".join(f"{m['role']}: {m['content']}" for m in conversation_history)
#     chunks_text = "\n".join(retrieved_chunks)
#     context = f"Conversation History:\n{history_text}\nRetrieved Context:\n{chunks_text}"
    
#     # Truncate context if too long
#     context_tokens = count_tokens(context)
#     if context_tokens > max_context_tokens:
#         print(f"Context exceeds {max_context_tokens} tokens ({context_tokens}). Truncating...")
#         # Truncate retrieved chunks first, then history if needed
#         chunks_text = "\n".join(chunk[:int(2000/len(retrieved_chunks))] for chunk in retrieved_chunks)  # Rough split
#         context = f"Conversation History:\n{history_text}\nRetrieved Context:\n{chunks_text}"
#         if count_tokens(context) > max_context_tokens:
#             history_text = history_text[:int(max_context_tokens/2)]
#             context = f"Conversation History:\n{history_text}\nRetrieved Context:\n{chunks_text}"
    
#     retries = 0
#     while retries < max_retries:
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4",
#                 messages=[
#                     {"role": "system", "content": SYSTEM_PROMPT},
#                     {"role": "user", "content": context}
#                 ],
#                 max_tokens=300,
#                 temperature=0.7
#             )
#             return response.choices[0].message.content.strip()
#         except openai.InternalServerError as e:
#             retries += 1
#             delay = initial_delay * (2 ** (retries - 1))
#             print(f"Internal Server Error: {e}. Retrying in {delay} seconds... ({retries}/{max_retries})")
#             time.sleep(delay)
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             break
#     print("Failed to generate response after retries.")
#     return "Sorry, I couldn’t generate a response due to a server error. Please try again later."

# def process_user_input(user_input, chunks, index):
#     retrieved_chunks = query_faiss(user_input, chunks, index)
#     print("\nRetrieved Chunks:")
#     for i, chunk in enumerate(retrieved_chunks):
#         print(f"Chunk {i + 1} (tokens: {count_tokens(chunk)}): {chunk[:100]}...")
#     response = generate_response(user_input, retrieved_chunks)
#     conversation_history.append({"role": "assistant", "content": response})
#     return response

# # --- Main Loop ---
# def main():
#     initial_pdf_path = "D:\\Downloads\\maxim\\Google ads knowledge base master.pdf"
#     model_dir = "models/embeddings"
#     pdf_storage_dir = PDF_STORAGE_DIR

#     if not os.path.exists(initial_pdf_path):
#         print(f"Error: Initial PDF file '{initial_pdf_path}' not found.")
#         return

#     initial_stored_path = os.path.join(pdf_storage_dir, "initial_google_ads_knowledge.pdf")
#     if not os.path.exists(initial_stored_path):
#         shutil.copy(initial_pdf_path, initial_stored_path)
#         print(f"Initial PDF copied to {initial_stored_path}")

#     chunks, embeddings, index = load_knowledge_base(model_dir)
#     if chunks is None or embeddings is None or index is None:
#         raw_text = extract_text_from_pdf(initial_stored_path)
#         if not raw_text:
#             print("Error: Could not extract text from the initial PDF.")
#             return
#         chunks, embeddings, index = preprocess_and_index(raw_text, model_dir)
#         if chunks is None or embeddings is None or index is None:
#             print("Error: Failed to preprocess and index the initial document.")
#             return

#     print("Chatbot is ready! You can provide text or upload a new PDF to add to the knowledge base.")
#     while True:
#         user_input = input("\nEnter your input (text or 'upload' to add a new PDF): ").strip()
#         if user_input.lower() in ["exit", "quit"]:
#             print("Exiting chatbot.")
#             break
#         elif user_input.lower() == "upload":
#             new_pdf_path = input("Enter the path to the new PDF: ").strip()
#             if not os.path.exists(new_pdf_path):
#                 print("Error: File not found.")
#                 continue
            
#             new_pdf_filename = f"added_pdf_{int(time.time())}.pdf"
#             stored_pdf_path = os.path.join(pdf_storage_dir, new_pdf_filename)
#             shutil.copy(new_pdf_path, stored_pdf_path)
#             print(f"New PDF stored at {stored_pdf_path}")
            
#             new_text = extract_text_from_pdf(stored_pdf_path)
#             if not new_text:
#                 print("Error: Could not extract text from the new PDF.")
#                 continue
#             chunks, embeddings, index = preprocess_and_index(new_text, model_dir, existing_chunks=chunks, existing_embeddings=embeddings)
#             if chunks is None or embeddings is None or index is None:
#                 print("Error: Failed to update the knowledge base with the new PDF.")
#                 continue
#             print("New PDF added to the knowledge base!")
#         else:
#             response = process_user_input(user_input, chunks, index)
#             print(f"\nResponse: {response}")

# if __name__ == "__main__":
#     main()



# import openai
# import re
# import pickle
# import os
# import numpy as np
# import faiss
# import fitz  # PyMuPDF
# import nltk
# from nltk.tokenize import sent_tokenize
# from dotenv import load_dotenv
# import tiktoken
# import time
# import shutil

# nltk.download('punkt')
# load_dotenv()

# # Initialize OpenAI client
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# if not os.getenv("OPENAI_API_KEY"):
#     raise ValueError("OPENAI_API_KEY not found in environment variables.")

# tokenizer = tiktoken.get_encoding("cl100k_base")
# conversation_history = []  # 🔹 Global conversation memory

# # 🔥 System Prompt to Prevent Hallucination 🔥
# SYSTEM_PROMPT = """
# You are a Google Ads assistant. Your ONLY task is to return direct, verbatim quotes from the knowledge base below.
# If the knowledge base does not contain relevant information, respond with:  
# "I don't have enough information in my knowledge base to answer that."

# 📖 **RULES**:
# 1️⃣ Use only the retrieved document chunks to answer.
# 2️⃣ If needed, consider previous user inputs to understand context.
# 3️⃣ If the user's question is unclear, ask for clarification instead of guessing.
# """

# PDF_STORAGE_DIR = "data/pdf_knowledge_base"
# os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

# # --- Text Preprocessing ---
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     try:
#         with fitz.open(pdf_path) as doc:
#             for page in doc:
#                 text += page.get_text("text") + "\n"
#     except Exception as e:
#         print(f"Error extracting text from {pdf_path}: {e}")
#         return None
#     return text

# def clean_text(text):
#     return re.sub(r"\s+", " ", text.strip())

# def chunk_text(text, max_words=500, overlap_words=100):
#     sentences = sent_tokenize(text)
#     chunks, current_chunk, word_count = [], [], 0
    
#     for sentence in sentences:
#         words = sentence.split()
#         word_count += len(words)
#         if word_count <= max_words:
#             current_chunk.append(sentence)
#         else:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = current_chunk[-overlap_words:] + [sentence]
#             word_count = sum(len(s.split()) for s in current_chunk)
    
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
    
#     return [chunk.strip() for chunk in chunks if chunk.strip()]

# # --- Embedding & FAISS Index ---
# def count_tokens(text):
#     return len(tokenizer.encode(text))

# def get_embeddings(texts, max_retries=5, initial_delay=1):
#     embeddings = []
#     for i, text in enumerate(texts):
#         retries = 0
#         while retries < max_retries:
#             try:
#                 response = client.embeddings.create(model="text-embedding-ada-002", input=text)
#                 embedding = response.data[0].embedding
#                 embeddings.append(embedding)
#                 break
#             except openai.RateLimitError:
#                 time.sleep(initial_delay * (2 ** retries))
#                 retries += 1
#             except Exception as e:
#                 print(f"Error embedding chunk {i}: {e}")
#                 break
#     return np.array(embeddings, dtype=np.float32)

# def build_faiss_index(embeddings):
#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(embeddings)
#     return index

# def preprocess_and_index(raw_text, output_dir="models/embeddings"):
#     cleaned_text = clean_text(raw_text)
#     new_chunks = chunk_text(cleaned_text, max_words=500, overlap_words=100)

#     embeddings = get_embeddings(new_chunks)
#     index = build_faiss_index(embeddings)

#     os.makedirs(output_dir, exist_ok=True)
#     with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
#         pickle.dump(new_chunks, f)
#     np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
#     faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))

#     return new_chunks, embeddings, index

# def load_knowledge_base(output_dir="models/embeddings"):
#     try:
#         with open(os.path.join(output_dir, "chunks.pkl"), "rb") as f:
#             chunks = pickle.load(f)
#         embeddings = np.load(os.path.join(output_dir, "embeddings.npy"))
#         index = faiss.read_index(os.path.join(output_dir, "faiss_index.bin"))
#         return chunks, embeddings, index
#     except:
#         return None, None, None

# # --- Query Handling ---
# def query_faiss(query, chunks, index, k=3):
#     query_embedding = get_embeddings([query])[0]
#     distances, indices = index.search(np.array([query_embedding]), k)
    
#     retrieved_chunks = [chunks[idx] for idx in indices[0]]

#     return retrieved_chunks

# def generate_response(query, retrieved_chunks):
#     global conversation_history  

#     conversation_history.append({"role": "user", "content": query})

#     if not retrieved_chunks:
#         return "I don't have enough information in my knowledge base to answer that."

#     # 🔹 Maintain full chat memory
#     past_conversation = "\n".join(
#         [f"{msg['role']}: {msg['content']}" for msg in conversation_history[-6:]]
#     )  

#     # 🔥 Dynamic context understanding (No predefined follow-up phrases)
#     structured_response = f"""
# 📖 **Based on the knowledge base, here are the relevant insights:**

# 🔹 **Key Insights:**
# - {retrieved_chunks[0]}

# 🛠 **Recommended Optimization Steps:**
# - {retrieved_chunks[1]}

# ❗ **Potential Issues Identified:**
# - {retrieved_chunks[2]}

# 🔗 **References:** Retrieved from the Google Ads Knowledge Base (PDF).

# 🗣 **Previous Conversation Context:**
# {past_conversation}
# """

#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": structured_response}
#         ],
#         max_tokens=600,
#         temperature=0.2  
#     )

#     return response.choices[0].message.content.strip()

# def process_user_input(user_input, chunks, index):
#     retrieved_chunks = query_faiss(user_input, chunks, index)
#     if not retrieved_chunks:
#         return "I don't have enough information in my knowledge base to answer that."

#     # 🔹 Append the user’s input to the conversation history
#     conversation_history.append({"role": "user", "content": user_input})

#     # 🔹 Generate AI response while considering past conversation
#     response = generate_response(user_input, retrieved_chunks)

#     # 🔹 Append AI response to conversation history
#     conversation_history.append({"role": "assistant", "content": response})

#     return response

# # --- Main Execution ---
# def main():
#     pdf_path = "D:\\Downloads\\maxim\\Google ads knowledge base master.pdf"
#     model_dir = "models/embeddings"

#     chunks, embeddings, index = load_knowledge_base(model_dir)

#     print("🔹 AI Chatbot Ready! Ask Google Ads questions.")
#     while True:
#         user_input = input("\n💬 Enter your question (or type 'exit' to quit): ").strip()
#         if user_input.lower() in ["exit", "quit"]:
#             print("👋 Exiting chatbot.")
#             break
#         else:
#             response = process_user_input(user_input, chunks, index)
#             print(f"\n🤖 AI Response: {response}")

# if __name__ == "__main__":
#     main()



# import openai
# import re
# import pickle
# import os
# import numpy as np
# import faiss
# import fitz  # PyMuPDF
# import nltk
# from nltk.tokenize import sent_tokenize
# from dotenv import load_dotenv
# import tiktoken
# import time
# import shutil

# nltk.download('punkt')
# load_dotenv()

# # Initialize OpenAI client
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# if not os.getenv("OPENAI_API_KEY"):
#     raise ValueError("OPENAI_API_KEY not found in environment variables.")

# tokenizer = tiktoken.get_encoding("cl100k_base")
# conversation_history = []  # Global conversation history

# # 🔥 System Prompt to Prevent Hallucination (for Google Ads topics)
# SYSTEM_PROMPT = """
# You are a Google Ads assistant. Your ONLY task is to return direct, verbatim quotes from the knowledge base below.
# If the knowledge base does not contain relevant information, respond with:  
# "I don't have enough information in my knowledge base to answer that."

# 📖 **RULES**:
# 1️⃣ Use only the retrieved document chunks to answer.
# 2️⃣ If needed, consider previous user inputs to understand context.
# 3️⃣ If the user's question is unclear, ask for clarification instead of guessing.
# """

# PDF_STORAGE_DIR = "data/pdf_knowledge_base"
# os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

# # --- Text Preprocessing ---
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     try:
#         with fitz.open(pdf_path) as doc:
#             for page in doc:
#                 text += page.get_text("text") + "\n"
#     except Exception as e:
#         print(f"Error extracting text from {pdf_path}: {e}")
#         return None
#     return text

# def clean_text(text):
#     return re.sub(r"\s+", " ", text.strip())

# def chunk_text(text, max_words=500, overlap_words=100):
#     sentences = sent_tokenize(text)
#     chunks = []
#     current_chunk = []
#     word_count = 0
#     for sentence in sentences:
#         words = sentence.split()
#         word_count += len(words)
#         if word_count <= max_words:
#             current_chunk.append(sentence)
#         else:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = current_chunk[-overlap_words:] + [sentence]
#             word_count = sum(len(s.split()) for s in current_chunk)
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
#     return [chunk.strip() for chunk in chunks if chunk.strip()]

# # --- Embedding & FAISS Index ---
# def count_tokens(text):
#     return len(tokenizer.encode(text))

# def truncate_text(text, max_tokens):
#     tokens = tokenizer.encode(text)
#     if len(tokens) > max_tokens:
#         return tokenizer.decode(tokens[:max_tokens])
#     return text

# def get_embeddings(texts, batch_size=10, max_retries=5, initial_delay=1):
#     """
#     Batch process embeddings without using unsupported parameters.
#     """
#     embeddings = []
#     total_batches = (len(texts) - 1) // batch_size + 1
#     for batch_start in range(0, len(texts), batch_size):
#         batch = texts[batch_start:batch_start + batch_size]
#         print(f"🔄 Processing batch {batch_start // batch_size + 1} of {total_batches}")
#         retries = 0
#         while retries < max_retries:
#             try:
#                 response = client.embeddings.create(model="text-embedding-ada-002", input=batch)
#                 batch_embeddings = [item.embedding for item in response.data]
#                 embeddings.extend(batch_embeddings)
#                 break
#             except openai.RateLimitError:
#                 time.sleep(initial_delay * (2 ** retries))
#                 retries += 1
#             except Exception as e:
#                 print(f"Error embedding batch {batch_start // batch_size + 1}: {e}")
#                 break
#     return np.array(embeddings, dtype=np.float32)

# def build_faiss_index(embeddings):
#     # Normalize embeddings to improve similarity search accuracy
#     norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
#     normalized_embeddings = embeddings / norms
#     index = faiss.IndexFlatL2(normalized_embeddings.shape[1])
#     index.add(normalized_embeddings)
#     return index

# def preprocess_and_index(raw_text, output_dir="models/embeddings"):
#     print("🔄 Cleaning text...")
#     cleaned_text = clean_text(raw_text)
#     print("✅ Text cleaned successfully.")
    
#     print("🔄 Chunking text into smaller segments...")
#     new_chunks = chunk_text(cleaned_text, max_words=500, overlap_words=100)
#     if not new_chunks:
#         print("❌ Error: No text chunks were created. PDF extraction might have failed.")
#         return [], None, None
#     print(f"✅ Created {len(new_chunks)} text chunks.")
    
#     print("🔄 Generating embeddings...")
#     embeddings = get_embeddings(new_chunks)
#     if embeddings.size == 0:
#         print("❌ Error: Embeddings were not generated properly.")
#         return [], None, None
#     print("✅ All embeddings generated successfully.")
    
#     print("🔄 Building FAISS index...")
#     index = build_faiss_index(embeddings)
#     if index is None or index.ntotal == 0:
#         print("❌ Error: FAISS index is empty. Something went wrong during indexing.")
#         return [], None, None
#     print("✅ FAISS index built successfully.")
    
#     os.makedirs(output_dir, exist_ok=True)
#     with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
#         pickle.dump(new_chunks, f)
#     np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
#     faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
#     return new_chunks, embeddings, index

# def load_knowledge_base(output_dir="models/embeddings"):
#     try:
#         with open(os.path.join(output_dir, "chunks.pkl"), "rb") as f:
#             chunks = pickle.load(f)
#         embeddings = np.load(os.path.join(output_dir, "embeddings.npy"))
#         index = faiss.read_index(os.path.join(output_dir, "faiss_index.bin"))
#         return chunks, embeddings, index
#     except Exception as e:
#         print(f"❌ Error loading knowledge base: {e}")
#         return None, None, None

# # --- Query Handling ---
# def query_faiss(query, chunks, index, k=3):
#     query_embedding = get_embeddings([query])[0]
#     query_embedding = query_embedding / np.linalg.norm(query_embedding)
#     distances, indices = index.search(np.array([query_embedding]), k)
#     # If the best (smallest) distance is too high, assume the query is not related
#     threshold = 0.7  # Adjust threshold as needed
#     if distances[0][0] > threshold:
#         return []  # Not relevant to knowledge base
#     retrieved_chunks = [chunks[idx] for idx in indices[0]]
#     return retrieved_chunks

# def generate_response(query, retrieved_chunks):
#     global conversation_history
#     # Append the current user query to the conversation history
#     conversation_history.append({"role": "user", "content": query})
    
#     # If no relevant knowledge base info is found, use fallback with conversation history
#     if not retrieved_chunks:
#         fallback_messages = []
#         fallback_messages.append({"role": "system", "content": "You are a helpful assistant that remembers previous conversation."})
#         fallback_messages.extend(conversation_history[-10:])
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=fallback_messages,
#             max_tokens=200,
#             temperature=0.2
#         )
#         assistant_response = response.choices[0].message.content.strip()
#         conversation_history.append({"role": "assistant", "content": assistant_response})
#         return assistant_response

#     # Truncate each retrieved chunk to a maximum of 200 tokens
#     max_chunk_tokens = 200
#     retrieved_chunks_truncated = [truncate_text(chunk, max_chunk_tokens) for chunk in retrieved_chunks]
    
#     # Build the messages list for the Chat API using both conversation history and knowledge base info
#     messages = []
#     messages.append({"role": "system", "content": SYSTEM_PROMPT})
#     # Include recent conversation history (only the last 4 messages)
#     messages.extend(conversation_history[-4:])
    
#     # Add a message that provides the retrieved knowledge base information
#     kb_message = (
#         f"Knowledge Base:\n"
#         f"- Key Insights: {retrieved_chunks_truncated[0]}\n"
#         f"- Recommended Optimization Steps: {retrieved_chunks_truncated[1]}\n"
#         f"- Potential Issues Identified: {retrieved_chunks_truncated[2]}"
#     )
#     messages.append({"role": "assistant", "content": kb_message})
    
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=messages,
#         max_tokens=600,
#         temperature=0.2
#     )
    
#     assistant_response = response.choices[0].message.content.strip()
#     conversation_history.append({"role": "assistant", "content": assistant_response})
#     return assistant_response

# def process_user_input(user_input, chunks, index):
#     retrieved_chunks = query_faiss(user_input, chunks, index)
#     response = generate_response(user_input, retrieved_chunks)
#     return response

# def main():
#     pdf_path = "D:\\Downloads\\maxim\\Google ads knowledge base master.pdf"
#     model_dir = "models/embeddings"
    
#     # Load or preprocess the knowledge base
#     chunks, embeddings, index = load_knowledge_base(model_dir)
#     if chunks is None or index is None or index.ntotal == 0:
#         print("ℹ️ Knowledge base not found or empty. Attempting to process PDF...")
#         raw_text = extract_text_from_pdf(pdf_path)
#         if raw_text is None:
#             print("❌ Error extracting text from PDF.")
#             return
#         chunks, embeddings, index = preprocess_and_index(raw_text, output_dir=model_dir)
    
#     print("🔹 AI Chatbot Ready! Ask Google Ads questions.")
#     while True:
#         user_input = input("\n💬 Enter your question (or type 'exit' to quit): ").strip()
#         if user_input.lower() in ["exit", "quit"]:
#             print("👋 Exiting chatbot.")
#             break
#         else:
#             response = process_user_input(user_input, chunks, index)
#             print(f"\n🤖 AI Response: {response}")

# if __name__ == "__main__":
#     main()





# import openai
# import re
# import pickle
# import os
# import numpy as np
# import faiss
# import fitz  # PyMuPDF
# import nltk
# from nltk.tokenize import sent_tokenize
# from dotenv import load_dotenv
# import tiktoken
# import time
# import shutil

# nltk.download('punkt')
# load_dotenv()

# # Initialize OpenAI client
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# if not os.getenv("OPENAI_API_KEY"):
#     raise ValueError("OPENAI_API_KEY not found in environment variables.")

# tokenizer = tiktoken.get_encoding("cl100k_base")
# conversation_history = []  # Global conversation history

# # 🔥 System Prompt: Focus solely on Google Ads using the provided knowledge base.
# SYSTEM_PROMPT = """
# You are a Google Ads assistant. Your task is to answer questions about Google Ads campaigns using the provided knowledge base.
# When answering, include any relevant direct, verbatim quotes from the knowledge base and provide suggestions or additional commentary if appropriate.
# If the knowledge base does not contain relevant information, respond with:
# "I don't have enough information in my knowledge base to answer that."
# """

# PDF_STORAGE_DIR = "data/pdf_knowledge_base"
# os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

# # --- Text Preprocessing ---
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     try:
#         with fitz.open(pdf_path) as doc:
#             for page in doc:
#                 text += page.get_text("text") + "\n"
#     except Exception as e:
#         print(f"Error extracting text from {pdf_path}: {e}")
#         return None
#     return text

# def clean_text(text):
#     return re.sub(r"\s+", " ", text.strip())

# def chunk_text(text, max_words=500, overlap_words=100):
#     sentences = sent_tokenize(text)
#     chunks = []
#     current_chunk = []
#     word_count = 0
#     for sentence in sentences:
#         words = sentence.split()
#         word_count += len(words)
#         if word_count <= max_words:
#             current_chunk.append(sentence)
#         else:
#             chunks.append(" ".join(current_chunk))
#             # Carry over the last few sentences for overlap
#             current_chunk = current_chunk[-overlap_words:] + [sentence]
#             word_count = sum(len(s.split()) for s in current_chunk)
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
#     return [chunk.strip() for chunk in chunks if chunk.strip()]

# # --- Embedding & FAISS Index ---
# def count_tokens(text):
#     return len(tokenizer.encode(text))

# def truncate_text(text, max_tokens):
#     tokens = tokenizer.encode(text)
#     if len(tokens) > max_tokens:
#         return tokenizer.decode(tokens[:max_tokens])
#     return text

# def get_embeddings(texts, batch_size=10, max_retries=5, initial_delay=1):
#     """
#     Batch process embeddings without using unsupported parameters.
#     """
#     embeddings = []
#     total_batches = (len(texts) - 1) // batch_size + 1
#     for batch_start in range(0, len(texts), batch_size):
#         batch = texts[batch_start:batch_start + batch_size]
#         print(f"🔄 Processing batch {batch_start // batch_size + 1} of {total_batches}")
#         retries = 0
#         while retries < max_retries:
#             try:
#                 response = client.embeddings.create(model="text-embedding-ada-002", input=batch)
#                 batch_embeddings = [item.embedding for item in response.data]
#                 embeddings.extend(batch_embeddings)
#                 break
#             except openai.RateLimitError:
#                 time.sleep(initial_delay * (2 ** retries))
#                 retries += 1
#             except Exception as e:
#                 print(f"Error embedding batch {batch_start // batch_size + 1}: {e}")
#                 break
#     return np.array(embeddings, dtype=np.float32)

# def build_faiss_index(embeddings):
#     # Normalize embeddings to improve similarity search accuracy
#     norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
#     normalized_embeddings = embeddings / norms
#     index = faiss.IndexFlatL2(normalized_embeddings.shape[1])
#     index.add(normalized_embeddings)
#     return index

# def preprocess_and_index(raw_text, output_dir="models/embeddings"):
#     print("🔄 Cleaning text...")
#     cleaned_text = clean_text(raw_text)
#     print("✅ Text cleaned successfully.")
    
#     print("🔄 Chunking text into smaller segments...")
#     new_chunks = chunk_text(cleaned_text, max_words=500, overlap_words=100)
#     if not new_chunks:
#         print("❌ Error: No text chunks were created. PDF extraction might have failed.")
#         return [], None, None
#     print(f"✅ Created {len(new_chunks)} text chunks.")
    
#     print("🔄 Generating embeddings...")
#     embeddings = get_embeddings(new_chunks)
#     if embeddings.size == 0:
#         print("❌ Error: Embeddings were not generated properly.")
#         return [], None, None
#     print("✅ All embeddings generated successfully.")
    
#     print("🔄 Building FAISS index...")
#     index = build_faiss_index(embeddings)
#     if index is None or index.ntotal == 0:
#         print("❌ Error: FAISS index is empty. Something went wrong during indexing.")
#         return [], None, None
#     print("✅ FAISS index built successfully.")
    
#     os.makedirs(output_dir, exist_ok=True)
#     with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
#         pickle.dump(new_chunks, f)
#     np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
#     faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
#     return new_chunks, embeddings, index

# def load_knowledge_base(output_dir="models/embeddings"):
#     try:
#         with open(os.path.join(output_dir, "chunks.pkl"), "rb") as f:
#             chunks = pickle.load(f)
#         embeddings = np.load(os.path.join(output_dir, "embeddings.npy"))
#         index = faiss.read_index(os.path.join(output_dir, "faiss_index.bin"))
#         return chunks, embeddings, index
#     except Exception as e:
#         print(f"❌ Error loading knowledge base: {e}")
#         return None, None, None

# # --- Query Handling ---
# def query_faiss(query, chunks, index, k=3):
#     query_embedding = get_embeddings([query])[0]
#     query_embedding = query_embedding / np.linalg.norm(query_embedding)
#     distances, indices = index.search(np.array([query_embedding]), k)
#     # If the best (smallest) distance is too high, assume the query is not related
#     threshold = 0.7  # Adjust threshold as needed
#     if distances[0][0] > threshold:
#         return []  # Not relevant to knowledge base
#     retrieved_chunks = [chunks[idx] for idx in indices[0]]
#     return retrieved_chunks

# def generate_response(query, retrieved_chunks):
#     global conversation_history
#     # Append the current user query to the conversation history
#     conversation_history.append({"role": "user", "content": query})
    
#     # If no relevant knowledge base info is found, do not generate a fallback answer.
#     if not retrieved_chunks:
#         assistant_response = "I'm sorry, this question is outside the scope of Google Ads. Please ask a question related to Google Ads campaigns."
#         conversation_history.append({"role": "assistant", "content": assistant_response})
#         return assistant_response


#     # Truncate each retrieved chunk to a maximum of 200 tokens
#     max_chunk_tokens = 200
#     retrieved_chunks_truncated = [truncate_text(chunk, max_chunk_tokens) for chunk in retrieved_chunks]
    
#     # Build the messages list for the Chat API using conversation history and knowledge base info
#     messages = []
#     messages.append({"role": "system", "content": SYSTEM_PROMPT})
#     # Include recent conversation history (only the last 4 messages)
#     messages.extend(conversation_history[-4:])
    
#     # Combine the retrieved knowledge base chunks
#     kb_context = "\n---\n".join(retrieved_chunks_truncated)
#     kb_message = (
#         f"Knowledge Base Excerpts:\n{kb_context}\n\n"
#         "Based on the above excerpts, please answer the question about the Google Ads campaign. "
#         "Include any relevant direct quotes from the provided knowledge and provide suggestions or recommendations if needed."
#     )
#     messages.append({"role": "assistant", "content": kb_message})
    
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=messages,
#         max_tokens=600,
#         temperature=0.2
#     )
    
#     assistant_response = response.choices[0].message.content.strip()
#     conversation_history.append({"role": "assistant", "content": assistant_response})
#     return assistant_response

# def process_user_input(user_input, chunks, index):
#     retrieved_chunks = query_faiss(user_input, chunks, index)
#     response = generate_response(user_input, retrieved_chunks)
#     return response

# def main():
#     pdf_path = "D:\\Downloads\\maxim\\Google ads knowledge base master.pdf"
#     model_dir = "models/embeddings"
    
#     # Load or preprocess the knowledge base
#     chunks, embeddings, index = load_knowledge_base(model_dir)
#     if chunks is None or index is None or index.ntotal == 0:
#         print("ℹ️ Knowledge base not found or empty. Attempting to process PDF...")
#         raw_text = extract_text_from_pdf(pdf_path)
#         if raw_text is None:
#             print("❌ Error extracting text from PDF.")
#             return
#         chunks, embeddings, index = preprocess_and_index(raw_text, output_dir=model_dir)
    
#     print("🔹 AI Chatbot Ready! Ask Google Ads questions.")
#     while True:
#         user_input = input("\n💬 Enter your question (or type 'exit' to quit): ").strip()
#         if user_input.lower() in ["exit", "quit"]:
#             print("👋 Exiting chatbot.")
#             break
#         else:
#             response = process_user_input(user_input, chunks, index)
#             print(f"\n🤖 AI Response: {response}")

# if __name__ == "__main__":
#     main()









import argparse
import openai
import re
import pickle
import os
import numpy as np
import faiss
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
import tiktoken
import time
import shutil

nltk.download('punkt')
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

tokenizer = tiktoken.get_encoding("cl100k_base")
conversation_history = []  # Global conversation history

# 🔥 System Prompt: Focus solely on Google Ads using the provided knowledge base.
SYSTEM_PROMPT = """
You are a Google Ads expert assistant. Answer user questions based on the provided knowledge base excerpts.
Include direct, verbatim quotes from the knowledge base when available.  
If appropriate, clearly provide practical suggestions or recommendations in bullet points.

If a question does not explicitly mention Google Ads but reasonably relates to Google Ads campaigns, assume the user intends a Google Ads-related answer and respond accordingly using the provided knowledge base.

If the knowledge base does not contain information relevant to the user's intent, explicitly respond:
"I don't have enough information in my knowledge base to answer that."
"""


# Directory where PDFs will be stored
PDF_STORAGE_DIR = "data/pdf_knowledge_base"
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

# --- Text Preprocessing ---
def extract_text_from_pdf(pdf_path):
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
            # Carry over the last few sentences for overlap
            current_chunk = current_chunk[-overlap_words:] + [sentence]
            word_count = sum(len(s.split()) for s in current_chunk)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# --- Embedding & FAISS Index ---
def count_tokens(text):
    return len(tokenizer.encode(text))

def truncate_text(text, max_tokens):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        return tokenizer.decode(tokens[:max_tokens])
    return text

def get_embeddings(texts, batch_size=10, max_retries=5, initial_delay=1):
    """
    Batch process embeddings without using unsupported parameters.
    """
    embeddings = []
    total_batches = (len(texts) - 1) // batch_size + 1
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start:batch_start + batch_size]
        print(f"🔄 Processing batch {batch_start // batch_size + 1} of {total_batches}")
        retries = 0
        while retries < max_retries:
            try:
                response = client.embeddings.create(model="text-embedding-ada-002", input=batch)
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                break
            except openai.RateLimitError:
                time.sleep(initial_delay * (2 ** retries))
                retries += 1
            except Exception as e:
                print(f"Error embedding batch {batch_start // batch_size + 1}: {e}")
                break
    return np.array(embeddings, dtype=np.float32)

def build_faiss_index(embeddings):
    # Normalize embeddings to improve similarity search accuracy
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    index = faiss.IndexFlatL2(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)
    return index

def preprocess_and_index(raw_text, output_dir="models/embeddings"):
    print("🔄 Cleaning text...")
    cleaned_text = clean_text(raw_text)
    print("✅ Text cleaned successfully.")
    
    print("🔄 Chunking text into smaller segments...")
    new_chunks = chunk_text(cleaned_text, max_words=500, overlap_words=100)
    if not new_chunks:
        print("❌ Error: No text chunks were created. PDF extraction might have failed.")
        return [], None, None
    print(f"✅ Created {len(new_chunks)} text chunks.")
    
    print("🔄 Generating embeddings...")
    embeddings = get_embeddings(new_chunks)
    if embeddings.size == 0:
        print("❌ Error: Embeddings were not generated properly.")
        return [], None, None
    print("✅ All embeddings generated successfully.")
    
    print("🔄 Building FAISS index...")
    index = build_faiss_index(embeddings)
    if index is None or index.ntotal == 0:
        print("❌ Error: FAISS index is empty. Something went wrong during indexing.")
        return [], None, None
    print("✅ FAISS index built successfully.")
    
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
        print(f"❌ Error loading knowledge base: {e}")
        return None, None, None

# --- New: PDF Upload and KB Update ---
def upload_pdf(new_pdf_path, pdf_storage_dir=PDF_STORAGE_DIR, output_dir="models/embeddings"):
    if not os.path.exists(new_pdf_path):
        print(f"❌ The file {new_pdf_path} does not exist.")
        return

    # Copy the new PDF to the storage directory
    destination = os.path.join(pdf_storage_dir, os.path.basename(new_pdf_path))
    shutil.copy(new_pdf_path, destination)
    print(f"✅ PDF uploaded to {destination}")

    # Process the new PDF
    raw_text = extract_text_from_pdf(new_pdf_path)
    if raw_text is None:
        print("❌ Error processing PDF.")
        return
    new_chunks = chunk_text(clean_text(raw_text))
    if not new_chunks:
        print("❌ No text chunks created from the PDF.")
        return
    new_embeddings = get_embeddings(new_chunks)
    if new_embeddings.size == 0:
        print("❌ Error generating embeddings for the new PDF.")
        return

    # Load existing knowledge base, if any
    existing_chunks, existing_embeddings, index = load_knowledge_base(output_dir)
    if existing_chunks is None or index is None:
        # No existing KB; create one from the new PDF
        existing_chunks = new_chunks
        existing_embeddings = new_embeddings
        index = build_faiss_index(new_embeddings)
        print("✅ New knowledge base created from the uploaded PDF.")
    else:
        # Append new data to the existing KB
        existing_chunks.extend(new_chunks)
        combined_embeddings = np.concatenate((existing_embeddings, new_embeddings), axis=0)
        # Rebuild the index from scratch for simplicity
        index = build_faiss_index(combined_embeddings)
        existing_embeddings = combined_embeddings
        print("✅ Knowledge base updated with the new PDF.")

    # Save updated knowledge base
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(existing_chunks, f)
    np.save(os.path.join(output_dir, "embeddings.npy"), existing_embeddings)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    print("✅ Knowledge base saved successfully.")

# --- Query Handling ---
def query_faiss(query, chunks, index, k=5):
    query_embedding = get_embeddings([query])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    distances, indices = index.search(np.array([query_embedding]), k)
    # If the best (smallest) distance is too high, assume the query is not related
    threshold = 0.7  # Adjust threshold as needed
    if distances[0][0] > threshold:
        return []  # Not relevant to knowledge base
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    return retrieved_chunks

def generate_response(query, retrieved_chunks):
    global conversation_history
    # Append the current user query to the conversation history
    conversation_history.append({"role": "user", "content": query})
    
    # If no relevant knowledge base info is found, return an off-topic message.
    if not retrieved_chunks:
        assistant_response = "I'm sorry, this question is outside the scope of Google Ads. Please ask a question related to Google Ads campaigns."
        conversation_history.append({"role": "assistant", "content": assistant_response})
        return assistant_response

    # Truncate each retrieved chunk to a maximum of 200 tokens
    max_chunk_tokens = 200
    retrieved_chunks_truncated = [truncate_text(chunk, max_chunk_tokens) for chunk in retrieved_chunks]
    
    # Build the messages list for the Chat API using conversation history and knowledge base info
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    # Include recent conversation history (only the last 4 messages)
    messages.extend(conversation_history[-4:])
    
    # Combine the retrieved knowledge base chunks
    kb_context = "\n---\n".join(retrieved_chunks_truncated)
    kb_message = (
        f"Knowledge Base Excerpts:\n{kb_context}\n\n"
        "Based on the above excerpts, please answer the question about the Google Ads campaign. "
        "Include any relevant direct quotes from the provided knowledge and provide suggestions or recommendations if needed."
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

def process_user_input(user_input, chunks, index):
    retrieved_chunks = query_faiss(user_input, chunks, index)
    response = generate_response(user_input, retrieved_chunks)
    return response

def main():
    parser = argparse.ArgumentParser(description="Google Ads Chatbot with PDF upload functionality")
    parser.add_argument("--upload", type=str, help="Path to a new PDF file to upload and update the knowledge base")
    args = parser.parse_args()

    model_dir = "models/embeddings"
    
    # If --upload is provided, process the new PDF and update the KB.
    if args.upload:
        upload_pdf(args.upload, pdf_storage_dir=PDF_STORAGE_DIR, output_dir=model_dir)
    
    # Load or preprocess the existing knowledge base.
    chunks, embeddings, index = load_knowledge_base(model_dir)
    # If no KB exists, prompt for a PDF to initialize the KB.
    if chunks is None or index is None or index.ntotal == 0:
        pdf_path = input("Knowledge base is empty. Please enter the path to your initial PDF knowledge base: ").strip()
        raw_text = extract_text_from_pdf(pdf_path)
        if raw_text is None:
            print("❌ Error extracting text from PDF.")
            return
        chunks, embeddings, index = preprocess_and_index(raw_text, output_dir=model_dir)
    
    print("🔹 AI Chatbot Ready! Ask Google Ads questions.")
    while True:
        user_input = input("\n💬 Enter your question (or type 'exit' to quit): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting chatbot.")
            break
        else:
            response = process_user_input(user_input, chunks, index)
            print(f"\n🤖 AI Response: {response}")

if __name__ == "__main__":
    main()









