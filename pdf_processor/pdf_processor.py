import os
import time
import json
import PyPDF2
from transformers import pipeline
import spacy
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logging setup
logging.basicConfig(filename="pdf_processor.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# MongoDB Connection Setup
try:
    mongo_uri = os.getenv("MONGODB_URI")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=30000, socketTimeoutMS=30000, connectTimeoutMS=30000)
    db = client['PDF_KEY_EXTRACTER']
    collection = db['key_extracter']
    logging.info("MongoDB connected successfully.")
except Exception as e:
    logging.error(f"Error connecting to MongoDB: {e}")

# Initialize spaCy model and Hugging Face summarizer
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="t5-small")

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text() is not None)
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return None

# Function to extract keywords using spaCy
def extract_keywords(text):
    try:
        doc = nlp(text)
        keywords = {chunk.text for chunk in doc.noun_chunks if len(chunk.text) > 2}
        return list(keywords)
    except Exception as e:
        logging.error(f"Error extracting keywords: {e}")
        return []

# Function to summarize text
def summarize_text(text):
    try:
        max_len, min_len = (250, 100) if len(text) > 3000 else (150, 40)
        summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return "Summary not available due to error."

# Function to store PDF metadata in MongoDB
def store_pdf_metadata(pdf_path):
    metadata = {
        "file_name": os.path.basename(pdf_path),
        "file_path": pdf_path,
        "file_size": os.path.getsize(pdf_path),
        "created_at": datetime.now(),
        "status": "processing"
    }
    try:
        return collection.insert_one(metadata).inserted_id
    except Exception as e:
        logging.error(f"Error storing metadata for {pdf_path}: {e}")
        return None

# Function to update MongoDB entry after processing
def update_pdf_metadata(pdf_id, summary, keywords):
    update_data = {
        "summary": summary,
        "keywords": keywords,
        "status": "processed",
        "processed_at": datetime.now()
    }
    try:
        collection.update_one({"_id": pdf_id}, {"$set": update_data})
    except Exception as e:
        logging.error(f"Error updating metadata for PDF ID {pdf_id}: {e}")

# Function to process a PDF
def process_pdf(pdf_path):
    logging.info(f"Processing: {pdf_path}")
    pdf_id = store_pdf_metadata(pdf_path)
    if pdf_id is None:
        logging.error(f"Metadata storage failed for {pdf_path}. Skipping.")
        return

    text = extract_text_from_pdf(pdf_path)
    if text is None:
        logging.error(f"Skipping processing for {pdf_path} due to text extraction failure.")
        return

    keywords = extract_keywords(text)
    summary = summarize_text(text)

    update_pdf_metadata(pdf_id, summary, keywords)
    logging.info(f"Processed {pdf_path} - Keywords: {keywords} - Summary: {summary[:100]}...")

# Ingest PDFs in folder and process in parallel
def ingest_pdfs_in_folder(folder_path):
    pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pdf')]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pdf_files}
        for future in futures:
            future.result()

# Watchdog event handler
class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.pdf'):
            logging.info(f"New PDF detected: {event.src_path}")
            process_pdf(event.src_path)

# Watchdog observer
class Watcher:
    def __init__(self, folder_to_watch):
        self.folder_to_watch = folder_to_watch
        self.event_handler = Handler()
        self.observer = Observer()

    def run(self):
        self.observer.schedule(self.event_handler, self.folder_to_watch, recursive=False)
        self.observer.start()
        logging.info(f"Watching folder: {self.folder_to_watch} for new PDFs...")
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()

# Main block to start watching the folder
if __name__ == "__main__":
    folder_to_watch = os.getenv("PDF_FOLDER_PATH", "./pdf_file")
    ingest_pdfs_in_folder(folder_to_watch)
    watcher = Watcher(folder_to_watch)
    watcher.run()
