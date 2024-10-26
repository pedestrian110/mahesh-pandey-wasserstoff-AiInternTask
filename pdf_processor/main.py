from fastapi import FastAPI
from pdf_processor import ingest_pdfs_in_folder, Watcher
import os

app = FastAPI()

# Set the folder to watch for PDF files (using relative path for Vercel compatibility)
folder_to_watch = os.getenv("PDF_FOLDER_PATH", "./pdf_file")

# Endpoint to trigger PDF ingestion and processing
@app.get("/ingest_pdfs")
async def ingest_pdfs():
    ingest_pdfs_in_folder(folder_to_watch)
    return {"message": "PDF ingestion started"}

# Main function to initialize folder watcher
def start_watcher():
    watcher = Watcher(folder_to_watch)
    watcher.run()

# For Vercel deployment, ensure app is accessible
if __name__ == "__main__":
    start_watcher()
