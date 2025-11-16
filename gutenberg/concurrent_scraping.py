import os
import csv
import shutil
import signal
import sys
import xml.etree.ElementTree as ET
import pandas as pd
from queue import Queue
from threading import Thread, Lock
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests

# File paths
xml_file = "pgmarc.xml"
output_csv = "gutenberg_books.csv"
text_dir = "gutenberg_texts"
checkpoint_interval = 1000  # Save progress every 1000 books

os.makedirs(text_dir, exist_ok=True)

# Namespace for MARC21 XML
namespace = {'marc': 'http://www.loc.gov/MARC21/slim'}

# Parse the XML file
tree = ET.parse(xml_file)
root = tree.getroot()

# Load previous progress if the CSV exists
if os.path.exists(output_csv):
    df = pd.read_csv(output_csv)
    downloaded_books = set(df["Link"].tolist())
    print(f"Resuming from {len(downloaded_books)} previously downloaded books.")
else:
    df = pd.DataFrame(columns=["Author", "Title", "Language", "Link", "Text Link", "Text File", "Text"])
    downloaded_books = set()
    print("Starting from scratch.")

# Extract basic metadata
books = []
for record in root.findall('marc:record', namespace):
    title = record.find('.//marc:datafield[@tag="245"]/marc:subfield[@code="a"]', namespace)
    author = record.find('.//marc:datafield[@tag="100"]/marc:subfield[@code="a"]', namespace)
    language = record.find('.//marc:datafield[@tag="041"]/marc:subfield[@code="a"]', namespace)
    link = record.find('.//marc:datafield[@tag="856"]/marc:subfield[@code="u"]', namespace)

    # Clean and validate fields
    title = title.text.strip() if title is not None else "Unknown Title"
    author = author.text.strip() if author is not None else "Unknown Author"
    language = language.text.strip() if language is not None else "Unknown Language"
    link = link.text.strip() if link is not None else None
    
    # Skip already downloaded books
    if link and link not in downloaded_books:
        books.append({
            "Author": author,
            "Title": title,
            "Language": language,
            "Link": link,
            "Text Link": None,
            "Text File": None,
            "Text": None
        })

# Create a queue for saving books to CSV
save_queue = Queue()
lock = Lock()

def save_to_csv(books):
    """Save a list of books to the main CSV file atomically with file locking."""
    temp_file = output_csv + ".tmp"
    try:
        with lock, open(temp_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=df.columns)
            # Write header only if the main file doesn't exist
            if not os.path.exists(output_csv):
                writer.writeheader()
            writer.writerows(books)
        
        # Append to the main CSV file
        with lock, open(output_csv, mode='a', newline='', encoding='utf-8') as f:
            shutil.copyfileobj(open(temp_file, 'r', encoding='utf-8'), f)
        os.remove(temp_file)
        print(f"Checkpoint saved: {len(df) + len(books)} books downloaded.")

    except Exception as e:
        print(f"Failed to save checkpoint: {e}")

def save_worker():
    """Worker thread that saves books to the CSV file."""
    buffer = []
    while True:
        book = save_queue.get()
        if book is None:
            break
        
        buffer.append(book)
        
        # Flush buffer to disk at intervals
        if len(buffer) >= checkpoint_interval:
            save_to_csv(buffer)
            buffer.clear()
        
        save_queue.task_done()
    
    # Final flush on exit
    if buffer:
        save_to_csv(buffer)

# Graceful shutdown handling
def handle_exit(signal_received, frame):
    """Ensure graceful exit on interrupt signals."""
    print("\nGracefully shutting down...")
    save_queue.put(None)
    save_thread.join()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# Start the saving thread
save_thread = Thread(target=save_worker, daemon=True)
save_thread.start()

# Function to download a single book
def download_book(book):
    try:
        session = requests.Session()
        response = session.get(book["Link"], timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find the "Plain Text UTF-8" link
        for a_tag in soup.find_all("a", href=True):
            if "Plain Text UTF-8" in a_tag.text:
                href = a_tag["href"]
                text_link = f"https://www.gutenberg.org{href}" if href.startswith("/") else href
                book["Text Link"] = text_link
                break
        
        # Download the actual text if a valid link was found
        if book["Text Link"]:
            text_response = session.get(book["Text Link"], timeout=10)
            if text_response.status_code == 200:
                book_text = text_response.text
                filename = f"{book['Author']}_{book['Title']}_{book['Language']}.txt".replace("/", "-").replace(" ", "_")[:100]
                filepath = os.path.join(text_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(book_text)
                
                book["Text File"] = filepath
                book["Text"] = book_text
                print(f"Downloaded: {book['Title']} by {book['Author']} ({book['Language']})")
                
                save_queue.put(book)
                    
    except Exception as e:
        print(f"Failed to download {book['Title']} by {book['Author']}: {e}")

# Download books in parallel
print("Downloading books in parallel...")
with ThreadPoolExecutor(max_workers=20) as executor:
    list(tqdm(executor.map(download_book, books), total=len(books)))

# Wait for all books to be saved
save_queue.join()

# Stop the save worker
save_queue.put(None)
save_thread.join()

print(f"Metadata saved to {output_csv} with {len(df)} books.")
 