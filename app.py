from flask import Flask, request, jsonify, render_template
from scraper import scrape_website
from embeddings import generate_embeddings
from vector_db import VectorDatabase
from llm import generate_response
from image_processing import process_image
import logging

app = Flask(__name__)
vector_db = VectorDatabase()

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ingest', methods=['POST'])
def ingest_data():
    urls = request.json.get('urls')
    if not urls:
        return jsonify({"status": "error", "message": "No URLs provided."}), 400

    for url in urls:
        if vector_db.is_url_cached(url):
            logging.info(f"URL {url} is already cached.")
            continue

        content = scrape_website(url)
        if content:
            chunks = segment_content(content)
            embeddings = generate_embeddings(chunks)
            vector_db.store_embeddings(embeddings, metadata={"url": url})
            vector_db.cache_url(url)  # Cache the URL
        else:
            logging.error(f"Failed to scrape {url}.")
            return jsonify({"status": "error", "message": f"Failed to scrape {url}."}), 400

    return jsonify({"status": "success", "message": "Data ingested successfully."})

@app.route('/query', methods=['POST'])
def query_data():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"status": "error", "message": "No query provided."}), 400

    query_embedding = generate_embeddings([user_query])
    relevant_chunks = vector_db.retrieve_similar(query_embedding)
    response = generate_response(relevant_chunks, user_query)
    return jsonify({"response": response})

@app.route('/process_image', methods=['POST'])
def handle_image():
    file = request.files.get('image')
    if not file:
        return jsonify({"status": "error", "message": "No image provided."}), 400

    processed_image = process_image(file)
    return jsonify({"status": "Image processed", "data": processed_image})

@app.route('/feedback', methods=['POST'])
def feedback():
    user_feedback = request.json.get('feedback')
    if not user_feedback:
        return jsonify({"status": "error", "message": "No feedback provided."}), 400

    # Here you can implement logic to store feedback for future model improvements
    logging.info(f"User  feedback received: {user_feedback}")
    return jsonify({"status": "success", "message": "Feedback received."})

def segment_content(content):
    # Add text preprocessing steps here
    content = content.replace('\n', ' ').strip()  # Normalize newlines
    return content.split('. ')  # Split by sentences

if __name__ == '__main__':
    app.run(debug=True)
