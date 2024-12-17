from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(texts):
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts (list): A list of strings to generate embeddings for.
    
    Returns:
        np.ndarray: An array of embeddings.
    """
    try:
        embeddings = model.encode(texts, convert_to_tensor=True)
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None
