from transformers import pipeline

# Load a pre-trained language model for text generation
llm = pipeline('text-generation', model='gpt2')

def generate_response(relevant_chunks, user_query):
    """
    Generate a response based on relevant chunks and user query.
    
    Args:
        relevant_chunks (list): A list of relevant text chunks.
        user_query (str): The user's query.
    
    Returns:
        str: The generated response.
    """
    if not relevant_chunks:
        return "No relevant information found."

    context = "\n".join(relevant_chunks)
    prompt = f"Using the following context: {context}\nAnswer the question: {user_query}"
    
    try:
        response = llm(prompt, max_length=150, num_return_sequences=1)
        return response[0]['generated_text'].strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "An error occurred while generating the response."
