import chromadb
from langchain_ollama import OllamaEmbeddings
import json

# initialise persistent client
client = chromadb.PersistentClient(path="./data/vector_db")

# get current collection
collection = client.get_collection(name="Allianz_20251219")

# store question
question = "What are my benefits?"

# embed query using same embedding model used during ingestion
query_vector = OllamaEmbeddings(model="nomic-embed-text").embed_query(question)

# store most relevant chunks
results = collection.query(query_embeddings=[query_vector], n_results=5)

# print results in json
print(json.dumps(results, indent=2))