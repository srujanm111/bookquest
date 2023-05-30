import argparse
import openai
import chromadb
import textract
import tiktoken
import math
import hashlib
import termcolor
import os
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Configuration
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 500
NUM_DOCUMENTS = 4
PERSIST_DIRECTORY = '.chroma-cache/'

# Command Line Arguments
parser = argparse.ArgumentParser(description="BookQuest Command Line Interface")
parser.add_argument('--file',
                    type=str,
                    required=True,
                    help='The file path to the book')
parser.add_argument('--question',
                    type=str,
                    required=True,
                    help='Your question to ask the AI')
parser.add_argument('--api-key',
                    dest='apikey',
                    type=str,
                    required=True,
                    help='Your OpenAI API key')
args = parser.parse_args()

# Initialization
openai.api_key = args.apikey
filename_hash = str(hashlib.md5(args.file.encode()).hexdigest())
chroma_client = chromadb.Client(Settings(persist_directory=PERSIST_DIRECTORY, chroma_db_impl="duckdb+parquet"))
embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=EMBEDDING_MODEL)
collections = [c.name for c in chroma_client.list_collections()]

# Show Configuration
# clear output due to issue with chromadb: https://github.com/chroma-core/chroma/issues/484
os.system('cls' if os.name == 'nt' else 'clear')
print(termcolor.colored("BookQuest - Use AI to Answer Questions from Books!", "green"))
print("Book: " + args.file)
print("Question: " + args.question)

# Get Collection
collection = None
if filename_hash in collections:
    # ask the user if they want to use the existing collection
    print(termcolor.colored("You've already learned this book before. Use cached learnings? (y/n) ", "blue"), end="")
    use_cache = input()
    if use_cache == 'y':
        collection = chroma_client.get_collection(name=filename_hash, embedding_function=embedding_function)
    else:
        chroma_client.delete_collection(name=filename_hash)
if collection is None:
    # File Reading
    print("Reading book... (this may take some time)")
    text = textract.process(args.file).decode()
    encoding = tiktoken.encoding_for_model(GPT_MODEL)
    tokens = encoding.encode(text)
    num_chunks = math.ceil(len(tokens) / CHUNK_SIZE)
    token_chunks = [tokens[i*CHUNK_SIZE:min((i+1)*CHUNK_SIZE, len(tokens))] for i in range(0, num_chunks)]
    documents = [encoding.decode(chunk) for chunk in token_chunks]

    # Collection Creation
    print("Learning book contents...")
    embeddings_result = openai.Embedding.create(input=documents, model=EMBEDDING_MODEL)["data"]
    embeddings = [embedding["embedding"] for embedding in embeddings_result]
    collection = chroma_client.create_collection(name=filename_hash, embedding_function=embedding_function)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(documents))],
    )

# Context Creation
query_embedding = openai.Embedding.create(input=args.question, model=EMBEDDING_MODEL)["data"][0]["embedding"]
query_result = collection.query(query_embeddings=query_embedding, n_results=NUM_DOCUMENTS)
most_similar_documents = query_result["documents"][0]
documents_context = "\n".join(most_similar_documents)
model_context = f'''
You have been given pieces of text from a larger file named {args.file}. You will use the following information from 
the contents of the file to answer the user's question. The information is contained within triple backticks. ```
{documents_context}```
'''

# Query the Model
print("Answering the question...")
response = openai.ChatCompletion.create(
    model=GPT_MODEL,
    messages=[
        {"role": "system", "content": model_context},
        {"role": "user", "content": f"Here is my question: {args.question}"},
    ],
    temperature=0,
)

# Print the Response
answer = response.choices[0]["message"]["content"]
print()
print(answer)
