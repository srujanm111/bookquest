import argparse
import openai
import chromadb
import textract
import tiktoken
import math
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Configuration
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 400
NUM_DOCUMENTS = 4

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
parser.add_argument('--apikey',
                    type=str,
                    required=True,
                    help='Your OpenAI API key')
args = parser.parse_args()

# OpenAI API Key Initialization
openai.api_key = args.apikey

# File Reading
print("Reading file...")
text = textract.process(args.file).decode()
encoding = tiktoken.encoding_for_model(GPT_MODEL)
tokens = encoding.encode(text)
num_chunks = math.ceil(len(tokens) / CHUNK_SIZE)
token_chunks = [tokens[i*CHUNK_SIZE:min((i+1)*CHUNK_SIZE, len(tokens))] for i in range(0, num_chunks)]
documents = [encoding.decode(chunk) for chunk in token_chunks]

# Vector Database Creation
print("Learning file contents...")
embeddings = [openai.Embedding.create(input=doc, model=EMBEDDING_MODEL)["data"][0]["embedding"] for doc in documents]
embedding_function = OpenAIEmbeddingFunction(openai.api_key, EMBEDDING_MODEL)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name='collection', embedding_function=embedding_function)
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
print(answer)
