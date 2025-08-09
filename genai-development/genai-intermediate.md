# Generative AI App Development - Intermediate Concepts

## 1.0 - Overview

### 1.1 - Objective

Embeddings play a crucial role in modern AI architectures, especially in Retrieval-Augmented Generation (RAG) systems. In RAG, embeddings are used to convert unstructured data (such as documents or knowledge bases) into dense vector representations. These vectors enable efficient semantic search and retrieval of relevant information, which can then be combined with generative models to produce more accurate, context-aware, and up-to-date responses. Mastering embeddings is essential for developers aiming to build robust, scalable, and intelligent AI applications that leverage both retrieval and generation capabilities.

The objective of this document is to equip intermediate-level developers with practical knowledge and skills for building generative AI applications, with a particular emphasis on embeddings, vector representations, vector databases, and similarity search techniques using Azure OpenAI services. Readers will learn how to generate and use embeddings, leverage Azure's vector database solutions, and implement similarity search to enhance AI-driven solutions.

### 1.2 - Requirements and environment setup

Requirements:

- Access to an Azure OpenAI GPT models.
- If you plan to test and run the provided code:
  - experience setting up a Python development environment and installing Python packages.
  - Intermediate development knowledge; specially calling REST APIs.

Environment setup:

- A GPT-4o or GPT-4.1 model deployed in Azure
- An GPT API key
  - In prod, Entra ID is recommended and login with `az login`
- Create a python environment
- Create an `.env` file with the following values:

```bash
FULL_ENDPOINT=https://YOUR_RESOURCE_NAME.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT_NAME/chat/completions?api-version=2025-01-01
FULL_EMB_ENDPOINT=https://<NAME>.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15
ENDPOINT=https://YOUR_RESOURCE_NAME.openai.azure.com/
API_KEY=<KEY>
API_VERSION=2025-01-01
GPT_MODEL=gpt-4o
EMB_MODEL=text-embedding-3-small
```

- Create and Python environment and install the following Python packages:
  - `pip install openai python-dotenv httpx azure-identity fastapi uvicorn[standard]`

> **Note:** To get the full OpenAI endpoint, in AI Foundry click on the model, and copy the full endpoint which has the following format: `https://YOUR_RESOURCE_NAME.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT_NAME/chat/completions?api-version=2024-02-01`
>

## 2.0 - Embeddings, vectors, dimension and vector databases

### 2.1 - What is an embedding, vector, and embedding dimension?

In the context of artificial intelligence (AI), an embedding is a representation of data in a format suitable for processing by machine learning models. It involves mapping high-dimensional data, such as text or images, into a lower-dimensional space while preserving relevant properties of the original data. This transformation is crucial for tasks like natural language processing, where words or sentences are converted into numerical vectors that capture semantic meaning.

A vector, in AI, is an array of numbers that represents a point in a multi-dimensional space. Each element of the array corresponds to a coordinate in one dimension of that space. Vectors are fundamental in machine learning as they provide a way to numerically represent and process data.

The vector dimension refers to the number of elements in the vector, which corresponds to the number of features or attributes the vector represents. In AI, each dimension can encode a specific characteristic of the data, and the overall vector provides a comprehensive representation of the data point.

Changing the vector dimension can have significant effects on the performance of AI models. Reducing dimensions, a process known as dimensionality reduction, can help to alleviate the curse of dimensionality, improve computational efficiency, and reduce storage requirements. However, it's essential to maintain a balance to ensure that the reduced representation still captures the critical information necessary for the AI model to perform its task effectively.

#### References

### 2.2 - What are some Azure OpenAI embedding models?

Azure OpenAI offers several embedding models that can be used to create dense vector representations of text. These embeddings can then be utilized in various machine learning tasks, such as semantic search or clustering. The vector sizes for these models vary, with some common configurations being 1536 dimensions for the text-embedding-3-small model and 3072 dimensions for the text-embedding-3-large model.

The embeddings can be resized to a different number of dimensions using the `dimensions` parameter in the API request. This allows the embeddings to retain their semantic properties even when the dimensionality is reduced, which can be beneficial for certain applications where smaller vector sizes are required, such as in resource-constrained environments or when optimizing for faster search performance.

#### Code

##### 2.2.1 - Calling an Azure OpenAI embedding endpoint using REST

```python
import requests
import os
from dotenv import load_env

# Read environment variables from .env file or the environment
load_env()
full_endpoint = os.getenv("FULL_ENDPOINT")
api_key = os.getenv("API_KEY")

headers = {
        "Content-Type": "application/json",
        "api-key": api_key
}

def get_embedding(input:str,model_version=2,dimensions=1536) -> tuple[str,list[float]]:
    json_data = json_data = {"input": input}

    if model_version == 3:
        json_data = {"input": input,"dimensions":dimensions}    
        
    response = requests.post(full_endpoint, 
                             headers=headers, 
                             json=json_data)                             
    response.raise_for_status()
    res = response.json()

    vector = res['data'][0]['embedding']
    print(f"Input: {input} Vector size: {len(vector)}")
    return (input,vector)

print(get_embedding("App Service is on of the most used services in Azure."))
```

##### 2.2.2 - Calling an Azure OpenAI embedding endpoint using the OpenAI SDK

```python
from openai import AzureOpenAI
import os
from dotenv import load_env

# Read environment variables from .env file or the environment
load_env()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("EMB_MODEL") # text-embedding-3-small

client = AzureOpenAI(api_key=api_key,azure_endpoint=endpoint, api_version=api_version)

def get_embedding(text, model=model, dimensions=NOT_GIVEN):
   text = text.replace("\n", " ")
   emb = client.embeddings.create(input = [text], model=model, dimensions=dimensions).data[0].embedding
   return emb

print(get_embedding("App Service is on of the most used services in Azure."))
```

### 2.3 - What is a vector databases?

A vector database is a specialized type of database designed to store, manage, and index high-dimensional vector data efficiently. These databases are particularly useful in AI for handling complex data representations that are transformed into vectors through machine learning models.

> **Tip:** A mature vector database provides a lot of features and functionality, but in simple applications vector could be stored and analyzed from a simple database such a SQLite database.

### 2.4 - What are some vector databases in Azure?

In Azure, there are several services that support vector database functionalities:

- **Azure AI Search**: Provides vector search capabilities, enabling the retrieval of information based on vector proximity or similarity, which is beneficial for semantic or contextual searches.
- **Azure Cosmos DB**: Offers integrated vector database capabilities, allowing for the storage, indexing, and querying of vector embeddings alongside the original data. This service is suitable for applications requiring high performance, scalability, and data consistency.
- **Azure SQL Database**: Recently announced support for vector data, allowing for the creation, storage, and efficient searching of vectors in a binary format.
- **Azure Cosmos DB for MongoDB**: Allows for the storage, indexing, and querying of high-dimensional vector data directly within the database, which can be advantageous for vector similarity search capabilities.

### 2.5 - What are some use cases for embeddings?

Some use cases for embeddings include:

- **Semantic Search**: Embeddings can capture the semantic meaning of text, allowing for more nuanced search results that go beyond keyword matching.
- **Clustering**: By converting text into numerical vectors, embeddings can be used to group similar items together, which is useful in organizing large datasets.
- **Recommendation Systems**: Embeddings help in identifying items that are similar to a user's past behavior or preferences, enhancing the performance of recommendation engines.
- **Anomaly Detection**: In datasets, embeddings can highlight items that are dissimilar from the rest, aiding in the detection of outliers or anomalies.
- **Natural Language Processing (NLP)**: Tasks like sentiment analysis, language translation, and topic modeling rely on embeddings to process and understand human language.
- **Information Retrieval**: Embeddings can improve the retrieval of relevant documents or information by understanding the context and content of the queries.

### 2.6 - What is Cosine similarity?

Cosine similarity is a metric used in the field of Artificial Intelligence to measure how similar two vectors are, irrespective of their size. Mathematically, it calculates the cosine of the angle between two vectors projected in a multi-dimensional space. This similarity measure is particularly useful because it is independent of the magnitude of the vectors, focusing solely on their orientation.

The main use cases of cosine similarity in AI include:

1. **Text Analysis**: It is widely used to assess the similarity of documents or texts by converting them into vector representations and then calculating the cosine similarity between them.
2. **Recommendation Systems**: Cosine similarity can help in recommending products, movies, or music by comparing user profiles or item descriptions as vectors.
3. **Machine Learning**: In clustering algorithms like k-means, cosine similarity is used to measure the similarity between different data points.
4. **Information Retrieval**: It aids in finding the documents most relevant to a search query by comparing the query vector with document vectors in a corpus.
5. **Data Mining**: Cosine similarity can be used to understand relationships and patterns within large datasets by comparing the vectors of different data items.

#### Code

##### 2.6.1 - Cosine similarity calculation without numpy (slower)

```python
import math

def cosine_similarity(embedding1 : list[float], embedding2: list[float]) -> float:
    # Calculate the dot product of the two embeddings
    dot_product = sum(x * y for x, y in zip(embedding1, embedding2))

    # Calculate the magnitudes of the two embeddings
    magnitude1 = math.sqrt(sum(x ** 2 for x in embedding1))
    magnitude2 = math.sqrt(sum(x ** 2 for x in embedding2))

    # Calculate the cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity

# Example usage:
embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5]
embedding2 = [0.6, 0.7, 0.8, 0.9, 1.0]

similarity = cosine_similarity(embedding1, embedding2)
print(similarity)

embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5]
embedding2 = [0.1, 0.2, 0.3, 0.4, 0.5]

similarity = cosine_similarity(embedding1, embedding2)
print(similarity)
```

##### 2.6.2 - Cosine similarity calculation with numpy

```python
import numpy as np

def cosine_similarity(embedding1 : list[float], embedding2 : list[float]) -> float:
    # Convert input to NumPy arrays if needed
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)

    # Calculate the dot product of the two embeddings
    dot_product = np.dot(embedding1, embedding2)

    # Calculate the magnitudes of the two embeddings
    magnitude1 = np.linalg.norm(embedding1)
    magnitude2 = np.linalg.norm(embedding2)

    # Calculate the cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity

# Example usage:
embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5]
embedding2 = [0.6, 0.7, 0.8, 0.9, 1.0]

similarity = cosine_similarity(embedding1, embedding2)
print(similarity)

embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5]
embedding2 = [0.1, 0.2, 0.3, 0.4, 0.5]

similarity = cosine_similarity(embedding1, embedding2)
print(similarity)
```

##### 2.6.3 - Cosine similarity search without a vector database

```python
import math
import requests
import os
from dotenv import load_env

# Read environment variables from .env file or the environment
load_env()

full_endpoint = os.getenv("FULL_ENDPOINT")
api_key = os.getenv("API_KEY")

headers = {
        "Content-Type": "application/json",
        "api-key": api_key
}

# Read environment variables from .env file or the environment
load_env()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("EMB_MODEL") # text-embedding-3-small

client = AzureOpenAI(api_key=api_key,azure_endpoint=endpoint, api_version=api_version)

def get_embedding(text, model=model, dimensions=1536) -> tuple[str,list[float]]:
   text = text.replace("\n", " ")
   emb = client.embeddings.create(input = [text], model=model, dimensions=dimensions).data[0].embedding
   return (text,emb)

def cosine_similarity(embedding1 : list[float], embedding2: list[float]) -> float:
    # Calculate the dot product of the two embeddings
    dot_product = sum(x * y for x, y in zip(embedding1, embedding2))

    # Calculate the magnitudes of the two embeddings
    magnitude1 = math.sqrt(sum(x ** 2 for x in embedding1))
    magnitude2 = math.sqrt(sum(x ** 2 for x in embedding2))

    # Calculate the cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity

if "__name__"=="__main__":
    content = [
        "Azure App Service enables you to host web applications in the cloud.",
        "Azure Functions allows you to run event-driven serverless code.",
        "Azure Logic Apps helps automate workflows and integrate services.",
        "Azure SQL Database is a fully managed relational database service.",
    ]

    ram_vector_database = [get_embedding(c,model,dimension) for c in content]

    (content,embedding) = get_embedding("What a PaaS database service in Azure?")
    print(content)
    print(embedding)

    # Perform near search with relevance and limits
    limit =3
    relevance=0.1
    results_list = []
    for entry in ram_vector_database:
        (content,entry_embedding) = entry
        cs = cosine_similarity(e1, entry_embedding)
        if cs>=relevance:
            results_list.append((content,cs))

    # print the results
    results_list.sort(key=lambda x: x[1], reverse=True)
    top_n = results_list[:limit]
    for entry in top_n:
        print(f"Similarity: {entry[1]}, Content: {entry[0]}")
```
