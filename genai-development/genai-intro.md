# Generative App Development - Foundational Concepts

## 1.0 - Foundational Concepts

### 1.1 - Objective

This guide is intended to provide AI developers with a thorough understanding of the concepts and principles involved in developing and utilizing generative AI applications.

Through this guide, we will delve into the key concepts, techniques, and best practices for developing and integrating generative AI models into various applications.

### 1.2 - Requirements and recommendations

- Access to an Azure OpenAI GPT account to deploy models.
- Experience setting up a python development environment and installing Python packages.
- Intermediate development knowledge; specially calling REST APIs.

#### References

- [Setting-up a Python environment with WSL and VS Code](https://genaitutor.am2703.com/?content=pyenv-wsl-vscode)

### 1.3 - What is a Large Language Model (LLM)?

Large Language Models (LLMs) are a cornerstone of generative AI. A large language model is a type of artificial intelligence that processes and generates human-like text by predicting the likelihood of a sequence of words. It's trained on vast amounts of text data and uses complex algorithms to understand and produce language in a way that is coherent and contextually relevant. These models can perform a variety of tasks, such as translation, summarization, answering questions, and even creating content. They are called 'large' because they consist of millions or even billions of parameters that help them understand the nuances of language. Their capabilities are continually evolving, making them powerful tools for both research and practical applications in numerous industries.

### 1.4 - Why is an LLM a Foundational Model?

Large language models (LLMs) are considered foundational models due to their extensive training on massive datasets, which enables them to understand and generate natural language. This foundational capability allows them to support a wide range of applications and tasks. Unlike models designed for specific domains, LLMs provide a broad base that can be adapted for various uses, making them more versatile and cost-effective.

The significance of LLMs extends beyond their technical capabilities; they have become integral to the adoption of AI across numerous business functions and use cases. Their ability to infer from context and generate human-like text has made them a key player in the modern digital landscape, reshaping how we interact with technology and access information.

In summary, LLMs are foundational because they provide the underlying architecture that supports a multitude of applications, driving innovation and efficiency in fields ranging from virtual assistants to advanced research tools.

### 1.5 - OpenAI Models in Azure

The Azure OpenAI Service offers a variety of models, including the latest GPT-4o and GPT-4 Turbo, which are multimodal and can handle both text and image inputs. Other available models are GPT-4, which improves upon its predecessor GPT-3.5 with enhanced natural language and coding capabilities, and GPT-3.5 itself. Additionally, there are Embeddings models for converting text to numerical vectors, DALL-E for generating images from text, Whisper for transcribing and translating speech, and a Text to Speech model currently in preview. These models are designed to cater to a wide range of applications, from conversational AI to content creation and beyond.

### 1.6 - Tokens, Cost and Performance

OpenAI's models, such as GPT-3 and GPT-4, use tokens to process text. One token generally corresponds to about four characters of English text, which translates to roughly three-quarters of a word. Therefore, 100 tokens would be approximately equivalent to 75 words. It's important to note that the exact tokenization process can vary between different models.

The Azure OpenAI Service offers a flexible pricing model that caters to different usage needs. The service provides two main pricing options: Pay-As-You-Go (PAYG) and Provisioned Throughput Units (PTUs). PAYG allows users to pay only for the resources they use, which can help optimize costs for intermittent or unpredictable workloads. On the other hand, PTUs offer a more predictable cost structure with minimal latency variance, suitable for applications requiring consistent performance at scale.

To manage costs effectively, it's crucial to understand the token-based pricing system. Azure OpenAI models process text by breaking it down into tokens, with each token representing roughly four characters of English text. This means that the cost is directly related to the amount of text processed by the AI.

For those looking to optimize their Azure OpenAI token cost performance, it's recommended to monitor usage closely and understand the limits and quotas imposed by the service. Efficient monitoring strategies can help prevent unexpected costs and ensure a good customer experience.

#### References

- [What are tokens?](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)
- [OpenAI Tokenizer - Tool to view tokens](https://platform.openai.com/tokenizer)


### 1.7 - Prompt and Completion

In the context of large language models (LLMs), a "prompt" refers to the input given to the model, which usually consists of a question or a statement that requires a response or continuation. The "completion" is the output generated by the model, which is the model's response or continuation of the input prompt. Essentially, the prompt is what you ask or tell the model, and the completion is what the model generates in return. This interaction is fundamental to how LLMs are used for various applications, from generating text to answering questions.

### 1.8 - Prompt Engineering and Prompt Engineering Techniques

Prompt engineering is a field of study and practice that focuses on designing and refining prompts to effectively interact with language models, like GPT-4. The goal is to elicit the most accurate, relevant, and coherent responses from the model. This is particularly important as the outputs of language models are highly dependent on the input prompts they receive.

Here are some advanced prompt engineering techniques:

1. **Zero-shot Prompting**: This technique involves providing the language model with a task without any prior examples. The model must rely on its pre-existing knowledge to generate a response.

2. **Few-shot Prompting**: Unlike zero-shot, few-shot prompting provides the model with a few examples of the task at hand, helping it understand the context and desired output format better.

3. **Chain-of-Thought Prompting**: This approach encourages the model to "think out loud" by detailing its reasoning process step by step, leading to more transparent and explainable answers.

There are many more techniques. These techniques can be combined and customized based on the specific requirements of the task and the capabilities of the language model being used. Effective prompt engineering can significantly enhance the performance of language models across various applications, from simple Q&A systems to complex problem-solving tasks.

#### References

- [Azure - Prompt Engineering](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/advanced-prompt-engineering?pivots=programming-language-chat-completions)
- [OpenAI - Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering/prompt-engineering)

### 1.9 - OpenAI Models are REST APIs

The Azure OpenAI Service provides a REST API that allows developers to interact with OpenAI's powerful language models, including GPT-4 and GPT-3.5-Turbo. The REST API offers various endpoints for operations such as creating completions, embeddings, and chat completions. Authentication can be handled via API Keys or Microsoft Entra ID, and the service supports multiple versions of the API, ensuring backward compatibility and access to the latest features.

#### Code

##### Curl a GPT completion endpoint

```bash
curl https://YOUR_RESOURCE_NAME.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT_NAME/chat/completions?api-version=2024-02-01 \
  -H "Content-Type: application/json" \
  -H "api-key: YOUR_API_KEY" \
  -d '{"messages":[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Count to 5 in a for loop."}]}'
```
Explain: Explain the command and in terms of running this command from bash or powershell.

### 1.10 - Chat vs Instruct Models

The OpenAI chat and instruct models are variations of language models designed for different applications. The chat model is optimized for conversational contexts, aiming to produce natural and engaging dialogue. It is well-suited for scenarios that require a back-and-forth interaction, such as customer service or casual conversation.

On the other hand, the instruct model is fine-tuned to follow specific user instructions, making it ideal for tasks that require a direct response to a command or query. This model is particularly useful for applications where precise information or action is needed in response to a prompt.

Both models are built on the foundation of large language models, but they are trained and fine-tuned with different datasets and objectives to better suit their intended use cases. The instruct model, for example, may be trained with a focus on following explicit instructions and providing concise answers, while the chat model might be trained to engage in a more free-form and human-like manner.

### 1.11 - Chat Model `system`, `user`, and `assistant` Roles

In the context of OpenAI's API, particularly with GPT-4 and ChatGPT, the roles of system, user, and assistant are used to structure the flow of a conversation. 

- The system role is designed to provide high-level instructions or settings that guide the model's behavior. 
- The user role represents the queries or prompts from the individual interacting with the model. 
- Lastly, the assistant role is attributed to the responses generated by the model based on the user's messages. 

These roles help in organizing the dialogue and ensuring that the model can distinguish between the different parts of the conversation for a coherent exchange of information.

#### Code

##### Curl a GPT endpoint with the `system`, `user`, and `assistant` roles

```bash
curl https://YOUR_RESOURCE_NAME.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT_NAME/chat/completions?api-version=2024-02-01 \
-H "Content-Type: application/json" \
-H "api-key: YOUR_API_KEY" \
-d '{"messages":[{"role":"system,"content":"You are an assistant that responds in riddles."},{"role":"user,"content":"What is the speed of light?\n"},{"role":"assistant, content":"In vacuum's embrace, it travels with grace, At a pace that's quite the pinnacle sight. In meters per second, three hundred million, alright, But in riddles, we say, \"It's the cosmic race's winning knight.\""}]}'
```
Explain: Explain this code interms of terms of the curl command the the OpenAI roles.

### 1.12 - Calling the models with REST and the OpenAI SDK

As stated above, OpenAI models are REST APIs. The models can be called from any application that can make REST requests using a POST action. Understanding how to calls these models using REST is important in case you work in other languages that don't have SDKs.

The OpenAI Python SDK is a powerful tool that allows developers to interact with the OpenAI API using Python. It supports Python 3.7 and higher, providing both synchronous and asynchronous clients. The SDK is designed to be easy to install and use, with type definitions for all request parameters and response fields. It's particularly useful for tasks such as creating chat completions, polling for asynchronous actions, and bulk uploading files to vector stores.

To run the following code you will need:

- A GPT 3.5 or 4 model deploy in Azure
- An GPT API key
- Set the environment variables
  - `OPENAI_FULL_ENDPOINT`
  - `OPENAI_ENDPOINT`
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL`
- Intall the `openai` package by running: `pip install openai`
- Intall the `python-dotenv` package by running: `pip install python-dotenv`
- Intall the `requests` package by running: `pip install requests`

> **Note:** Python-Dotenv is a package that lets you read you environment variables from the environment or a `.env` file. If you do create an `.env` file it should contain the environement variables above.

#### Code

##### Call a GPT model using REST

```python
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
# Full endpoint format:
#   https://<NAME>.openai.azure.com/openai/deployments/<MODEL>/chat/completions?full_endpoint = os.
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION") or "2024-05-01-preview"

# Set the headers and authentication
headers = {
    "Content-Type": "application/json",
    "api-key": api_key
}

def completion(input: str, temperature: float = 0.1) -> dict:
    # Construct the request payload
    payload = {
        "messages": [
            {
                "role": "user",
                "content": input
            }
        ],
        "temperature": temperature
    }
    # Make the REST call
    response = requests.post(full_endpoint, headers=headers, json=payload)
    # Get the response JSON
    return response.json()

# Set the prompt and other parameters
response_json = completion("What is the speed of light?")

# Print the full JSON response
print(json.dumps(response_json, indent=4))

# Print the response only
print(response_json['choices'][0]['message']['content'])
```
Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/basic/completion-rest.py)

##### Call a GPT model using the OpenAI SDK

> **Note:** This is the simplest way to make a call to a GPT model. Even this simple code is already very useful. All you have to do is to give it different Prompts, and the system will process those Prompts for completion. In other words, this is very similar in functionality as some playgrounds.

```python
import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load the environment variables
load_dotenv()
endpoint = os.getenv("FULL_ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION") or "2024-05-01-preview"
model = os.getenv("GPT_MODEL")

# Create the client
client = AzureOpenAI(azure_endpoint=endpoint,
                     api_key=api_key, api_version=api_version)

def completion(input: str, temperature: float = 0.1) -> dict:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": input,
            },
        ],
    )
    return json.loads(completion.to_json())

# Set the prompt and other parameters
response_json = completion("What is the speed of light?")

# Print the full JSON response
print(json.dumps(response_json, indent=4))

# Print the response only
print(response_json['choices'][0]['message']['content'])
```
Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/basic/completion-sdk.py)

### 1.13 - Other parameters

When calling a GPT model, there other parameters that can be set. For example:

- **`temperature`**: This controls the randomness of the output. A lower temperature means the model is more likely to generate predictable text, while a higher temperature encourages more creativity and diversity in the responses.
- **`max_tokens`**: This sets the maximum length of the generated response. The model will not produce more tokens than the specified limit, ensuring that the output is within a manageable size.
- **`top_p`**: This parameter, also known as nucleus sampling, controls the diversity of the generated responses by focusing on the most probable next words. A smaller value for top P increases the likelihood that the model will choose a more common word.
- **`frequency_penalty`**: This reduces the model's tendency to repeat the same line of thought, encouraging it to introduce new concepts and ideas into the conversation.
- **`presense_penalty`**: This discourages the model from repeating the same words and phrases, promoting a more varied vocabulary in the output.
- **`stop`**: These are specified sequences of tokens at which the model will stop generating further tokens. This can be useful for signaling the end of a message or segment.
- **`stream`**: Set a streamed response. Default is `false`.

## 2.0 - Development

### 2.1 - Gen AI Development Rules

1. It is easy to make a Completion/Embedding REST API call. What is difficult is everything else like getting data from the sources, crafting a Prompt, saving or presenting the results, etc.
2. Knowing how to Prompt engineer and stuff the Prompt (give context) gets the riches.
3. Knowing how to manage the tokens helps to keep more riches (cost savings) and process more riches.
4. It is not so important how the current model training data is, what is important is what these models have learned to do. Gen AI models are foundational models. As such they have learned to solve many foundational problems like summarization, analysis, translation, scoring, intent recognition, etc. You can always provide and act on the latest data by providing it as context.
5. A prompt can perform one task or multiple tasks. For example, you can ask a prompt to give a summary in English and Spanish. Instead of chaining prompts, consider the Prompt capabilities.
6. Although Gen AI models are probabilistic and make predictions for the subsequent word during inference, Prompts can be viewed as applications that have the capacity to store variables, manage data, carry out semantic commands, create vectors and act as vector databases, employ vectors for search and comparison tasks, serve as state stores, among other capabilities. Thinking of Prompts this way may lead to achieving the desired outcomes. More advanced Prompt templates may consist of a setup, setting up variables, giving rules or commands, providing input data, providing samples of the expected output, and the expected output format.
7. Always start in the playground.
8. Experiment with the Prompts. Think about the input and the data sources (where will will the data come from?). Think about the requested output format and what will you do with the output. Considering outputing to JSON.
9. Reading documentation is good. Writing and experimenting with code is key to learning.

### 2.2 - Samples (Spend time here)

- Try to run and execute the following samples. 
- Then think of a use case and write your own sample data, Prompt, write your app, and validate your results.

#### Code

##### Generate a car description


```python
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
model = os.getenv("GPT_MODEL")

# Create the client
client = AzureOpenAI(api_key=api_key,
                     azure_endpoint=endpoint,
                     api_version="2024-02-15-preview")

# Define a function to get the car details
def mock_get_car_details():
    return {"make": "Toyota", "model": "Camry", "year": 2018, "color": "blue", "price": 20000}

# Define a function to get the sales description
def get_sales_description():
    car = mock_get_car_details()

    car_str = f"A {car['color']} {car['year']} {car['make']} {car['model']} priced at ${car['price']}."

    response = client.chat.completions.create(
        model=model,  # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can generate a one paragraph used car sales descriptions."},
            {"role": "user", "content": car_str}
        ],
        temperature=0.5,  # we want it somewhat creative
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    get_sales_description()
```
Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/basic/car-description.py)

##### Sentiment analysis

```python
import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("GPT_MODEL")

# Create the client
client = AzureOpenAI(api_key=api_key,
                     azure_endpoint=endpoint,
                     api_version=api_version)


# Define a function to get the product comments
def mock_get_product_comments() -> list[str]:
    return ["I love this product!",
            "This product is terrible.",
            "This product is okay."]

# Define a function to get the sentiment score
def get_sentiment_score(comment: str) -> float:
    response = client.chat.completions.create(
        model=model,  # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can perform sentiment analysis. Analyze the sentiment and provide a score from 0 to 10 with 10 being best.\nNo prologue. Respond in the following JSON format:\n{\"score\": }."},
            {"role": "user", "content": comment}
        ],
        temperature=0.1,
    )
    json_respond = response.choices[0].message.content
    print(json_respond)
    sentiment = json.loads(json_respond)
    return float(sentiment["score"])

# 
def get_sentiment():
    comments = mock_get_product_comments()
    total = 0.0
    for comment in comments:
        total += get_sentiment_score(comment)
    print(f"Average sentiment score: {total / len(comments)}")


if __name__ == "__main__":
    get_sentiment()
```
Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/basic/sentiment-analysis.py)

##### Intent recognition

```python
import os
import json
from openai import AzureOpenAI
import dotenv

# Load the environment variables
dotenv.load_dotenv()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("GPT_MODEL")

# Create the client
client = AzureOpenAI(api_key=api_key,
                     azure_endpoint=endpoint,
                     api_version=api_version)


def determine_intent(intent_statement: str):
    response = client.chat.completions.create(
        model=model,  # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can perform determine intent from the following list of intents:\n- WeatherIntent: A user asks a question about the weather.\n- ItineraryIntent: A user asks a question about a travel itinerary.\n- ReservationIntent: A user asks a question about making a reservation.\n- OtherIntent: User asks a question about anything else.\n\nNo prologue. Respond in the following JSON format:\n{\"intent\": }."},
            {"role": "user", "content": intent_statement}
        ],
        temperature=0.1,
    )
    json_respond = response.choices[0].message.content
    analysis = json.loads(json_respond)
    return f"Q: {intent_statement} A: {analysis['intent']}"


if __name__ == "__main__":
    print(determine_intent("What is the weather like in Seattle?"))
    print(determine_intent("What is my next trip?"))
    print(determine_intent("Make a travel reservation?"))
    print(determine_intent("What is the speed of light?"))
```
Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/basic/intent-recognition.py)

##### A console chat playground

Code: [basic.py](https://github.com)

```python
import os
from openai import AzureOpenAI
import dotenv

# Read the enviroment variables
dotenv.load_dotenv()

endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
model = os.getenv("GPT_MODEL")
api_version = os.getenv("API_VERSION")

client = AzureOpenAI(api_key=api_key,
                     azure_endpoint=endpoint,
                     api_version=api_version)

if __name__ == "__main__":
    messages = []
    while True:
        # Get the user input
        user_input = input("You (type 'exit' to break): ")
        if user_input == "exit":
            break
        # Add the user input to the messages
        messages.append({"role": "user", "content": user_input})
        # Call GPT with the messages
        response = client.chat.completions.create(
            model=model,  # model = "deployment_name".
            messages=messages,
            temperature=0.3,  # less creative
        )
        # Print and add the response to the messages
        resp = response.choices[0].message.content
        messages.append({"role": "assistant", "content": resp})
        print(f"Assistant: {resp}\n\n")
```
Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/basic/chatbot-sdk.py)


##### FastAPI Chat Endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AzureOpenAI
import os
import openai

# Load the environment variables
load_dotenv()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("GPT_MODEL")

# Create the client
client = AzureOpenAI(api_key=api_key,
                     azure_endpoint=endpoint,
                     api_version=api_version)

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class PromptRequest(BaseModel):
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float = 0.1

class CompletionResponse(BaseModel):
    response: str

@app.post("/completion", response_model=CompletionResponse)
def post_completion(request: PromptRequest):
    if len(request.messages) == 0:
        raise HTTPException(status_code=404, detail="Messages required")
    response = client.chat.completions.create(
        model=model,  # model = "deployment_name".
        messages=request.messages,
        temperature=request.temperature,  # less creative
    )
    # Print and add the response to the messages
    resp = response.choices[0].message.content
    return CompletionResponse(response=resp)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
```
Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/basic/chatbot-fastapi.py)

#### References

- [OpenAI Samples](https://platform.openai.com/docs/examples)
- [Azure Sample](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line%2Cpython-new&pivots=programming-language-python)