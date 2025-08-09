# Generative AI App Development - Foundational Concepts

## 1.0 - Foundational Concepts

### 1.1 - Objective

This document provides a comprehensive introduction to foundational concepts and practical techniques for developing applications with Generative AI, focusing on Large Language Models (LLMs) such as those available through Azure OpenAI. It covers essential topics including what LLMs are, why they are considered foundational models, tokenization and cost management, prompt and context engineering, REST API usage, and the differences between chat, reasoning, and instruct models. The guide also includes hands-on code samples for making API calls, prompt engineering, sentiment analysis, intent recognition, and building chatbots using Python and FastAPI. Additionally, it introduces advanced features like function calling, offering both conceptual explanations and practical examples to help developers effectively leverage generative AI in real-world applications.

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
ENDPOINT=https://YOUR_RESOURCE_NAME.openai.azure.com/
API_KEY=<KEY>
API_VERSION=2025-01-01
GPT_MODEL=gpt-4o
```

- Create and Python environment and install the following Python packages:
  - `pip install openai python-dotenv httpx azure-identity fastapi uvicorn[standard]`

> **Note:** To get the full OpenAI endpoint, in AI Foundry click on the model, and copy the full endpoint which has the following format: `https://YOUR_RESOURCE_NAME.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT_NAME/chat/completions?api-version=2024-02-01`

#### References

- [Setting-up a Python environment with WSL and VS Code](https://genaitutor.am2703.com/?content=pyenv-wsl-vscode)

### 1.3 - What is a Large Language Model (LLM)?

A Large Language Model (LLM) is a type of artificial intelligence system built using deep neural networks with billions or trillions of parameters that has been trained on vast amounts of text data from the internet, books, articles, and other written sources to understand and generate human-like text. These models use transformer architecture to process and understand the relationships between words, phrases, and concepts, enabling them to comprehend context, follow instructions, engage in conversations, write code, solve problems, and perform a wide variety of language-related tasks.

LLMs like OpenAI's GPT series are trained through unsupervised learning to predict the next word in a sequence, which teaches them grammar, facts, reasoning patterns, and even some level of common sense, making them capable of producing coherent, contextually appropriate responses across diverse topics and applications. The "large" aspect refers both to the massive scale of training data (often terabytes of text) and the enormous number of parameters that allow these models to capture complex patterns in human language and knowledge.

### 1.4 - Why is an LLM a Foundational Model?

An LLM is considered a foundational model because it serves as a versatile base that can be adapted and fine-tuned for a wide variety of downstream tasks without requiring task-specific training from scratch. These models are trained on massive, diverse datasets that give them broad knowledge and capabilities across multiple domains - from natural language understanding and generation to reasoning, coding, and problem-solving.

Rather than being designed for a single purpose, foundational models like GPT-4 provide a general-purpose intelligence that can be specialized through techniques like prompt engineering, fine-tuning, or retrieval-augmented generation (RAG) to excel at specific applications such as customer service, content creation, code generation, or domain-specific question answering.

This foundational nature makes them incredibly cost-effective and powerful, as one pre-trained model can be the basis for hundreds of different AI applications, democratizing access to advanced AI capabilities across industries and use cases.

### 1.5 - Inference

Inference in the context of large language models refers to the process of using a pre-trained model to generate predictions, responses, or outputs based on new input data that the model hasn't seen during training. During inference, the model applies the patterns, knowledge, and relationships it learned during training to process your prompt or query and produce a relevant response, whether that's answering a question, generating text, writing code, or performing any other task within its capabilities.

This is the "thinking" phase where the model uses its billions of parameters to calculate probabilities and select the most appropriate tokens to generate, transforming your input into meaningful output. Inference is distinct from training (where the model learns from data) and represents the operational phase where the model is actively being used to provide value, making it the core process that powers all interactions with AI services like those available through Azure OpenAI, and it's during inference that costs are incurred based on token consumption and computational resources used.

### 1.6 - OpenAI models in Azure AI Foundry

The Azure OpenAI Service offers a variety of models, including the latest GPT-4o and GPT-4.1, which are multi-modal and can handle both text and image inputs. Additionally, there are embeddings models for converting text to numerical vectors, DALL-E 3 for generating images from text, Whisper for transcribing and translating speech, and a Text to Speech model currently in preview. These models are designed to cater to a wide range of applications, from conversational AI to content creation and beyond.

### 1.7 - Tokens, cost and performance

Tokens are the fundamental units that language models use to process and understand text, representing pieces of words, whole words, or even punctuation marks that the model breaks text into during analysis. In OpenAI models deployed on Azure, a token roughly corresponds to 3-4 characters in English text, meaning that a typical word might be 1-2 tokens, while longer or less common words could be broken into multiple tokens. For example, "hello" might be one token, while "understanding" could be split into "under" and "standing" as separate tokens. The tokenization process varies by language, with some languages like Chinese or Arabic requiring more tokens per character than English, and technical terms, code, or special characters often requiring additional tokens to represent properly.

The Azure OpenAI Service offers a flexible pricing model that caters to different usage needs. The service provides two main pricing options: Pay-As-You-Go (PAYG) and Provisioned Throughput Units (PTUs). PAYG allows users to pay only for the resources they use, which can help optimize costs for intermittent or unpredictable workloads. On the other hand, PTUs offer a more predictable cost structure with minimal latency variance, suitable for applications requiring consistent performance at scale.

From a performance perspective, models have token limits (context windows) that determine how much text they can process at once - exceeding these limits requires truncating conversation history or splitting requests, which can impact response quality and coherence. Additionally, more tokens generally mean longer processing times and higher latency, so optimizing prompt length, managing conversation history efficiently, and designing concise interactions not only reduces costs but also improves user experience through faster response times and better resource utilization in Azure deployments.

#### References

- [What are tokens?](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)
- [OpenAI Tokenizer - Tool to view tokens](https://platform.openai.com/tokenizer)

### 1.8 - Token limits, throttling, concurrent jobs and retry logic

When deploying an OpenAI model in Azure, administrators must configure a tokens-per-minute (TPM) limit. Even with high-throughput models like GPT-4.1 (which can handle up to 1 million TPM), developers need to carefully design their GenAI applications to avoid hitting these limits and ensure reliable performance. Key considerations include:

- **Implement retry logic** for handling rate limits (HTTP 429) and other transient errors.
- **Distribute load** across multiple OpenAI endpoints if necessary.
- **Manage concurrency** to prevent exceeding the TPM limit with parallel tasks.
- **Optimize prompt size** by managing conversation history and avoiding unnecessarily large prompts.
- **Use prompt compression and caching** to reduce token usage and improve efficiency.
- **Batch tasks** into a single prompt when possible.
  - *Tip:* A single prompt can perform multiple actions (e.g., summarize content and provide translations in English, Spanish, and French).
- **Monitor application performance** and define service level objectives (SLOs).
- **Design user experiences** that keep users engaged while waiting for completions.
- **Follow responsible AI best practices** to ensure ethical and safe use of the models.

#### References

- [Inference visualization](https://bbycroft.net/llm)

### 1.9 - Prompt and Completion

A **prompt** is the input text or instruction that you provide to a large language model to initiate a conversation or request a specific task, serving as the starting point for the model's response generation. Prompts can range from simple questions like "What is the capital of France?" to complex instructions that include context, examples, formatting requirements, and specific guidelines for how the model should respond. The quality and structure of your prompt significantly influences the model's output, making prompt engineering a crucial skill for getting optimal results from AI systems.

A **completion** is the model's generated response to your prompt, representing the text that the model predicts should logically follow based on the patterns it learned during training. The completion process involves the model analyzing your prompt, understanding the context and intent, and then generating tokens one by one until it reaches a natural stopping point, produces a specified number of tokens, or hits a defined completion criteria.

When you make a request to OpenAI's API through Azure or directly, the response includes much more than just the completion text itself. The API returns structured metadata including **usage statistics** that show the number of prompt tokens, completion tokens, and total tokens consumed (essential for cost tracking), a **finish_reason** that indicates why the generation stopped (such as reaching a natural end, hitting token limits, or being filtered by content policies), **model information** specifying which exact model version was used, and often **additional fields** like response timestamps, request IDs for debugging, and confidence scores. Some responses may also include **logprobs** (log probabilities) that show the model's confidence in each token choice, **alternative completions** when multiple outputs are requested, and **content filter results** that indicate if any safety mechanisms were triggered, providing developers with comprehensive information to monitor performance, debug issues, manage costs, and ensure responsible AI usage in their applications.

Sample completion:

```json
{
  "id": "chatcmpl-8VwKjX9Y2L4nQ6mR5tP3sU7vW1xZ",
  "object": "chat.completion",
  "created": 1701234567,
  "model": "gpt-4o-2024-08-06",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris. It's the largest city in France and serves as the country's political, economic, and cultural center.",
        "refusal": null
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 28,
    "total_tokens": 40,
    "prompt_tokens_details": {
      "cached_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0
    }
  },
  "system_fingerprint": "fp_2f57f81c65"
}
```

### 1.10 - Prompt Engineering and Prompt Engineering techniques

Prompt engineering is a field of study and practice that focuses on designing and refining prompts to effectively interact with language models, like GPT-4. The goal is to elicit the most accurate, relevant, and coherent responses from the model. This is particularly important as the outputs of language models are highly dependent on the input prompts they receive. Some techniques include:

1. **Zero-shot Prompting**: This technique involves providing the language model with a task without any prior examples. The model must rely on its pre-existing knowledge to generate a response.
2. **Few-shot Prompting**: Unlike zero-shot, few-shot prompting provides the model with a few examples of the task at hand, helping it understand the context and desired output format better.
3. **Chain-of-Thought Prompting**: This approach encourages the model to "think out loud" by detailing its reasoning process step by step, leading to more transparent and explainable answers.

There are many techniques. These techniques can be combined and customized based on the specific requirements of the task and the capabilities of the language model being used. If you are developer, you may think of prompts as program that has inputs, carries out semantic instructions and rules, and outputs the results in the requested format with samples of the outputs if necessary. Thinking of prompts this way may help you craft more powerful prompts for productivity and for your applications.

#### References

- [Prompting techniques](https://www.promptingguide.ai/techniques)
- [Azure - Prompt Engineering](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/advanced-prompt-engineering?pivots=programming-language-chat-completions)
- [OpenAI - Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering/prompt-engineering)

### 1.11 - Context Engineering vs Prompt Engineering

Prompt Engineering is the art of crafting precise, effective inputs to guide AI behavior. It’s about asking the right question in the right way. Think of it as giving clear instructions to a very smart assistant. Context Engineering, on the other hand, is about designing the environment in which the AI operates. This includes:

- Structuring memory and retrieval systems
- Managing long-term vs. short-term context
- Curating relevant data and metadata
- Orchestrating multi-agent collaboration

While prompt engineering focuses on what you say to the model, context engineering focuses on what the model knows when it responds.

Why does this matter?

- Prompt engineering is great for quick wins and one-off tasks.
- Context engineering is essential for building scalable, consistent, and intelligent systems—especially in enterprise and multi-agent environments.

### 1.12 - OpenAI models are exposed as REST APIs

Azure OpenAI Service exposes its language models, such as GPT-4o and GPT-4.1, through a REST API. This API enables developers to perform tasks like text completions and embeddings by sending HTTP requests to specific endpoints. Authentication is supported via API keys or Microsoft Entra ID. Multiple API versions are available, allowing you to choose the latest features or maintain compatibility with existing applications.

#### References

- [Azure API Reference](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference)

#### Code

##### Curl a GPT completion endpoint

```bash
curl https://YOUR_RESOURCE_NAME.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT_NAME/chat/completions?api-version=2025-01-01-preview \
  -H "Content-Type: application/json" \
  -H "api-key: YOUR_API_KEY" \
  -d '{"messages":[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Count to 5 in a for loop."}]}'
```

Explain: Explain the command and in terms of running this command from bash or powershell.

#### POST using the HTTP REST Client in Visual Studio Code

```text
POST https://YOUR_RESOURCE_NAME.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT_NAME/chat/completions?api-version=2025-01-01-preview
content-type: application-json
api-key: <KEY>

{
    "model":"gpt-4o",
    "temperature": 0.1,
    "messages":[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
}
```

### 1.13 - Model types

Chat models are designed to generate natural, engaging conversations and are ideal for interactive scenarios like customer support or chatbots, where maintaining context across multiple turns is important.

Reasoning models, such as o1 and GPT-5, are specialized or fine-tuned to handle tasks that require logical thinking, inference, and multi-step problem-solving. These models often provide both their reasoning process and the final answer in their responses.

There are also Instruct models. These models on the other hand, are optimized for direct question-and-answer tasks rather than ongoing conversations. They excel when context does not need to be preserved between interactions.

### 1.14 - Chat Model `system`, `user`, and `assistant` Roles

In OpenAI's Chat API (such as GPT-4o), each message in a conversation is assigned a `role` to clarify its purpose:

- **`system`**: Sets the overall behavior or persona of the assistant. Use this to provide instructions, context, or constraints that guide how the model should respond.
- **`user`**: Represents input from the end user—questions, commands, or prompts that drive the conversation.
- **`assistant`**: Contains the model's responses to the user's messages.

Using these roles helps structure the conversation, allowing the model to distinguish between instructions, user input, and its own replies for more coherent and context-aware interactions.

> **Note:** The `system` role is optional but highly recommended for consistent and predictable model behavior. It is especially important in agent-based systems, where each agent may have a distinct function or responsibility.

### 1.15 - Managing the chat history

Managing chat history is crucial when building applications with large language models, especially in chat-based scenarios. The chat history—comprising all previous messages exchanged between the user and the assistant—directly impacts cost, capacity, throughput, and performance:

- **Cost**: Each token in the chat history counts toward your API usage and billing. Longer histories mean more tokens per request, increasing operational costs.
- **Capacity**: Models have a fixed context window (token limit). If the chat history grows too large, you may exceed this limit, forcing you to truncate or omit important context, which can degrade the quality of responses.
- **Throughput**: Sending large chat histories increases payload size and processing time, reducing the number of requests your application can handle per minute and potentially causing throttling.
- **Performance**: Excessive or irrelevant history can confuse the model, leading to less accurate or coherent responses. Efficiently managing history ensures the model focuses on the most relevant context.

Role and content diagram:

```text
[system]    You are a helpful assistant.
[user]      What are some compute services in Azure?
[assistant] Some compute services in Azure include: Virtual Machines, App Service, AKS
[user]      What are some more?
[assistant] Other services include: ACI, ACA, Azure Functions
```

Some techniques for managing chat history include:

- **Truncate old messages**: Keep only the most recent exchanges or those relevant to the current topic. Discard or summarize older messages to stay within token limits.
- **Summarize history**: Use the model to generate concise summaries of earlier conversation turns, replacing long message chains with a brief recap.
- **Selective inclusion**: Only include messages that are necessary for the current task or question, omitting unrelated or redundant exchanges.
- **Windowed context**: Implement a sliding window that maintains a fixed number of recent messages, ensuring the prompt remains within the model’s context window.
- **External storage**: Store full chat histories outside the prompt (e.g., in a database) and retrieve only the relevant context for each request.
- **Prompt compression**: Use techniques to compress or encode history, such as removing filler words or using structured formats.

By thoughtfully managing chat history, you can optimize costs, maintain high throughput, and ensure your application delivers fast, accurate, and contextually relevant responses.

#### Code

##### Curl a GPT endpoint with the `system`, `user`, and `assistant` roles

```bash
curl https://YOUR_RESOURCE_NAME.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT_NAME/chat/completions?api-version=2024-02-01 \
-H "Content-Type: application/json" \
-H "api-key: YOUR_API_KEY" \
-d '{"messages":[{"role":"system,"content":"You are an assistant that responds in riddles."},{"role":"user,"content":"What is the speed of light?\n"},{"role":"assistant, content":"In vacuum's embrace, it travels with grace, At a pace that's quite the pinnacle sight. In meters per second, three hundred million, alright, But in riddles, we say, \"It's the cosmic race's winning knight.\""}]}'
```

Explain: Explain this code in terms of the curl command the the OpenAI roles.

### 1.16 - Other Chat model parameters

When calling a GPT model, there other parameters that can be set, including:

- **`temperature`**: This controls the randomness of the output. A lower temperature means the model is more likely to generate predictable text, while a higher temperature encourages more creativity and diversity in the responses.
- **`max_tokens`**: This sets the maximum length of the generated response. The model will not produce more tokens than the specified limit, ensuring that the output is within a manageable size.
- **`top_p`**: This parameter, also known as nucleus sampling, controls the diversity of the generated responses by focusing on the most probable next words. A smaller value for top P increases the likelihood that the model will choose a more common word.
- **`frequency_penalty`**: This reduces the model's tendency to repeat the same line of thought, encouraging it to introduce new concepts and ideas into the conversation.
- **`presense_penalty`**: This discourages the model from repeating the same words and phrases, promoting a more varied vocabulary in the output.
- **`stop`**: These are specified sequences of tokens at which the model will stop generating further tokens. This can be useful for signaling the end of a message or segment.
- **`stream`**: Set a streamed response. Default is `false`.

> **Note:** setting the max tokens may improve your models ability to handle multiple requests. The model uses this information to optimize the overall capacity. If you know the expected token size, it may be a very good idea to set it.

### 1.17 - Calling the models with REST and the OpenAI SDK

OpenAI models are accessed via REST APIs, meaning you can call them from any application capable of making HTTP POST requests. This is especially useful when working in languages or environments that do not have a dedicated SDK.

For Python developers, the OpenAI Python SDK offers a convenient and robust way to interact with the API. Supporting Python 3.9 and above, the SDK provides both synchronous and asynchronous clients, complete with type definitions for all request parameters and response fields. It simplifies common tasks such as creating chat completions, handling asynchronous operations, and managing bulk file uploads for vector stores.

#### Code

##### Call a GPT model using REST

```python
import httpx
import os
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()
full_endpoint = os.getenv("FULL_ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")

headers = {"Content-Type": "application/json", "api-key": api_key}


async def completion(input: str, temperature: float = 0.1) -> dict:
    payload = {
        "messages": [{"role": "user", "content": input}],
        "temperature": temperature,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(full_endpoint, headers=headers, json=payload)
        return response.json()


async def main():
    response_json = await completion("What is the speed of light?")
    print(json.dumps(response_json, indent=4))
    print(response_json["choices"][0]["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())
```

Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/basic/completion-rest.py)

##### Call a GPT model using the OpenAI SDK

```python
import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

# Load the environment variables
load_dotenv()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("GPT_MODEL")

# Create the async client
client = AsyncAzureOpenAI(
    azure_endpoint=endpoint, api_key=api_key, api_version=api_version
)


# Make a completion request
async def completion(input: str, temperature: float = 0.1) -> tuple[dict, str]:
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": input,
            },
        ],
    )
    return (json.loads(completion.to_json()), completion.choices[0].message.content)


# Set the prompt and other parameters
async def main():
    full, response = await completion("What is the speed of light?")
    print(json.dumps(full, indent=4))
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
```

Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/basic/completion-sdk.py)

## 2.0 - Development

### 2.1 - Gen AI Development Rules

1. Always start in the playground.
2. It is easy to make a Completion/Embedding REST API call. What is difficult is everything else like getting data from the sources, crafting a prompt, saving or presenting the results, etc.
3. Knowing how to Prompt engineer and stuff the Prompt (give context) gets the riches.
4. Knowing how to manage the tokens helps to keep more riches (cost savings) and process more riches.
5. It is not so important how the current model training data is, what is important is what these models have learned to do. Gen AI models are foundational models. As such they have learned to solve many foundational problems like summarization, analysis, translation, scoring, intent recognition, etc. You can always provide and act on the latest data by providing it as context.
6. A prompt can perform one task or multiple tasks. For example, you can ask a prompt to give a summary in English and Spanish. Instead of chaining prompts, consider the Prompt capabilities.
7. Although GenAI models are probabilistic engines that make predictions for the subsequent word during inference, prompts can be viewed as applications that have the capacity to store variables, manage data, carry out semantic commands, create vectors and act as vector databases, employ vectors for search and comparison tasks, serve as state stores, among other capabilities. Thinking of Prompts this way may lead to achieving the desired outcomes. More advanced Prompt templates may consist of a setup, setting up variables, giving rules or commands, providing input data, providing samples of the expected output, and the expected output format.
8. Experiment with the Prompts. Think about the input and the data sources (where will will the data come from?). Think about the requested output format and what will you do with the output. Considering output to JSON.
9. Reading documentation is good. Writing and experimenting with code is key to learning.

### 2.2 - Samples (Spend time here)

- Try to run and execute the following samples.
- Then think of a use case and write your own sample data, Prompt, write your app, and validate your results.

#### Code

##### Technical document author

```python
import os
import asyncio
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
model = os.getenv("GPT_MODEL")

# Create the async client
client = AsyncAzureOpenAI(
    api_key=api_key, azure_endpoint=endpoint, api_version="2024-02-15-preview"
)


async def generate_documents():
    response = await client.chat.completions.create(
        model=model,  # model = "deployment_name".
        messages=[
            {
                "role": "system",
                "content": "You are a technical document writer. The user will provide a topic, and you will write a full technical document.",
            },
            {"role": "user", "content": "Prompt engineering"},
        ],
        temperature=0.1,  # we want it somewhat creative
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(generate_documents())
```

Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/basic/car-description.py)

##### Risk scoring

```python
import os
import json
import asyncio
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("GPT_MODEL")

# Create the async client
client = AsyncAzureOpenAI(
    api_key=api_key, azure_endpoint=endpoint, api_version=api_version
)


def get_mock_document() -> str:
    return """
    Incident Summary:
    On June 12, 2024, at approximately 09:15 UTC, monitoring systems observed an unusual increase in inbound network traffic targeting public-facing web services. Telemetry data showed a higher than normal volume of requests from a diverse set of IP addresses. The traffic pattern is atypical. Investigation is ongoing to determine the nature and intent of the observed behavior, and precautionary monitoring measures have been implemented.
"""


async def evaluate(content: str) -> float:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": 'You are a risk assessment evaluator. The user will provide a summary of the condition(s) and you need to evaluate the risk level. Provide a score from 0 to 1 with 1 indicating a risky condition.\nNo prologue. Respond in the following JSON format:\n{"score":"","reason":"" }.',
            },
            {"role": "user", "content": content},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(asyncio.run(evaluate(get_mock_document())))
```

Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/basic/risk-scoring.py)

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
client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)


def determine_intent(intent_statement: str):
    response = client.chat.completions.create(
        model=model,  # model = "deployment_name".
        messages=[
            {
                "role": "system",
                "content": 'You are a helpful assistant that can determine the best intent from the following list of intents:\n- WeatherIntent: A user asks a question about the weather.\n- ItineraryIntent: A user asks a question about a travel itinerary.\n- ReservationIntent: A user asks a question about making a reservation.\n- OtherIntent: User asks a question about anything else.\n\nNo prologue. Respond in the following JSON format:\n{"intent": }.',
            },
            {"role": "user", "content": intent_statement},
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
import asyncio
from openai import AsyncAzureOpenAI
import dotenv

# Read the environment variables
dotenv.load_dotenv()

endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("GPT_MODEL")

client = AsyncAzureOpenAI(
    api_key=api_key, azure_endpoint=endpoint, api_version=api_version
)


async def main():
    messages = []
    while True:
        user_input = input("You (type 'exit' to break): ")
        if user_input == "exit":
            break
        messages.append({"role": "user", "content": user_input})
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
        resp = response.choices[0].message.content
        messages.append({"role": "assistant", "content": resp})
        print(f"Assistant: {resp}\n\n")


if __name__ == "__main__":
    asyncio.run(main())
```

Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/basic/chatbot-sdk.py)

##### FastAPI Chat Endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
import os

# Load the environment variables
load_dotenv()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("GPT_MODEL")

# Create the async client
client = AsyncAzureOpenAI(
    api_key=api_key, azure_endpoint=endpoint, api_version=api_version
)

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
async def post_completion(request: PromptRequest):
    if len(request.messages) == 0:
        raise HTTPException(status_code=404, detail="Messages required")
    response = await client.chat.completions.create(
        model=model,
        messages=request.messages,
        temperature=request.temperature,
    )
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

### 2.3 - Function calling

Function Calling is a feature in Azure OpenAI (and OpenAI's GPT models) that enables the model to trigger external functions in response to user input. Rather than only generating text, the model can detect when a request requires structured data or an external operation, and then call a specified function with the necessary arguments. This capability is essential for building agent-based systems, where the model can interact with tools, APIs, or services to complete complex tasks.

#### Code

##### Simple function calling example

```python
import os
import json
import asyncio
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

# Azure OpenAI configuration
load_dotenv()
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("GPT_MODEL")

# Create the async client
client = AsyncAzureOpenAI(
    api_key=api_key, azure_endpoint=endpoint, api_version=api_version
)

# Define the function schema
functions = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. Miami, FL",
                }
            },
            "required": ["location"],
        },
    }
]


# Simulate the function implementation
def get_weather(location):
    return {"location": location, "temperature": "88°F", "condition": "Sunny"}


async def main():
    # Chat completion with function calling
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What's the weather like in New York?"}],
        functions=functions,
        function_call="auto",
    )

    # Check if the model wants to call a function
    if response.choices[0].finish_reason == "function_call":
        function_call = response.choices[0].message.function_call
        function_name = function_call.name
        arguments = json.loads(function_call.arguments)

        # Call the function
        if function_name == "get_weather":
            result = get_weather(**arguments)

            # Send the result back to the model using the same AzureOpenAI client
            follow_up = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "What's the weather like in New York?"},
                    response.choices[0].message,
                    {
                        "role": "function",
                        "name": function_name,
                        "content": json.dumps(result),
                    },
                ],
            )

            print(follow_up.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
```

Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/basic/function-calling.py)
