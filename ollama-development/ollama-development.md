# Application Development Basics with Ollama and SLMs

#### Related

- [OpenAI Development - Foundational](https://genaitutor.am2703.com/?content=genai-intro)
- [OpenAI Development - Intermediate](https://genaitutor.am2703.com/?content=genai-intermediate)

## 1.0 - Application Development Basics with Ollama and SLMs 

### 1.1 - Guide Requirements and recommendations

- A local machine with CPU/RAM
  - A GPU/vRAM is recommended for better performance
- Intermediate Python development knowledge including:
  - Setting up a Python development environment
  - Calling REST endpoints
- Don't skip on reading the `Notes`, they provide valuable information
- Reading is good, actually developing is best

### 1.2 - What is Ollama?

Ollama is a framework for running small language models (LLMs) locally on a user's machine. It was written in Golang and has some C++ components.

It provides a simple API for creating, running, and managing models, along with a library of pre-built models that can be utilized in various applications. The framework is designed to be lightweight and extensible, making it easier to work with LLMs without the need for complex infrastructure. For users looking to install Ollama, there are scripts available that facilitate the installation process on different operating systems. 

Additionally, Ollama supports customization of models through a Modelfile, allowing users to tailor the behavior of the LLMs to their specific needs.

### 1.3 - Installing Ollama

Ollama can be installed in:

- Windows (Preview)
  - [**Download**](https://ollama.com/download/OllamaSetup.exe)
- Linux
  - Run: `curl -fsSL https://ollama.com/install.sh | sh`
- MacOS
  - [**Download**](https://ollama.com/download/Ollama-darwin.zip)


> **Note:** In Windows ollama comes with an update services. In Linux, ollama can be updated by running the installation script again.

### 1.4 - Pulling, running, installing, listing, and deleting models

Once you have installed ollama, you can pull a model by typing:

```bash
ollama pull llava3
```

You can pull and run a model by typing:

```text
ollama run phi3:medium-128k

>>> What is 1+1?
 The sum of 1 and 1 is 2. This is a basic arithmetic operation known as addition, where you combine quantities to find 
their total amount. In this case, combining one unit with another one unit results in two units altogether. Here's the 
calculation for clarity:

1 + 1 = 2

>>> """system:
... You are an assistant that answers user questions in riddles.
...
... user:
... What is the speed of light?
... """
 I travel so fast, yet not a step am I taking,
In vacuum my pace is set at exactly twenty-nine thousand kilometers per second meticulously. What am I?
(Answer: The speed of light)

>>> Send a message (/? for help
```

To list models, type:

```bash
ollama ls
```

To delete a model type:

```bash
ollama rm phi3
```

> **Note:** Ollama offers several version for the same model. For example, there's a phi3, phi3-medium, phi3-128k-medium, etc. There are also different quantization models. Make sure to check to options to find the model that may best fit your requirements.

#### References

- [Ollama quick start documentation](https://github.com/ollama/ollama/blob/main/README.md#quickstart)
- [WSL NVIDIA Driver WSL Installation](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)

### 1.5 - Performance (CPU/RAM vs GPT/vRAM)

You can run ollama with CPU and RAM, but for best performance it is recommended that you have a system with GPU and vRAM. Make sure you install the NVDIA drivers, for example if you are using Windows WSL.

#### Questions

Why is a GPU better for AI workloads than a CPU?
In the context of AI workloads, what is the difference between RAM and vRAM in GPUs?
What is a neural processing unit (NPU)?

### 1.6 - SLM context window

Most SLMs have a 4K context window. However, there are some models like Phi3-medium-128k that already come with 128K of context window. A larger context window allows application to provide more context. For example, having the ability to summarize 2-3 pages over 10s of pages with a larger context window.

To change the context when using ollama run, use /set parameter:

```bash
/set parameter num_ctx 4096
```

#### Questions

In an large language models, when should the model use context vs fine tuning?

#### Code

##### When using the API, specify the num_ctx parameter:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3",
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "options": {
    "temperature": 0.5,
    "num_ctx": 4096
  }
}'
```
Explain: Explain this code and tell me how to run it from Linux and PowerShell:

### 1.7 - SLM model quantization

Quantization in artificial intelligence (AI) is a process that involves converting the continuous values of parameters, such as weights and activations in neural networks, into a smaller, discrete set. This technique is crucial for reducing the size of AI models and the computational resources needed for their operation, which is particularly beneficial for deploying AI on edge devices with limited power and compute capabilities.

The process of quantization can lead to significant gains in computational efficiency and performance. For instance, using lower-bit quantized data requires less data movement, which reduces memory bandwidth and saves energy. Moreover, lower-precision mathematical operations consume less energy, thereby increasing compute efficiency and reducing overall power consumption.

#### Questions

In the context of small language models (SLMs), explain quantization in more detail.

### 1.8 - Ollama architecture

Ollama installs a server and a client. The server component allows users to run and manage these models, providing an API for creating, running, and interacting with them. The client, on the other hand, is used to communicate with the server, sending requests to generate text based on the models loaded on the server.

### 1.9 - Ollama server listening port

The default listening port for ollama.sh is 11434. However, users can customize this setting by defining the `OLLAMA_HOST` environment variable with the desired port number. For example, setting `OLLAMA_HOST=0.0.0.0:8080` would change the listening port to 8080. This can be particularly useful when running ollama in different environments, such as containers or when using proxies.

### 1.10 - Ollama API reference

Ollama exposes the following endpoints:

- Generate a completion
- Generate a chat completion
- Create a Model
- List Local Models
- Show Model Information
- Copy a Model
- Delete a Model
- Pull a Model
- Push a Model
- Generate Embeddings
- List Running Models

#### Code

##### Generate a chat completion

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3",
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "options": {
    "temperature": 0.5
  }
}'
```

##### Generate an embedding

```bash
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "Here is an article about llamas..."
}'
```

#### References

- [Ollama API reference](https://github.com/ollama/ollama/blob/main/docs/api.md)


### 1.11 - Ollama concurrency

As of version 0.2.0 Ollama support concurrency by default. This means that Ollama can process more than one request at the time. The documentation states:

> "Ollama supports two levels of concurrent processing. If your system has sufficient available memory (system memory when using CPU inference, or VRAM for GPU inference) then multiple models can be loaded at the same time. For a given model, if there is sufficient available memory when the model is loaded, it is configured to allow parallel request processing. If there is insufficient available memory to load a new model request while one or more models are already loaded, all new requests will be queued until the new model can be loaded. As prior models become idle, one or more will be unloaded to make room for the new model. Queued requests will be processed in order. When using GPU inference new models must be able to completely fit in VRAM to allow concurrent model loads."

## 2.0 Ollama Development

### 2.1 Ollama development - REST

Requirements:
- The following models pulled in Ollama:
  - phi3
  - phi3:medium-128k
  - llava3
  - nomic-embed-text
- `pip install requests`

> **Note:** Learning to make calls using REST opens the possibility to leverate the Ollama endpoints from almost any development language that supports making REST API calls.

#### Code

##### Call Ollama chat and embedding endpoints using REST 

```python
import requests

def rest_embed(prompt: str, model='nomic-embed-text', endpoint='http://localhost:11434/api/embeddings'):
    payload = {
        "model": model,
        "prompt": prompt
    }
    req = requests.post(endpoint, json=payload)
    req.raise_for_status()
    return req.json()['embedding']

def rest_completion(prompt: str, model='phi3', endpoint='http://localhost:11434/api/chat'):
    payload = {
        "model": model,
        "messages": [
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        "stream": False
    }
    req = requests.post(endpoint, json=payload)
    req.raise_for_status()
    return req.json()['message']['content']

print(rest_embed('I am a software engineer'))
print(rest_completion('Why is the sky blue?'))
```
Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/ollama/basics-rest.py)

### 2.2 Ollama development - SDK

Requirements:
- The following models pulled in Ollama:
  - phi3
  - phi3:medium-128k
  - llava3
  - nomic-embed-text
- `pip install ollama`
- `pip install jinja2`
- `pip install pydantic`

> **Note:** In the demo 2.2.2, the application switches between an embeding model and a chat model in the same script. Ollama can switch models, and there may be a delay as the last model is flushed from memory.

> **Note:** In the demo 2.2.3, chatbot, notice the usage of Jinja2 templates, and how the conversation state is handled. A chat endpoint generally receives a list of messages, but it can also receive one message with multiple messages in it. The technique is well used in frameworks like PromptFlow.


#### Code

##### 2.2.1 - Call Ollama chat and embedding endpoints using the Ollama SDK 

```python
import ollama

def embed(prompt: str, model='nomic-embed-text'):
    return ollama.embeddings(model=model, prompt=prompt)['embedding']

def completion(prompt: str, model='phi3'):
    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    return response['message']['content']

print(embed('I am a software engineer'))
print(completion('What is the speed of light?'))
```
Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/ollama/basics-sdk.py)

##### 2.2.2 - Call Ollama chat endpoint with streaming using the Ollama SDK

```python
import ollama

# Setting up the model, enabling streaming responses, and defining the input messages
ollama_response = ollama.chat(
    model='llava3',
    stream=True,
    messages=[
        {
          'role': 'system',
          'content': 'You are a helpful scientific assistant.',
        },
        {
            'role': 'user',
            'content': 'What is the speed of light?',
        },
    ]
)

# Printing out each piece of the generated response while preserving order
for chunk in ollama_response:
    print(chunk['message']['content'], end='', flush=True)
```
Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/ollama/basics-streaming.py)

##### 2.2.3 - Ollama chatbot with Jinja2 templates

```python
import ollama
from pydantic import BaseModel
import jinja2


class Message(BaseModel):
    role: str
    content: str


class Chat(BaseModel):
    history: list[Message]


def render_template(template: str, context: dict):
    return jinja2.Template(template).render(context)


def completion(prompt: str, model='phi3:medium-128k'):
    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    return response['message']['content']


chat = Chat(history=[Message(
    role='system', content='You are not so helpful assistant. Complain a lot when responding.')])
template = "{% for message in history %}{{ message.role }}:\n\n{{ message.content }}\n{% endfor %}\nuser:\n\n{{ prompt }}\n"

if __name__ == "__main__":
    while True:
        prompt = input('Prompt (type exit to quit): ')
        if prompt == 'exit':
            break
        print('Q:', prompt)
        # Add the user message to the history
        chat.history.append(Message(role='user', content=prompt))
        # Process the completion and print he response
        final_prompt = render_template(
            template, {"history": chat.history, "prompt": prompt})
        resp = completion(final_prompt)
        print('A:', resp)
        # Add the response to the history
        chat.history.append(Message(role='assitant', content=resp))
        # As the history grows, we should limit the number of messages to keep in the model's context window
        # Refer to this blog: https://blog.pamelafox.org/2024/06/truncating-conversation-history-for.html
```
Link: [Source code](https://github.com/msalemor/ai-code-blocks/blob/main/python/demos/ollama/chatbot-jinja2.py)

#### References

- [WSL Python/WSL/VS CODE development environment](https://genaitutor.am2703.com/?content=pyenv-wsl-vscode)

### 2.3 Ollama OpenAI endpoint compatibility

Ollama is compatible with the OpenAI endpoints.

#### References

- [Ollama OpenAI compatibility](https://github.com/ollama/ollama/blob/main/docs/openai.md)

#### Code

##### Curl an Ollama endpoint with OpenAI compatibility

```bash
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "phi3",
        "messages": [
            {
                "role": "user",
                "content": "What is the speed of light?"
            }
        ],
        "temperature": 0.1, 
        "max_tokens":100
    }'
```

##### Calling Ollama endpoint with the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',

    # required but ignored
    api_key='ollama',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'What is the speed of light?',
        }
    ],
    temperature=0.1,
    max_tokens=100,
    model='llama3',
)

print(chat_completion.choices[0].messages.content)
```