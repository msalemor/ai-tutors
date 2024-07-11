# Lmstudio application development

#### Related

- [Ollama Application Development](https://genaitutor.am2703.com/?content=ollama-development)
- [OpenAI Development Intro](https://genaitutor.am2703.com/?content=genai-intro)
- [OpenAI Development Intermediate](https://genaitutor.am2703.com/?content=genai-intermediate)
- [WSL Python/WSL/VS CODE development environment](https://genaitutor.am2703.com/?content=pyenv-wsl-vscode)

## 1.0 - Application Development Basics with Ollama and SLMs 

### 1.1 - Guide Requirements and recommendations

- A local machine with CPU/RAM
  - A GPU/vRAM is recommended for better performance
- Intermediate Python development knowledge including:
  - Setting up a Python development environment
  - Calling REST endpoints
- Don't skip on reading the `Notes`, they provide valuable information
- Reading is good, actually developing is best

### 1.2 - What is LMStudio?

LMStudio allows you to:

- 🤖 Run LLMs on your laptop, entirely offline
- 👾 Use models through the in-app Chat UI or an OpenAI compatible local server
- 📂 Download any compatible model files from HuggingFace 🤗 repositories
- 🔭 Discover new & noteworthy LLMs in the app's home page

![LMStudio screenshot](https://lmstudio.ai/static/media/demo2.9df5a0e5a9f1d72715e0.gif)

### 1.3 - Installing LMStudio

Requirements:

- Apple Silicon Mac (M1/M2/M3) with macOS 13.6 or newer
- Windows / Linux PC with a processor that supports AVX2 (typically newer PCs)
- 16GB+ of RAM is recommended. For PCs, 6GB+ of VRAM is recommended
- NVIDIA/AMD GPUs supported

LMStudio can be installed in:

- Windows
  - [**Download**](https://releases.lmstudio.ai/windows/0.2.27/latest/LM-Studio-0.2.27-Setup.exe)
- Linux (Beta)
  - Run: `https://releases.lmstudio.ai/linux/x86/0.2.27/beta/LM_Studio-0.2.27.AppImage`
- MacOS
  - [**Download**](https://releases.lmstudio.ai/mac/arm64/0.2.27/latest/LM-Studio-0.2.27-arm64.dmg)

> **Note:** In Windows LMStudio comes with an update services.

### 1.4 - Pulling, running, installing, listing, and deleting models

LMStudio implements a point and click UI interface to pull, chat with, and delete models.

> **Note:** LMStudio offers several version for the same model. For example, there's a phi3, phi3-medium, phi3-128k-medium, etc. There are also different quantization models. Make sure to check to options to find the model that may best fit your requirements.

#### References

- [LMStudio documentation](https://lmstudio.ai/docs/welcome)
- [WSL NVIDIA Driver WSL Installation](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)

### 1.5 - Performance (CPU/RAM vs GPT/vRAM)

You can run LMStudio with CPU and RAM, but for best performance it is recommended that you have a system with GPU and vRAM. Make sure you install the NVDIA drivers, for example if you are using Windows WSL.

#### Questions

Why is a GPU better for AI workloads than a CPU?
In the context of AI workloads, what is the difference between RAM and vRAM in GPUs?
What is a neural processing unit (NPU)?

### 1.6 - SLM context window

Most SLMs have a 4K context window. However, there are some models like Phi3-medium-128k that already come with 128K of context window. A larger context window allows application to provide more context. For example, having the ability to summarize 2-3 pages over 10s of pages with a larger context window.

You can change the context window as part of the chat UI or when making a REST call using the API.

#### Questions

In an large language models, when should the model use context vs fine tuning?

#### Code

##### When using the API, specify the num_ctx parameter:

```bash
curl http://localhost:1234/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{ 
  "model": "phi3",
  "messages": [ 
    { "role": "system", "content": "You are a helpful coding assistant." },
    { "role": "user", "content": "How do I init and update a git submodule?" }
  ], 
  "temperature": 0.7, 
  "max_tokens": -1,
  "stream": true
}'
```
Explain: Explain this code and tell me how to run it from Linux and PowerShell:

### 1.7 - SLM model quantization

Quantization in artificial intelligence (AI) is a process that involves converting the continuous values of parameters, such as weights and activations in neural networks, into a smaller, discrete set. This technique is crucial for reducing the size of AI models and the computational resources needed for their operation, which is particularly beneficial for deploying AI on edge devices with limited power and compute capabilities.

The process of quantization can lead to significant gains in computational efficiency and performance. For instance, using lower-bit quantized data requires less data movement, which reduces memory bandwidth and saves energy. Moreover, lower-precision mathematical operations consume less energy, thereby increasing compute efficiency and reducing overall power consumption.

#### Questions

In the context of small language models (SLMs), explain quantization in more detail.

### 1.8 - LMStudio architecture

TBD

### 1.9 - LMStudio server listening port

The default listening port for LMStudio.ai is 1234. 

### 1.10 - LMStudio API reference

LMStudio exposes the following endpoints:

- GET /v1/models
- POST /v1/chat/completions
- POST /v1/embeddings            
- POST /v1/completions

#### Code

##### Generate a chat completion

```bash
curl http://localhost:1234/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{ 
  "model": "phi3",
  "messages": [ 
    { "role": "system", "content": "You are a helpful coding assistant." },
    { "role": "user", "content": "How do I init and update a git submodule?" }
  ], 
  "temperature": 0.7, 
  "max_tokens": -1,
  "stream": true
}'
```

##### Generate an embedding

```bash
curl http://localhost:1234/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Your text string goes here",
    "model": "model-identifier-here"
  }'
```

#### References

- [Ollama API reference](https://github.com/ollama/ollama/blob/main/docs/api.md)


### 1.11 - LMStudio concurrency

By default, LM Studio processes one request at a time, queuing incoming requests1. Currently, there isn’t a built-in configuration to enable parallel request processing directly within the app1. However, you can manage multiple requests by running multiple instances of LM Studio or using a load balancer to distribute requests across these instances.

## 2.0 - LMStudio Development

### 2.1 - LMStudio development - REST

Requirements:
- The following models pulled in Ollama:
  - phi3
  - phi3:medium-128k
  - llava3
  - nomic-embed-text
- `pip install requests`

> **Note:** Learning to make calls using REST opens the possibility to leverate the Ollama endpoints from almost any development language that supports making REST API calls.

#### Code

##### 2.1.1 - Call LMStudio chat and embedding endpoints

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

### 2.2 LMStudio development - OpenAI SDK

Requirements:
- The following models pulled in Ollama:
  - phi3
  - phi3:medium-128k
  - llava3
  - nomic-embed-text
- `pip install openai`
- `pip install jinja2`
- `pip install pydantic`

> **Note:** In the demo 2.2.2, the application switches between an embeding model and a chat model in the same script. Ollama can switch models, and there may be a delay as the last model is flushed from memory.

> **Note:** In the demo 2.2.3, chatbot, notice the usage of Jinja2 templates, and how the conversation state is handled. A chat endpoint generally receives a list of messages, but it can also receive one message with multiple messages in it. The technique is well used in frameworks like PromptFlow.

#### Code

##### 2.2.1 - Call Ollama chat and embedding endpoints using the Ollama SDK 

```python
from openai import OpenAI
client = OpenAI()

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
from openai import OpenAI
client = OpenAI()

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
from openai import OpenAI
client = OpenAI()
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
