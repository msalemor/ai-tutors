# SLM-LLM Integration

## 1 - SLM-LLM Integration

### 1.1 - Overview

### 1.2 - Possible Scenarios

1. **LLM Only:** The solition requires using only an LLM.
2. **SLM Only:** The solution requires using only an SLM.
3. **Hybrid Only:** The solution involves doing some processing in SLMs and some processing in LLMs.

### 1.3 - Hybrid scenarios

### 1.3 - SLM Capabilities

- **Resource Efficiency**: SLMs require less computational power, making them suitable for running on local machines with acceptable performance.
- **Speed**: SLMs can generate data faster due to their smaller model size, especially when the number of concurrent users for an LLM might slow down its inference capabilities.
- **Specialization**: SLMs can be more efficient for specialized tasks that do not require the extensive understanding of language that LLMs provide.
- **Cost-effectiveness**: Deploying SLMs can be more cost-effective, particularly for applications with limited budgets or computational resources.

### 1.4 - Selection criteria

- **Task Specificity**: Small language models are more suitable for specialized tasks.
- **Efficiency**: They are more lightweight and efficient, making them ideal for real-time inference.
- **Accessibility**: Small language models are more accessible and can be deployed on less powerful devices like laptops or mobile devices.

## 2 - Development

### 2.1 - Leveraging existing LLM knowledge for SLMs

It is fairly straight forward to transfer existing knowlege of working with LLMs to SLMs.

#### Code

##### 2.1.1 - Embeddings and completions with Ollama

```python
import requests
import ollama

def embed(prompt:str, model='nomic-embed-text'):
    return ollama.embeddings(model=model, prompt=prompt)['embedding']

def rest_embed(prompt:str, model='nomic-embed-text',endpoint='http://localhost:11434/api/embeddings'):
    payload = {
        "model": model,
        "prompt": prompt
    }
    req = requests.post(endpoint, json=payload)
    req.raise_for_status()
    return req.json()['embedding']

print(len(embed('I am a software engineer')))
print(rest_embed('I am a software engineer'))

def completion(prompt:str, model='phi3'):
    response = ollama.chat(model=model, messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])
    return response['message']['content']


def rest_completion(prompt:str, model='phi3',endpoint='http://localhost:11434/api/chat'):
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

print(completion('Why is the sky blue?'))
print(rest_completion('Why is the sky blue?'))
```
Link: [Source code](https://github.com/msalemor/llm-use-cases/blob/main/notebooks/ollama-basics.ipynb)

### 2.2 - SLM/LLM processing

#### Code

##### 2.2.1 - Process based on token size

```python
import ollama
import common
import tiktoken

client = common.get_openai_client(api_key=common.api_KEY,
        api_version=common.api_version,
        azure_endpoint=common.api_URI)

def gpt_completion(prompt:str, model='gpt4o'):
    completion = client.chat.completions.create(
        model=model,
        messages=[
        {
           "role": "user",
            "content": prompt,
        }]
    )
    
    return completion.choices[0].message.content

    
def ollama_completion(prompt:str, model='llava'):
    response = ollama.chat(model=model, messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])
    return response['message']['content']

encoding = tiktoken.get_encoding("o200k_base")
def get_tokens(text: str) -> str:
    return encoding.encode(text)

def summarize(text: str) -> str:
    template = f"Summarize the following text in one paragraph:\n{text}"
    token_count = len(get_tokens(template))
    print(f"Token count: {token_count}")
    if  token_count> 4096:
        # GPT processing
        print("Processing with GPT")
        return gpt_completion(template)
    else:
        # Ollama processing
        print("Processing with Ollama")
        return ollama_completion(template)

def read_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

short_data = read_file("data/401k.txt")
print(summarize(short_data))

short_data = read_file("data/401k.txt")
print(summarize(short_data))
```
Link: [Source Code](https://github.com/msalemor/llm-use-cases/blob/main/notebooks/slm-llm-processing.ipynb)

##### 2.2.2 - SLM/LLM process based on intent or complexity

```python
import ollama
import common

client = common.get_openai_client(api_key=common.api_KEY,
        api_version=common.api_version,
        azure_endpoint=common.api_URI)

def read_file(file:str):
    with open(file, 'r') as f:
        return f.read()
        
file_401k = read_file('data/401k.txt')
file_benefits = read_file('data/benefits.txt')

def gpt_completion(prompt:str, model='gpt4o',temperature=0.1) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
        {
           "role": "user",
            "content": prompt,
        }],
        temperature=temperature
    )
    
    return completion.choices[0].message.content

    
def ollama_completion(prompt:str, model='llava',temperature=0.1) -> int:
    response = ollama.chat(model=model, messages=[
    {
        'role': 'user',
        'content': prompt,        
    },    
    ],options = {'temperature': temperature})
    
    return response['message']['content']

def determine_complexity(prompt:str):
    template ="""system:
You are an assistant that can score the complexity of a user's question or statement. A score can be from 0 to 10, where 0 is the least complex and 10 is the most complex.

user:
Score the following question or statement from 0 to 10:

<PROMPT>

Respond with the score only.
"""
    res = ollama_completion(template.replace('<PROMPT>', prompt))
    # is res a number?
    try:
        return int(res)
    except:
        return 0



def intent_detection(prompt:str):
    template ="""system:
You are an assistant that can determine intent from the following list of intents:

401k: a user asks a 401k.
benefits: a user asks a company benefits.
weather: a user asks a questions about the weather.
other: a user asks a question about anything else.

user:
<PROMPT>

Respond with the intent word only unless it is the weather intent then respond with the intent and the location. For example: weather, New York.
"""

    return ollama_completion(template.replace('<PROMPT>', prompt))

def process_for_complexity(prompt: str):
    score = determine_complexity(prompt)
    print("Score:",score)
    if score > 5:
        # GPT processing
        print("GPT processing")
        print(gpt_completion(prompt))
    else:
        # Ollama processing
        print("Ollama processing")
        print(ollama_completion(prompt))


def process_for_intent(prompt: str):
    intent = intent_detection(prompt).strip()
    print("Intent:",intent)
    if intent == 'weather':
        location = intent.split(',')[1].strip()
        print(f"Intent is weather for {location}. You need a weather service.")
    elif intent == '401k':
        print("GPT processing")
        print(gpt_completion(f"{prompt}\nContent:\nfile_401k\nRespond in one senetence. Use only the provided content."))
    elif intent == 'benefits':
        print("Ollama processing")
        print(ollama_completion(f"{prompt}\nContent:\nfile_benefits\nRespond in one sentence. Use only the provided content."))
    else:
        print("Ollama processing")
        print(ollama_completion(prompt))

# Process for intent
process_for_intent("What is the speed of light?")
process_for_intent("What is a 401k account?")
process_for_intent("What are some company benefits?")
process_for_intent("What is the weather in London?")

# Process for complexity
process_for_complexity("What is the speed of light?")
process_for_complexity("""Who was the most decorated (maximum medals) individual athlete in the Olympic games that were held at Sydney? Take a step-by-step approach in your response, cite sources and give reasoning before sharing final answer in the below format: ANSWER is: <name>""")
```
Link: [Source code](https://github.com/msalemor/llm-use-cases/blob/main/notebooks/slm-llm-processing-intent-scoring.ipynb)
