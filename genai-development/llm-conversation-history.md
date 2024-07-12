# Managing LLM Conversation History

#### Related

- [OpenAI Development Intro](https://genaitutor.am2703.com/?content=genai-intro)
- [OpenAI Development Intermediate](https://genaitutor.am2703.com/?content=genai-intermediate)

## 1.0 - Managing LLM Conversation history

### 1.1 - Chat completion API

Generally, a chat model, like OpenAI GPT4, can keep a conversation history. Calling a model involves passing one or more messages that contain a role and the content. 

A chat model generally implements three roles:

- **`system`**: Sets the context and instructions for the conversation, guiding the behavior of the model.
- **`user`**: Represents the human user, providing queries or prompts to which the model responds.
- **`assistant`**: The model itself, responding to the user's input based on the context set by the system.

These roles help the system to understand the structure of the conversation.

Imagine a user is accessing a travel assistant bot and asks, "What are some restaurants in London?" and after receiving the response, the user asks, "what are some more?" How would the system know that the user is refererring to restaurants in London? The answer is in keeping the conversation history which may endup looking something like the code in section 1.1.1.

#### Code

##### 1.1.1 - Curl a chat endpoint

```bash
POST https://{endpoint}/openai/deployments/{deployment-id}/chat/completions?api-version=2024-06-01

{
 "messages": [
  {
   "role": "system",
   "content": "You're a helpful assistant that can provide travel related information."
  },
  {
   "role": "user",
   "content": "What are some good restaurants in London?"
  },
  {
   "role": "assistant",
   "content": "- Fish & Chips Limited\n- La Bella Roma"
  },
  {
   "role": "user",
   "content": "What are some more?"
  }
 ]
}
```

### 1.2 - Ways of handling conversation history

There are two main ways of handling the conversation history:

1. The first way involve keeping a list of the messages and passing those messages to the completion API. In other words, the messages propperty looks like this:

```text
"messages": [
  {
   "role": "system",
   "content": "You're a helpful assistant that can provide travel related information."
  },
  {
   "role": "user",
   "content": "What are some good restaurants in London?"
  },
  {
   "role": "assistant",
   "content": "- The Laughing Hallibut\n- Zedel"
  },
  {
   "role": "user",
   "content": "What are some more?"
  }
 ]
```

2. The second way involves using a template. The template will render one long message, and that one long message will carry the history and will be passed to the completion API as one messge. In other words, the message looks like this:

```text
"messages": [
  {
   "role": "user",
   "content": "system:\nYou're a helpful assistant that can provide travel related information.\n
user:\What are some good restaurants in London?\nassistant:\n- The Laughing Hallibut\n
- Zedel\nuser:What are some more?"
  }
 ]
```
> **Note:** Many Prompt Flow flows use this template technique.

#### References

- [Prompt Flow - Chat basic](https://github.com/microsoft/promptflow/blob/main/examples/flows/chat/chat-basic/chat.jinja2)

### 1.3 - Trimming the messages (Cost & Performance)

Most chat completion models have token limitations, granted this have been increasing from 4k to 16k to now some larger than 128k. It is still up to the application to manage the number of message it shoud use in a conversation history. Good management can have a impact on the response quality, performance and cost of the system.

LangChain, for example, provides a trimming object (section 1.3.1). Using this object, the user, for example, can set how many messages should be used in the history, and if the system message should always be included.

#### Code

##### 1.3.1 - LangChain trimming method

```python
from langchain_core.messages import SystemMessage, trim_messages

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)
```
Link: [Souce code](https://python.langchain.com/v0.2/docs/tutorials/chatbot/)

### 1.4 - Console based chatbots

In the previous section, we discussed two different approaches for implementing chatbots: using arrays and using Jinja2 templates.

The first approach, demonstrated in section 1.4.1, involves implementing a chatbot using arrays. In this approach, the conversation history is initialized with an empty array called `messages`. User messages and response messages are then added to this array using the `messages.append` method.

On the other hand, the second approach, shown in section 1.4.2, utilizes Jinja2 templates to handle the conversation history. In this approach, the messages are stored in a `Chat` object, which contains a list of `Message` objects. The template is rendered before passing it to the completion model, allowing for dynamic generation of the conversation history.

These two approaches provide different ways to manage and handle the conversation history in chatbot implementations. The choice between them depends on the specific requirements and preferences of the project.

#### Code

##### 1.4.1 Chatbot with an array of message

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

##### 1.4.2 - Chatbot with a template

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

### 1.5 - UI Based Chatbot Conversation History

When implementing a UI Chatbot, it is important to handle the conversation history. There are different approaches to achieve this. One common method is to leverage the user's `localstorage` to store the conversation history locally on the client side. Another option is to save the conversation state in a database like Cosmos DB, allowing for persistence and retrieval of the conversation history across sessions.

Managing the conversation history is crucial for maintaining context and providing a seamless user experience. By storing and retrieving previous messages, the chatbot can better understand user queries and provide relevant responses. This can be especially useful in scenarios where the conversation spans multiple interactions or sessions.

Implementing conversation history in a UI Chatbot requires careful consideration of the storage mechanism and the data structure used to store the messages. It is important to ensure that the conversation history is easily accessible and can be efficiently retrieved when needed.

#### Code

##### 1.5.1 - FastAPI Chatbot

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

##### 1.5.2 - LLM messages trimmer function

```typescript
export interface IMessage {
    role: string
    content: string
}

export function trimmer(messages: IMessage[],
    keepSystemMessage: boolean = true,
    history: number = 2) {

    if (!messages || messages.length === 0)
        return []

    const final = []

    // There should only be one system message, but just in case
    const systemMessage = messages.filter(message => message.role === 'system')[0]

    // Add the system message from the messages
    if (keepSystemMessage && systemMessage)
        final.push(systemMessage)

    // Keep 2 * history messages from the bottom
    if (messages.length > 2 * history + 1 + (systemMessage ? 1 : 0)) {
        const start = messages.length - 2 * history - 1
        for (let i = start; i < messages.length; i++) {
            final.push(messages[i])
        }
        return final
    }
    else
        // if it is small return all the messages
        return messages
}
```

##### 1.5.2 - Calling an LLM with a trimmer function

```Typescript
import axios from "axios";

const config = {
    headers: {
        'Content-Type': 'application/json',
        'api-key': 'API_KEY'
    }
}

const OPENAI_URI = 'https://api.openai.com/v1/engines/davinci-codex/completions'

export async function chatbotService(messages: IMessage[]) {
    try {
        const payload = {
            messages: trimmer(messages),
            temperature: 0.1,
        }
        const resp = await axios.post(OPENAI_URI, payload, config)
        return resp.data
    }
    catch (error) {
        console.error(error)
    }
}
```