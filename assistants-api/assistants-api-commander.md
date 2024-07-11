# Azure Assistants API Commander

## 1.0 - Running Assistants API Commander

### 1.1 - Overview

Assistants API Commander is an OpenAI Assistants API application designed to showcase how to build an application using Assistants API.

![Assistants API Commander screenshot](https://raw.githubusercontent.com/msalemor/assistants-api-commander/main/images/assistants-api-commander.png)

This application implements a Python FastAPI/uvicorn backend with a SolidJS Frontend. The backend implements basically three APIs:

- `POST /api/create`
- `POST /api/process`
- `DELETE /api/delete/{username}`

The main Assistants processing code is in the `playground.py` file.

The Frontend also implements logic to hit these endpoints.

#### References

- [GitHub - Full code](https://github.com/msalemor/assistants-api-commander)

### 1.2 - Assistant API Concepts

The OpenAI Assistant API is a robust framework designed to facilitate the creation of AI-powered assistants within applications. Here's an in-depth look at its main concepts:

1. **Assistant Creation**: An Assistant is an entity that can be configured to respond to user messages. It is created by defining custom instructions and selecting a model. Tools such as Code Interpreter, File Search, and Function Calling can be enabled to enhance the Assistant's capabilities.

2. **Thread Management**: A Thread represents a conversation between a user and one or many Assistants. It is created when a conversation starts and manages the flow of messages. Messages can contain text and files, and there's no limit to the number added to a Thread.

3. **Message Handling**: Users or applications generate Message objects that are added to the Thread. The API smartly truncates any context that doesn't fit into the model's context window.

4. **Running the Assistant**: Once all user Messages are added to a Thread, the Assistant can be run to generate a response. This process leverages the model and tools associated with the Assistant to produce replies, which are then added to the Thread as Assistant Messages.

5. **Tool Integration**: The API currently supports three types of tools:
   - **Code Interpreter**: Executes code snippets.
   - **File Search**: Searches through files.
   - **Function Calling**: Calls external functions.

6. **Persistent Threads**: Assistants can access persistent Threads, which simplify AI application development by storing message history and truncating it as needed.

7. **File Access**: Assistants can interact with files in various formats, either as part of their creation or within Threads.

8. **Stateful Interactions**: Unlike the stateless chat completions API, the Assistants API is stateful, meaning it manages conversation states and optimizes threads to fit within the model's context window.

9. **Use Cases**: The API can power a wide range of applications, from product recommenders and sales analyst apps to coding assistants and Q&A chatbots.

For developers, the Assistants API represents a significant advancement, offering a streamlined process for integrating sophisticated AI functionalities into applications. It provides a foundation for creating interactive, intelligent systems capable of understanding and responding to user queries in a contextually relevant manner.

### 1.3 - Backend Environment Variables

```bash
OPENAI_URI=https://<NAME>.openai.azure.com/
BASE_URL=https://<NAME>openai.azure.com/openai/
OPENAI_KEY=<API_KEY>
OPENAI_GPT_DEPLOYMENT=<MODEL>
OPENAI_VERSION=2024-02-15-preview
DEPLOY_SPA=True
```

### 1.4 - Backend Package Requirements (Requirements.txt)

The required packages include:

```text
python-dotenv
fastapi
pydantic
uvicorn
yfinance
openai
httpx
```

Install by typing: `pip install -r requirements.txt`

### 1.5 - Backend APIs

 The backend implements basically three APIs:

- `POST /api/create`
- `POST /api/process`
- `DELETE /api/delete/{username}`

The main Assistants processing code is in the `playground.py` file.

#### Code

##### App Setup

```python
from fastapi.staticfiles import StaticFiles
import kvstore
from openai import AzureOpenAI
from models import AssistantCreateRequest, AssistantCreateResponse, ResponseMessage, PromptRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
import playground
import logging
import settings
# Read the environment variables into settings
settings = settings.Instance()


logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

# Create the SQLite KV store
kvstore.create_store()


# Create an Azure OpenAI client
client = AzureOpenAI(api_key=settings.api_key,
                     api_version=settings.api_version,
                     azure_endpoint=settings.api_endpoint)

# Create a FastAPI app
app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

##### POST Create Assistant

```python
@app.post("/api/create", response_model=AssistantCreateResponse)
async def create_assistant(request: AssistantCreateRequest):

    if request.userName is None or request.userName == "":
        raise HTTPException(
            status_code=400, detail=".userName was not provided")
    if request.name is None or request.name == "":
        raise HTTPException(
            status_code=400, detail=".name was not provided")
    if request.instructions is None or request.instructions == "":
        raise HTTPException(
            status_code=400, detail=".instructions were note provided")
    if request.fileURLs is None or request.fileURLs == []:
        raise HTTPException(
            status_code=400, detail=".fileURLs missing. No files were provided")

    # Create the files
    file_ids = await playground.create_files(
        client, request.userName, request.fileURLs)

    # Create the Assistant and the thread for the user
    (assistant_id, thread_id, tools) = playground.create_assistant(client,
                                                                   request.userName, request.name, request.instructions, file_ids, settings.api_deployment_name)

    if assistant_id is None or thread_id is None:
        raise HTTPException(
            status_code=500, detail="Unable to create the assistant")

    return AssistantCreateResponse(userName=request.userName, name=request.name,
                                   instructions=request.instructions, tools=tools,
                                   assistant_id=assistant_id,
                                   thread_id=thread_id, file_ids=file_ids)
```

##### POST - Process

```python
@app.post("/api/process", response_model=list[ResponseMessage])
async def post_process(request: PromptRequest):
    if request.userName is None or request.userName == "":
        raise HTTPException(
            status_code=400, detail="No user name name was provided. User name is required.")

    if request.prompt is None or request.prompt == "":
        raise HTTPException(
            status_code=400, detail="No prompt was provided. Prompt is required.")

    # Find the assistant for the user
    user_assistant = kvstore.get_assistant(request.userName)
    assistant = None
    if user_assistant is None:
        raise HTTPException(
            status_code=404, detail=f"Assistant not found for user {request.userName}")
    try:
        assistant = client.beta.assistants.retrieve(user_assistant.value)
    except:
        raise HTTPException(
            status_code=404, detail=f"Assistant not found for user {request.userName}")

    # Find the thread for the user
    user_thread = kvstore.get_thread(request.userName)
    thread = None
    if user_thread is None:
        raise HTTPException(
            status_code=404, detail=f"thread not found for user {request.userName}")
    try:
        thread = client.beta.threads.retrieve(user_thread.value)
    except:
        raise HTTPException(
            status_code=404, detail=f"thread not found for user {request.userName}")

    return await playground.process_prompt(client, assistant, thread, request.prompt, settings.email_URI, request.userName)

```

##### DELETE - Delete Assistant for a given username

```python
# Delete an Assistant
@app.delete("/api/delete/{userName}")
def delete(userName: str):
    error = playground.delete_assistant(client, userName)
    if error is not None:
        raise HTTPException(
            status_code=404, detail=f"User {userName} note found")

    # Delete the KVStore entries for the user
    count = kvstore.del_user(userName)
    if count > 0:
        return {"message": f"Assistant deleted for user: {userName}"}
    else:
        raise HTTPException(
            status_code=404, detail=f"User {userName} not found")
```

#### References

- [File: `playground.py`](https://github.com/msalemor/assistants-api-commander/blob/main/src/backend/playground.py)

### 1.6 - Frontend

#### Code

##### Create an Assistant per user

```typescript
 const CreateAssistant = async () => {
    if (processing()) return
    if (runningAssistant().assistant_id !== '') {
      alert('An AI Assistant is already running. Please delete it before creating a new one.')
      return
    }
    if (settings().user === '' || settings().name === '' || settings().instructions === '' || settings().files === '') {
      alert('User ID, Assistant Name, Instructions and files are required to create an AI Assistant.')
      return
    }

    setProcessing(true)
    const fileURLs = settings().files.split(',').map((file) => file.trim())
    const payload: IAssistantCreateRequest = {
      userName: settings().user,
      name: settings().name,
      instructions: settings().instructions,
      fileURLs,
    }
    try {
      await axios.post<IAssistantCreateResponse>(POST_CREATE, payload)
      await UpdateStatus()
    }
    catch (err) {
      console.log(err)
    }
    finally {
      setProcessing(false)
    }
  }
```

##### Process an Assistant Prompt for a user

```typescript
  const Process = async () => {
    if (processing()) return
    try {
      const payload: { userName: string, prompt: string } = {
        userName: settings().user,
        prompt: prompt()
      }
      setProcessing(true)
      const response = await axios.post<IResponseMessage[]>(POST_PROCESS, payload)
      const additional_messages = response.data
      setThreadMessages([...threadMessages(), ...additional_messages])
      setPrompt('')
    }
    catch (err) {
      console.log(err)
    }
    finally {
      setProcessing(false)
    }
  }
```

##### Delete Assistant

```typescript
const DeleteAssistant = async () => {
    try {
      const ask = confirm('Are you sure you want to delete the AI Assistant?')
      if (!ask) return
      setProcessing(true)
      const response = await axios.delete(DELETE_ASSISTANT.replace('{name}', settings().user))
      console.log(response)
      setPrompt('')
      setSettings({ ...settings(), user: '', name: '', instructions: '', files: '' })
      setThreadMessages([])
      setRunningAssistant({ ...runningAssistant(), assistant_id: '', thread_id: '', files: [] })
    }
    catch (err) {
      console.log(err)
    }
    finally {
      setProcessing(false)
    }
  }
```