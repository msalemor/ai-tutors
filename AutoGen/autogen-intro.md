# AutoGen - Intro


## 1.0 - AutoGen

### 1.1 - What is AutGen?

![Autogen](https://microsoft.github.io/autogen/assets/images/autogen_agentchat-250ca64b77b87e70d34766a080bf6ba8.png)

AutoGen is an open-source programming framework developed by Microsoft for building AI agents and facilitating cooperation among multiple agents to solve tasks. It aims to streamline the development and research of agentic AI, similar to how PyTorch does for deep learning.

Key features of AutoGen include:

- Multi-agent interactions: Agents can interact with each other to solve complex tasks.
- Customizable agents: Developers can create agents with specialized capabilities and roles.
- Enhanced LLM inference: It supports advanced usage patterns like error handling and context programming.
- Autonomous and human-in-the-loop workflows: It integrates both automated processes and human inputs.
- Diverse conversation patterns: Supports various conversation topologies and patterns for complex workflows.

#### References

- [AutoGen Introduction](https://microsoft.github.io/autogen/docs/tutorial/introduction/)

### 1.2 - What is an Agent?

In AutoGen, an agent is an entity designed to act on behalf of human intent. Agents can send and receive messages, respond to other agents, and perform actions based on their capabilities. They can be powered by various backends, including large language models (LLMs) like GPT-4, code executors, human inputs, or a combination of these.

### 1.3 - Multi-Agent Conversations

AutoGen is considered a multi-agent conversation framework because it enables multiple agents to interact and collaborate to solve complex tasks. 

#### References

- [https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat](https://microsoft.github.io/autogen/docs/tutorial/introduction/)

### 1.4 - Enhanced Inference and Code Execution?

Enhaced inference and Code Execution are feature that can be used without building full Agents:

- Enhanced Inference: caching, templates, multi-endpoint
- Code Execution: local and docker

#### Code

##### 1.4.1 - Enhanced Inference

```python
import os
from autogen import OpenAIWrapper
from dotenv import load_dotenv

load_dotenv()

model = os.getenv("GPT_MODEL")
endpoint=os.getenv("ENDPOINT")
api_key=os.getenv("API_KEY")
api_version=os.getenv("API_VERSION")

config_list = [
    {
        "base_url": endpoint,
        "api_key": api_key,
        "model": model,
        "api_type": "azure",
        "api_version": api_version
    }
]

client : OpenAIWrapper = OpenAIWrapper(config_list=config_list)

# First call
response = client.create(messages=[{"role": "user", "content": "What are some Python learning tips."}], model=model)
print(client.extract_text_or_completion_object(response))
client.print_usage_summary()
print(response.cost)

# Caching
response = client.create(messages=[{"role": "user", "content": "What are some Python learning tips."}], model=model)
print(client.extract_text_or_completion_object(response))
client.print_usage_summary()
print(response.cost)

# Disabling cache
client : OpenAIWrapper = OpenAIWrapper(config_list=config_list, cache_seed=None)
response = client.create(messages=[{"role": "user", "content": "Python learning tips."}], model=model)
print(client.extract_text_or_completion_object(response))
client.print_usage_summary()
print(response.cost)
```

##### 1.4.2 - Code Execution

```python
import os
import pprint
from pathlib import Path
from autogen import ConversableAgent
from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor
from dotenv import load_dotenv

load_dotenv()

model = os.getenv("GPT_MODEL")
endpoint=os.getenv("ENDPOINT")
api_key=os.getenv("API_KEY")
api_version=os.getenv("API_VERSION")

work_dir.mkdir(exist_ok=True)

executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
print(
    executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="python", code="print('Hello Word!')\nfor i in range(0,10,2):\n  print(i)"),
        ]
    )
)
```

#### References

- [Enhanced Inference](https://microsoft.github.io/autogen/docs/Use-Cases/enhanced_inference/)
- [Enahanced Inference Demo](https://github.com/msalemor/ai-code-blocks/blob/main/python/notebooks/autogen/enhanced-inference.ipynb)
- [Code Execution](https://microsoft.github.io/autogen/docs/topics/code-execution/cli-code-executor)
- [Code Executor Demo](https://github.com/msalemor/ai-code-blocks/blob/main/python/notebooks/autogen/code-executor.ipynb)


### 1.5 - What are some challenges of working with AutoGen agents?

- Configuring the agents
- Taking control of the conversation and the history of the conversation
- Stopping the conversation between agents
- Using the conversation in an application


#### Code

##### 1.5.1 - Sample Code

```python
```

## 2.0 - AutoGen Studio

### 2.1 - What is AutoGen Studio?

AutoGen Studio is an open-source, user-friendly interface designed to help you rapidly prototype and manage AI agents. It is built on top of the AutoGen framework, which orchestrates multi-agent workflows. Here are some key features:

- Declarative Agent Definition: You can define and modify agents and workflows through a point-and-click, drag-and-drop interface1.
- Interactive Prototyping: Allows you to create chat sessions with agents, view results, and manage workflows.
- Skill Enhancement: You can add specific skills to agents to accomplish more complex tasks.
- Gallery: Stores your AI development sessions for future reference.
- Open Source: Available for installation via pip, making it accessible for developers.

> **Note:** AutoGen Studio is an amazing tool to learn about AutoGen.

### 2.2 - How does Studio Handle Conversations?

#### References

- [AutoGen Studio](https://github.com/microsoft/autogen/tree/main/samples/apps/autogen-studio/autogenstudio)
- [workflowmanager.py](https://github.com/microsoft/autogen/blob/main/samples/apps/autogen-studio/autogenstudio/workflowmanager.py)


#### Code

##### 2.2.1 - Extended Conversable Agent

```python
class ExtendedConversableAgent(autogen.ConversableAgent):
    def __init__(
        self,
        message_processor=None,
        a_message_processor=None,
        a_human_input_function=None,
        a_human_input_timeout: Optional[int] = 60,
        connection_id=None,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.message_processor = message_processor
        self.a_message_processor = a_message_processor
        self.a_human_input_function = a_human_input_function
        self.a_human_input_response = None
        self.a_human_input_timeout = a_human_input_timeout
        self.connection_id = connection_id

    def receive(
        self,
        message: Union[Dict, str],
        sender: autogen.Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        if self.message_processor:
            self.message_processor(sender, self, message, request_reply, silent, sender_type="agent")
        super().receive(message, sender, request_reply, silent)

    async def a_receive(
        self,
        message: Union[Dict, str],
        sender: autogen.Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> None:
        if self.a_message_processor:
            await self.a_message_processor(sender, self, message, request_reply, silent, sender_type="agent")
        elif self.message_processor:
            self.message_processor(sender, self, message, request_reply, silent, sender_type="agent")
        await super().a_receive(message, sender, request_reply, silent)

    # Strangely, when the response from a_get_human_input == "" (empty string) the libs call into the
    # sync version.  I guess that's "just in case", but it's odd because replying with an empty string
    # is the intended way for the user to signal the underlying libs that they want to system to go forward
    # with whatever function call, tool call or AI generated response the request calls for.  Oh well,
    # Que Sera Sera.
    def get_human_input(self, prompt: str) -> str:
        if self.a_human_input_response is None:
            return super().get_human_input(prompt)
        else:
            response = self.a_human_input_response
            self.a_human_input_response = None
            return response

    async def a_get_human_input(self, prompt: str) -> str:
        if self.message_processor and self.a_human_input_function:
            message_dict = {"content": prompt, "role": "system", "type": "user-input-request"}

            message_payload = {
                "recipient": self.name,
                "sender": "system",
                "message": message_dict,
                "timestamp": datetime.now().isoformat(),
                "sender_type": "system",
                "connection_id": self.connection_id,
                "message_type": "agent_message",
            }

            socket_msg = SocketMessage(
                type="user_input_request",
                data=message_payload,
                connection_id=self.connection_id,
            )
            self.a_human_input_response = await self.a_human_input_function(
                socket_msg.dict(), self.a_human_input_timeout
            )
            return self.a_human_input_response

        else:
            result = await super().a_get_human_input(prompt)
            return result


class ExtendedGroupChatManager(autogen.GroupChatManager):
    def __init__(
        self,
        message_processor=None,
        a_message_processor=None,
        a_human_input_function=None,
        a_human_input_timeout: Optional[int] = 60,
        connection_id=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.message_processor = message_processor
        self.a_message_processor = a_message_processor
        self.a_human_input_function = a_human_input_function
        self.a_human_input_response = None
        self.a_human_input_timeout = a_human_input_timeout
        self.connection_id = connection_id

    def receive(
        self,
        message: Union[Dict, str],
        sender: autogen.Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        if self.message_processor:
            self.message_processor(sender, self, message, request_reply, silent, sender_type="groupchat")
        super().receive(message, sender, request_reply, silent)

    async def a_receive(
        self,
        message: Union[Dict, str],
        sender: autogen.Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ) -> None:
        if self.a_message_processor:
            await self.a_message_processor(sender, self, message, request_reply, silent, sender_type="agent")
        elif self.message_processor:
            self.message_processor(sender, self, message, request_reply, silent, sender_type="agent")
        await super().a_receive(message, sender, request_reply, silent)

    def get_human_input(self, prompt: str) -> str:
        if self.a_human_input_response is None:
            return super().get_human_input(prompt)
        else:
            response = self.a_human_input_response
            self.a_human_input_response = None
            return response

    async def a_get_human_input(self, prompt: str) -> str:
        if self.message_processor and self.a_human_input_function:
            message_dict = {"content": prompt, "role": "system", "type": "user-input-request"}

            message_payload = {
                "recipient": self.name,
                "sender": "system",
                "message": message_dict,
                "timestamp": datetime.now().isoformat(),
                "sender_type": "system",
                "connection_id": self.connection_id,
                "message_type": "agent_message",
            }
            socket_msg = SocketMessage(
                type="user_input_request",
                data=message_payload,
                connection_id=self.connection_id,
            )
            result = await self.a_human_input_function(socket_msg.dict(), self.a_human_input_timeout)
            return result

        else:
            result = await super().a_get_human_input(prompt)
            return result
```
