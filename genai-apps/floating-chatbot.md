# Floating Chatbot

#### Related

- [OpenAI Development Intro](https://genaitutor.am2703.com/?content=genai-intro)
- [OpenAI Development Intermediate](https://genaitutor.am2703.com/?content=genai-intermediate)

## 1.0 - Floating Chatbot

### 1.1 - Overview

![Screenshot](https://github.com/msalemor/ai-tutors/blob/main/genai-apps/images/floating-chatbot-0.png?raw=true?raw=true)

![Screenshot](https://github.com/msalemor/ai-tutors/blob/main/genai-apps/images/floating-chatbot-1.png?raw=true)

This tutorial showcases how to add a floating AI chatbot to an existing application. All the code is React frontend code. The main files are:

1. `src/App.tsx`: This simulates the landing page.
2. `src/components/floatingbot.tsx`: This is a react component with the main floating chatbot functionality.
3. `src/services/chatbotservice.tsx`: This is a service to call OpenAI from the frontend while maintaining history.

### 1.2 - Features

The site displays the landing page. This could be an e-retail, informational, blog, or any other application.

## 2.0 - React/TailwindCSS Application Frontend

### 2.1 - Required Packages

Beside the React core packages, the application relies on:

- axios: Making API calls.
- react-icons: Library of icons.
- react-markdown: Converts markdown results to HTML.
- remark-gfm: Give the HTML converter more functionality.

#### Code

##### 2.1.1 - NPM packages

```json
{
  "dependencies": {
    "axios": "^1.7.2",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-icons": "^5.2.1",
    "react-markdown": "^9.0.1",
    "remark-gfm": "^4.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "tailwindcss": "^3.4.4",
  }
}
```
### 2.1 - `src/App.tsx`

##### Code Title

```tsx
import FloatingBot from "./components/floatingbot"

function App() {

  return (
    <>
      <header className="flex items-center h-[40px] px-2 bg-slate-950 text-white text-lg">
        <label className="text-xl font-bold">Awesome Website</label>
      </header>
      <main className="container mx-auto relative">

        <div className="text-center">
          <h2 className="text-4xl mt-10 mb-10">Awesome Website</h2>
        </div>

        <div className="flex mb-10">
          <div className="w-1/3 flex flex-col">
            <label className="font-bold text-center bg-slate-950 text-white m-1">Products</label>
            <ul className="list-disc">
              <li>Electronics</li>
              <li>Cloathing</li>
            </ul>
          </div>
          <div className="w-1/3 flex flex-col">
            <label className="font-bold text-center bg-slate-950 text-white m-1">Services</label>
            <ul className="list-disc">
              <li>Repair</li>
              <li>Installation</li>
            </ul>
          </div>
          <div className="w-1/3 flex flex-col">
            <label className="font-bold text-center bg-slate-950 text-white m-1">Contact Us</label>
            <ul className="list-disc">
              <li>Phone: (999)999-9999</li>
              <li>Email: name@company.com</li>
            </ul>
          </div>
        </div>
      </main>
      <FloatingBot />
    </>
  )
}

export default App

```
Link: ()[]

### 2.1 - `src/components/floatingbot.tsx`

#### Code

##### 2.1.1 - The floating chatbot React component

```tsx
import { useState } from "react"
import { FaTrash } from "react-icons/fa"
import { IoCloseCircleSharp } from "react-icons/io5"
import { LuBot } from "react-icons/lu"
import { RiSendPlane2Fill } from "react-icons/ri"
import { IMessage } from "../types"
import { chatbotService } from "../service/chatbotservice"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"

const FloatingBot = () => {
    const [status, setStatus] = useState(false)
    const [input, setInput] = useState<string>("")
    const [messages, setMessages] = useState<IMessage[]>([
        { role: "system", content: "You are a helpful assistant." }
    ])
    const [processing, setProcessing] = useState<boolean>(false)

    const onReset = () => {
        setMessages([])
    }

    const Process = async () => {
        if (processing)
            return
        if (input === "") {
            alert("Please enter a message")
            return
        }
        setProcessing(true)
        const newMessages = [...messages, { role: "user", content: input }]
        setMessages(newMessages)
        const resp = await chatbotService(newMessages);
        const assistantMessage = resp.choices[0].message.content
        setMessages([...newMessages, { role: "assistant", content: assistantMessage }])
        setInput("")
        setProcessing(false)
    }

    return (
        <>
            {!status && <button className="fixed bottom-4 right-4 bg-slate-900 rounded-lg opacity-60 text-white text-5xl text-center" onClick={() => setStatus(!status)}>
                <LuBot />
            </button>}
            {status && <div className="fixed bottom-10 right-4 bg-slate-900 h-[calc(100vh-10%)] w-[400px] flex flex-col text-white opacity-90 rounded">
                <div className="flex p-1">
                    <div className="flex-grow"></div>
                    <button className="text-xl" onClick={() => setStatus(!status)}><IoCloseCircleSharp /></button>
                </div>
                <div className="h-[100%] bg-slate-950 overflow-auto p-2 flex flex-col text-black space-y-2">
                    {messages.map((message, index) => (<>
                        {(message.role != 'system') && <div key={index} className={"px-2 rounded-lg w-[90%] " + (message.role === 'user' ? 'ml-auto bg-blue-800 text-white' : 'bg-slate-800 text-white')}>

                            <Markdown remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
                        </div>}
                    </>))}
                </div>
                <div className="flex items-center m-1">
                    <textarea className="flex-grow outline-none resize-none text-black p-1" rows={5}
                        onChange={(e) => setInput(e.target.value)}
                        value={input}
                    />
                    <div className="flex flex-col text-lg">
                        <button className="bg-blue-600 p-2 text-white opacity-70"
                            onClick={Process}
                        ><RiSendPlane2Fill /></button>
                        <button className="bg-red-600 p-2 text-white opacity-70"
                            onClick={onReset}
                        ><FaTrash /></button>
                        <button className="bg-slate-900 p-2 text-white opacity-70"
                            onClick={() => setStatus(!status)}
                        ><IoCloseCircleSharp /></button>
                    </div>
                </div>
            </div>}
        </>
    )
}

export default FloatingBot
```
Link: ()[]