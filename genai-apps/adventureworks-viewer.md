# Adventureworks AI Viewer

#### Related

- [OpenAI Development Intro](https://genaitutor.am2703.com/?content=genai-intro)
- [OpenAI Development Intermediate](https://genaitutor.am2703.com/?content=genai-intermediate)
- [LLM Conversation History](https://genaitutor.am2703.com/?content=llm-conversation-history)

## 1.0 - Adventureworks AI viewer

### 1.1 - Overview

![Screen shot](https://github.com/msalemor/ai-tutors/blob/main/genai-apps/images/adventureworks-ai-viewer.png?raw=true)

Advetureworks viewer is a showcase application desgined to add AI at different levels of intelligence including:

1. **No AI:** The application with no AI. This simulates a legacy application where a user can lookup information related to customer, products and orders from the Adventureworks database.
2. **Chatbot:** Chatbot that can answer quetions about top customers and products.
3. **SQLbot:** A bot that can generate SQL and display the results on the grid.
4. **Assistans API Bot:** Assistants API bot that can perform complex analysis on top products and services and generate bar and charts.
5. **Multi-agent Bot:** A bot that can recognize intent and leverage the other bots to answer.

## 2.0 - The application

### 2.1 - Azure SQL Database (Adventurework sample)

This application relies on the Azure SQL sample database. To get started make sure that you deploy an Azure SQL database and during provisioning deploy it with the Adventurework sample database. Once the database has been deployed, execute the following DDL to add the required SQL views.

#### Code

##### 2.1.1 - Create there required views

```sql
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE view [SalesLT].[vCustomers]
as
    select A.*, B.Orders, B.TotalDue
    from
        (
select a.CustomerID, A.LastName, a.FirstName, a.EmailAddress, a.SalesPerson, c.City, c.StateProvince, c.CountryRegion
        from SalesLt.Customer A
            left outer join
            SalesLT.CustomerAddress B
            on b.CustomerID=a.CustomerID and b.AddressType='main office'
            left outer join
            SalesLT.Address C
            on c.AddressID=B.AddressID
) A
        left outer join
        (select CustomerID, count(*) Orders, sum(TotalDue) TotalDue
        from SalesLT.SalesOrderHeader
        group by CustomerID) B
        on b.CustomerID=a.CustomerID
GO

CREATE VIEW [SalesLT].[vTopCustomers]
AS
    select A.CustomerID, b.LastName, B.FirstName, B.EmailAddress, b.SalesPerson, A.Total, D.City, D.StateProvince, D.CountryRegion
    from (
select A.CustomerID, sum(a.TotalDue) Total
        from SalesLT.SalesOrderHeader A
        GROUP by a.CustomerID
) A
        inner join
        SalesLT.Customer B
        on A.CustomerID = b.CustomerID
        inner JOIN
        SalesLT.CustomerAddress C
        on A.CustomerID=c.CustomerID and c.AddressType='main office'
        inner JOIN
        SalesLT.Address D
        on c.AddressID=d.AddressID
GO

CREATE VIEW [SalesLT].[vOrderDetails]
AS
    select b.CustomerID, A.SalesOrderID, A.ProductID, D.Name Category, E.Name Model, G.[Description], A.OrderQty, A.UnitPrice, A.UnitPriceDiscount, A.LineTotal
    from SalesLT.SalesOrderDetail A
        inner join
        SalesLT.SalesOrderHeader B
        on A.SalesOrderID = b.SalesOrderID
        inner JOIN
        SalesLT.Product C
        on A.ProductID=c.ProductID
        inner join
        SalesLT.ProductCategory D
        on c.ProductCategoryID=d.ProductCategoryID
        inner join
        SalesLT.ProductModel E
        on c.ProductModelID=e.ProductModelID
        INNER join
        SalesLT.ProductModelProductDescription F
        on f.ProductModelID=e.ProductModelID and f.Culture='en'
        inner join
        SalesLT.ProductDescription G
        on g.ProductDescriptionID = f.ProductDescriptionID
GO

CREATE VIEW [SalesLT].[vTopProductsSold]
AS
    select ProductId, category, model, [description], sum(orderqty) TotalQty
    from SalesLT.vOrderDetails
    group by productid,category,model,[description]
GO
```

### 2.1 - Backend

The backend Python Fast API. It exposes the basic application functionality including the endpoints to POST messages to the different bots.


#### Code

##### 2.1.1 - Fast API Python backend

```python
import logging
import os
import asyncio
import requests
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from openai import AzureOpenAI
from kcvstore import KCVStore
from agents import AgentSettings, AgentRegistration, AgentProxy, AssistantAgent, GPTAgent, SQLAgent, RAGAgentAISearch, SQLAgent
from agents.Models import ChatRequest

import database as rep
import dotenv
import asyncio


# region: FastAPI App
app = FastAPI(openapi_url=OPENAPI_URL,
              title="AdventureWorks API", version="0.1.0")
# endregion

# region: FastAPI CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# endregion

# region: FastAPI APIs
@app.post('/api/chatbot')
def chatbot(request: ChatRequest):
    return gpt_agent.process('user', 'user', request.input, context=rep.get_top_customers_csv_as_text()+rep.get_top_products_csv_text())

@app.post('/api/sqlbot')
def sqlbot(request: ChatRequest):
    results = sql_agent.process(
        'user', 'user', request.input, context=rep.sql_schema)
    # Find the assistant message
    for result in results:
        if result.role == "assistant":
            sql_statement = result.content
            row_and_cols = rep.sql_executor(sql_statement)
            columns = row_and_cols['columns']
            rows = row_and_cols['rows']
            result.columns = columns
            result.rows = rows
    return results

@app.post('/api/rag')
def ragbot(request: ChatRequest):
    return rag_agent.process('user', 'user', request.input, context=rep.sql_schema)

@app.get("/api/status")
def get_app_status():
    total = rep.get_db_status() + rep.get_files_status()

    system_down = ""
    if rep.get_db_status() == 0:
        system_down += "Db"
    if rep.get_files_status() == 0:
        system_down += "Files"

    if total == 2:
        return {"status": "Online", "total": total}
    elif total == 1:
        return {"status": f"{system_down} degraded", "total": total}
    elif total == 0:
        return {"status": "Offline", "total": total}
    else:
        return {"status": "Unknown", "total": total}

@app.post('/api/multiagent')
def chatbot(request: ChatRequest):
    results = proxy.process(request.user_name, request.user_id, request.input)
    for result in results:
        if result.role == "assistant":
            sql_statement = result.content
            row_and_cols = rep.sql_executor(sql_statement)
            columns = row_and_cols['columns']
            rows = row_and_cols['rows']
            result.columns = columns
            result.rows = rows
    return results

# endregion
```

### 2.2 - React Frontend

The frontend allows the user to navigate the UI and leverage the differenet bots.

#### Code

##### 2.2.1 - React Frontend

```typescript
const Process = async () => {
        /* Process the input on the backend */
        if (processing) return
        setProcessing(true)

        try {
            const payload = { input: input }
            let URL = URL_BASE
            let msgs: IMessage[] = messages

            if (settings.mode === Mode.Chatbot) {
                URL = URL_BASE + '/api/chatbot'
            } else if (settings.mode === Mode.SqlBot) {
                URL = URL_BASE + '/api/sqlbot'
                // } else if (settings.mode === Mode.RAG) {
                //     URL = URL_BASE + '/api/rag'
            } else if (settings.mode === Mode.Assistant) {
                URL = URL_BASE + '/api/assistants'
            } else if (settings.mode === Mode.RAG) {
                URL = URL_BASE + '/api/rag'
            } else if (settings.mode === Mode.MultiAgent) {
                URL = URL_BASE + '/api/multiagent'
            }

            if (URL === URL_BASE)
                throw new Error('Invalid mode')

            const resp = await axios.post<IResponse[]>(URL, payload)
            const data = resp.data

            data.forEach((msg: IResponse) => {
                msgs.push({ role: msg.role, content: msg.content, imageUrl: null, mode: (msg.role === "assistant" ? settings.mode : null) })
                if ((settings.mode === Mode.SqlBot || settings.mode === Mode.MultiAgent) && msg.role === 'assistant') {
                    if (msg.rows) {
                        console.info(msg.rows)
                        if (msg.rows.length > 0) {
                            setGridColsRow({ columns: msg.columns, rows: msg.rows })
                            msgs.push({ role: 'assistant', content: 'Please check the grid for the answer', imageUrl: null, mode: Mode.SqlBot })
                        } else {
                            setGridColsRow({ columns: [], rows: [] })
                        }

                    }
                }
            })

        }
        catch (error) {
            console.error(error)
        }
        finally {
            // Scroll the div to the bottom
            let messagesArea: HTMLElement | null = document.getElementById('messages')
            if (messagesArea) {
                //messagesArea.scrollIntoView({ block: "start" })
                messagesArea.scrollTo({
                    top: messagesArea.scrollHeight,
                    behavior: "smooth",
                });
            }
            setInput('')
            setProcessing(false)
        }
    }
```
