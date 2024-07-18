# RAG Pattern Fundamentals

#### Related

- [Improving RAG Results](https://genaitutor.am2703.com/?content=rag-results)
- [OpenAI Development Intro](https://genaitutor.am2703.com/?content=genai-intro)
- [OpenAI Development Intermediate](https://genaitutor.am2703.com/?content=genai-intermediate)

## 1 - RAG Pattern Fundamentals

### 1.1 - Overview

Retrieval-Augmented Generation (RAG) is an advanced technique in the field of artificial intelligence, particularly within natural language processing (NLP). It enhances the capabilities of large language models (LLMs) by combining them with a retrieval system that accesses an external knowledge base. This process allows the LLM to produce more accurate, relevant, and contextually appropriate responses.

Here's a breakdown of how RAG works:

1. **Ingestion**: This is a phase at which data is read from source as text, the text is broken into pieces, and the pieces together with their vector representation are stored in a vector database.

2. **Retrieval**: When a query is received, the RAG system first identifies relevant information from a vast database or knowledge base. This step is crucial as it ensures that the generation process is informed by the most pertinent and authoritative data available.

3. **Augmentation**: The retrieved information is then used to augment the generative capabilities of the LLM. By integrating this external data, the model can generate responses that are not solely based on its pre-existing knowledge, which may be outdated or limited.

4. **Generation**: Finally, the LLM uses the augmented data to construct a coherent and contextually relevant response to the query. This step leverages the model's ability to understand and manipulate language to produce a final output that aligns with the user's request.

The significance of RAG lies in its ability to address some of the inherent limitations of LLMs, such as the static nature of their training data, which can lead to outdated or inaccurate responses. By dynamically incorporating up-to-date information, RAG models can provide more precise answers and reduce the spread of misinformation.

RAG is particularly useful for applications that require high levels of accuracy and specificity, such as chatbots, question-answering systems, and content generation tools. Its ability to cross-reference authoritative knowledge sources makes it a powerful tool for creating AI systems that users can trust to provide reliable information.

### 1.2 - Ingestion Phase

In this phase the text from source documents, for example documents in an Azure storage account, is extracted. For each file, the text is broken into chunks, and the text chunk and the vector representation of the chunk is stored in a vector database.

By why is the text chunked? LLM models have a context window, and the context window is not infinite (for now). Context windo has been increasing from 4k to now LLMs with over 128k. Still, there need to be a though as to how much content the LLM windows can process and a consideration for the token counts including both the input and output counts.

### 1.3 - Retreival Phase

During this phase, a users asks a question. The question itself is embedded and vector database is queried using this vector. More advanced systems like Azure AI Search may use not only the vector but the keywords and something called semantic ranker. 

At this phase the user decides how many chunks should be retrieved based and what relevance. The best way to think about this, is to try to estimate how many chunks could be useful to answer the user's question, and control these results with both a limit and relevance.

If the system was based on straigh vector comparison, relevance relates to Cosine similarity. Cosine similarity calculates the cosine of the angle between the input vector (user's question) and the vectors stored in the vector database.

### 1.4 - Augmentation Phase

In this phase, the results from the retrieval phase are added to a final prompt. This prompt includes at least both the original question (the text), and the retrieved results (the chunks, not the embeddings).

### 1.5 - Generation Phase

In this phase, the system sends a prompt for completion that includes the orignal question and the retried information for completion grounding the prompt with the retrieved data.

## 2 - Semantic Kernel RAG Notebook

### 2.1 - Notebook explenation

In this guide RAG pattern is implemented with a C# interactive notebook using Semantic Kernel SDK using SQLite as a vector database. The notebook implements the phases discussed above.

#### Reference

- [Full Notebook Source Code](https://github.com/msalemor/sk-dev-training/blob/main/notebooks/sk-rag-pattern.ipynb)

#### Code

##### 2.1.1 - Ingestion

```C#
// Read the data
var jsonFileContents = File.ReadAllText("data/learnings.json");
var learnings = System.Text.Json.JsonSerializer.Deserialize<List<Learning>>(jsonFileContents);

// Keep a list of chunks
var chunks = new List<Chunk>();

// For each learning process the chunks
foreach(var learning in learnings)
{
    // Break the learnings into paragraphs
    var paragraphs = learning.Content.Split("\n\n");
    
    // For each paragraph create a chunk
    for(var i=0;i<paragraphs.Length;i++)
    {
        // Add the chunk to the list
        chunks.Add(new Chunk(learning.Id+"-"+(i+1),paragraphs[i],""));
    }
}

// Save the chunk and the embedding
foreach(var chunk in chunks)
{    
    await textMemory.SaveInformationAsync(MemoryCollectionName, id: chunk.Id, text: chunk.Text);
}
```

##### 2.1.2 - Retreival

```C#
//var query = await InteractiveKernel.GetInputAsync("What is your query?");
var question = "What scenario is FrontDoor good for?";

#pragma warning disable CS8618,IDE0009,CA1051,CA1050,CA1707,CA2007,VSTHRD111,CS1591,RCS1110,CA5394,SKEXP0001,SKEXP0002,SKEXP0003,SKEXP0004,SKEXP0010,SKEXP0011,SKEXP0012,SKEXP0020,SKEXP0021,SKEXP0022,SKEXP0023,SKEXP0024,SKEXP0025,SKEXP0026,SKEXP0027,SKEXP0028,SKEXP0029,SKEXP0030,SKEXP0031,SKEXP0032,SKEXP0040,SKEXP0041,SKEXP0042,SKEXP0050,SKEXP0051,SKEXP0052,SKEXP0053,SKEXP0054,SKEXP0055,SKEXP0060,SKEXP0061,SKEXP0101,SKEXP0102
IAsyncEnumerable<MemoryQueryResult> queryResults =
                textMemory.SearchAsync(MemoryCollectionName, question, limit: 3, minRelevanceScore: 0.77);
```

##### 2.1.3 - Augmentation

```C#
// Keep the text for the recalled memories
StringBuilder memoryText = new StringBuilder();

#pragma warning disable CS8618,IDE0009,CA1051,CA1050,CA1707,CA2007,VSTHRD111,CS1591,RCS1110,CA5394,SKEXP0001,SKEXP0002,SKEXP0003,SKEXP0004,SKEXP0010,SKEXP0011,SKEXP0012,SKEXP0020,SKEXP0021,SKEXP0022,SKEXP0023,SKEXP0024,SKEXP0025,SKEXP0026,SKEXP0027,SKEXP0028,SKEXP0029,SKEXP0030,SKEXP0031,SKEXP0032,SKEXP0040,SKEXP0041,SKEXP0042,SKEXP0050,SKEXP0051,SKEXP0052,SKEXP0053,SKEXP0054,SKEXP0055,SKEXP0060,SKEXP0061,SKEXP0101,SKEXP0102
await foreach (MemoryQueryResult r in queryResults)
{
    // Append the text
    memoryText.Append(r.Metadata.Text+"\n\n");
}

// Final augmented text
var promptContext = memoryText.ToString();
Console.WriteLine($"User:\n{question}\n\nNearest results:\n{promptContext}")
```

##### 2.1.4 - Generation

```C#
// Prepare the prompt
const string promptTemplate = "{{$input}}\n\nText:\n\"\"\"{{$context}}\n\"\"\"Use only the provided text.";
var excuseFunction = kernel.CreateFunctionFromPrompt(promptTemplate, new OpenAIPromptExecutionSettings() { MaxTokens = 100, Temperature = 0.4, TopP = 1 });

// Process the completion
var arguments = new KernelArguments()
        {
            ["input"] = question,
            ["context"] = promptContext
        };

var result = await kernel.InvokeAsync(excuseFunction, arguments);
Console.WriteLine(result);
```