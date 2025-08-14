# Agentic AI - Foundational Concepts

## Overview

## GenAI vs Agentic AI

### Key Differences

| Dimension | Generative AI (Baseline LLM) | Agentic AI (LLM + Agency Layer) |
|----------|------------------------------|----------------------------------|
| Core Capability | Produces text, code, images from prompts | Pursues goals through iterative perception–planning–action |
| Control Loop | Single stateless inference | Multi-step loop with feedback and adaptive decisions |
| Autonomy | Reactive: responds when called | Proactive: can decide next steps, branch, retry |
| Goal Handling | Implicit in prompt | Explicit goals, sub-goal decomposition |
| Planning | Minimal (in-context reasoning only) | Structured planners (task graphs, chains, trees) |
| Memory | Ephemeral context window | Layered memory (short-term, long-term, vector stores) |
| Tool Use | Possible if manually scaffolded | Native orchestration of tools, APIs, external systems |
| Environment Interaction | None beyond output text | Reads/writes files, calls APIs, schedules tasks, triggers workflows |
| State Management | Caller responsibility | Internal state machine / world model maintenance |
| Error Recovery | Relies on new prompt | Self-reflection, retries, corrective strategies |
| Evaluation | External (human or wrapper) | Embedded self-critique, quality gates, guards |
| Adaptation | Few-shot prompt tweaks | Ongoing adaptation via memory and feedback loops |
| Safety Surface | Narrow (prompt injection, hallucination) | Broader (tool misuse, loops, escalation risks) |
| Observability | Single output to log | Traces: steps, decisions, tool calls, intermediate thoughts |
| Deployment | API call integration | Orchestrated runtimes / agent frameworks (ReAct, AutoGen, LangChain, custom) |
| Example Use | Draft an email, summarize text | Continuous research assistant, workflow automation, multi-tool ops |
| When Preferred | Simple, bounded, single-turn tasks | Complex, evolving, multi-step, autonomous tasks |

### Summary

Generative AI answers; Agentic AI pursues. Generative = content synthesis. Agentic = goal-directed orchestration combining reasoning, planning, memory, and tool execution over time.

### Selection Guidance

Choose baseline generative AI when:

- Task is single-shot, well-scoped, low risk.
- Deterministic orchestration preferred.
- Latency and cost must be minimal.

Choose Agentic AI when:

- Tasks require multi-step decomposition.
- External systems/tools must be coordinated.
- Continuous improvement, monitoring, or adaptation are needed.
- Human oversight wants structured traceability.

### Design Considerations for Agentic Systems

- Explicit goal and success criteria schema
- Planner + executor separation
- Memory tiering (context vs. persistent)
- Tool contract validation and sandboxing
- Loop guards (max depth, time, budget)
- Observability: step logs, metrics, lineage
- Safety filters pre/post tool invocation

### Common Pitfalls Moving From Generative to Agentic

- Unbounded recursion or task explosion
- Tool hallucination (calling non-existent capabilities)
- Over-trusting self-evaluation
- Memory bloat reducing relevance
- Latency and cost creep without budgeting

### Minimal Mental Model

Agentic AI = (LLM reasoning) + (Planner) + (Memory) + (Tooling) + (Control Loop + Policies) + (Observability & Safety)

Keep early prototypes shallow: start with explicit task lists, then layer adaptive planning, then memory, then autonomous triggers.

## Agentic AI Frameworks

### Categories

- General-purpose orchestration SDKs
- Memory / retrieval–centric stacks
- Multi-agent conversation & collaboration
- Hosted/managed agent platforms
- Visual or low-code builders
- Patterns (reference implementations) rather than full frameworks

### Notable Frameworks & Platforms

1. LangChain (Python/JS): Chain & agent abstractions (tools, memory, retrievers), large ecosystem.
2. LlamaIndex: Data ingestion + retrieval + agent routing; strong document querying focus.
3. Semantic Kernel (C# / Python / JS): Pluggable planners, skills (tools), memories, orchestration w/ .NET friendliness.
4. Microsoft AutoGen: Multi-agent conversation graphs, role definitions, tool execution, human-in-the-loop hooks.
5. OpenAI Assistants API: Hosted threads, tools (code interpreter, retrieval, functions), state managed by provider.
6. CrewAI: Team-based agent roles (pilot, researcher, reviewer) with task delegation.
7. Haystack Agents: Retrieval-first pipeline with tool nodes; elastic search / RAG integration.
8. Dify: GUI + API for building agents, flows, datasets, evaluation dashboards.
9. Flowise: Visual node editor for LangChain-style graphs; quick prototyping.
10. AWS Bedrock Agents: Managed orchestration, tool (action group) invocation, grounding.
11. Azure AI (Agent / Functions + Semantic Kernel combo): Enterprise integration, governance, vector + search services.
12. Google Vertex AI (Agents / Extensions): Tool and API integration, enterprise data connectors.
13. OpenAI Swarm (experimental): Lightweight multi-agent coordination primitives.
14. AutoGPT / BabyAGI (early prototypes): Illustrate autonomous loops; good for learning pitfalls, not production.
15. ReAct / Plan-and-Execute (patterns): Prompt-level templates enabling reasoning + action; underpin many frameworks.
16. CAMEL / Role-Playing Agents: Structured multi-agent role prompting pattern.
17. AgentVerse / MetaGPT: Opinionated multi-agent coordination layers.
18. Dagger / Airflow + LLM Operators: Workflow engines extended with LLM steps (pattern for existing orchestrators).

### Quick Comparison (Indicative)

| Name | Maturity | Style | Strengths | Typical Use |
|------|----------|-------|-----------|-------------|
| LangChain | High | Code SDK | Ecosystem breadth | Complex tool chains |
| LlamaIndex | High | Code SDK | Retrieval + routing | RAG-centric agents |
| Semantic Kernel | Medium | Code SDK | .NET integration, planners | Enterprise apps |
| AutoGen | Medium | Multi-agent convo | Conversation graphs | Collaborative agents |
| OpenAI Assistants | High | Managed API | Hosted state + tools | Fast prod integration |
| CrewAI | Medium | Role orchestration | Division of labor | Research/workflow teams |
| Dify | Medium | Low-code | UI + eval + deploy | Rapid prototyping |
| Flowise | Medium | Visual | Diagram-based design | Demos / internal tools |
| Bedrock Agents | Emerging | Managed | AWS integration | Enterprise workloads |
| Vertex AI Agents | Emerging | Managed | GCP data/services | Cloud-aligned solutions |

### Selection Guidance

- Need fastest path + hosted state: OpenAI Assistants (or cloud-managed agent service).
- Heavy retrieval / data graph focus: LlamaIndex.
- Rich tool chaining & community integrations: LangChain.
- Enterprise .NET or Azure stack: Semantic Kernel (+ Azure AI services).
- Multi-agent conversational workflows: AutoGen or CrewAI.
- Low-code + monitoring out-of-box: Dify or Flowise.
- Strict cloud governance & IAM: Bedrock Agents / Vertex AI / Azure.

### Evaluation Criteria

- Tooling: Ease of defining, validating, sandboxing tools.
- Planning: Built-in planners vs. manual chain design.
- Memory: Native vector stores vs. pluggable; summarization support.
- Observability: Step traces, token metrics, cost tracking.
- Safety: Guardrails, content filters, action approval checkpoints.
- Extensibility: Custom planners, runtime adapters, event hooks.
- Deployment: Serverless, container, managed, edge feasibility.
- Ecosystem: Plugins, templates, community velocity.

### Practical Strategy

1. Start with a pattern (ReAct / function-calling) inside your existing stack.
2. Graduate to an SDK (LangChain, LlamaIndex, Semantic Kernel) once tool/memory complexity grows.
3. Introduce multi-agent coordination (AutoGen, CrewAI) only if clear role separation adds value.
4. Consider managed platforms (Assistants API, Bedrock, Vertex) for faster compliance, scaling, and fewer ops.
5. Layer observability (tracing, cost budgets) and safety gates early to avoid opaque autonomous behavior.

### Common Integration Add-ons

- Vector DBs: Pinecone, Weaviate, Chroma, Azure AI Search.
- Orchestration / Workflow: Airflow, Temporal, Prefect, Dagster wrapping agent steps.
- Monitoring: Langfuse, Helicone, Weights & Biases (LLM tracing), OpenTelemetry exporters.
- Guardrails: Guardrails AI, Llama Guard, model provider safety filters.

### When to Build Your Own

- Need deterministic auditability beyond existing abstractions.
- Must embed into stringent latency-critical pipelines.
- Require proprietary planning algorithms or domain-specific tool governance.

Keep scope minimal initially: single-agent + explicit task list + vetted tools before layering autonomous triggers or recursive planners. Measure step cost, success rate, and failure modes prior to scaling complexity.
