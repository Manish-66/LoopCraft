# LoopCraft 🔁🧠
### Stateful, Cyclic & Human-Supervised AI Agent Framework

![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-latest-green?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-brightgreen?style=flat-square)
![Status](https://img.shields.io/badge/status-active-success?style=flat-square)

> **LoopCraft** moves beyond brittle, linear LLM chains into robust, production-grade **cyclic and stateful agentic systems**. It provides modular patterns for iterative self-reflection, persistent conversational checkpoints, agentic web search, and human-in-the-loop decision controls.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Architectural Capabilities](#-key-architectural-capabilities)
- [Architecture & Workflow](#-architecture--workflow)
- [Repository Structure](#-repository-structure)
- [Module Deep Dive](#-module-deep-dive)
- [Quickstart Guide](#-quickstart-guide)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🔍 Overview

Traditional LLM pipelines are unidirectional: `Input ➔ LLM ➔ Output`. However, complex real-world challenges require agents that can:
- **Plan and strategize** before generating answers.
- **Act and query tools** dynamically to fetch factual knowledge.
- **Self-reflect and critique** their own drafts in an iterative loop until criteria are met.
- **Pause and seek human supervision** before executing risky actions.
- **Maintain durable memory** across restarts and thread sessions.

**LoopCraft** demonstrates how to build these sophisticated cognitive patterns using **LangGraph** and **LangChain**.

---

## ⚡ Key Architectural Capabilities

### 1. 🔁 Cyclic Graphs & Iterative Reflection
Unlike static Directed Acyclic Graphs (DAGs), LoopCraft uses cyclic execution paths where nodes can loop back to previous states based on conditional routing logic. This enables self-correction, iterative drafting, and automated refinement loops.

### 2. 💾 Checkpointing & Multi-Session Persistence
Every step of the agent's graph is serialized into persistent storage (`SqliteSaver` / `MemorySaver`). Agents can pause, shut down, resume weeks later, or maintain hundreds of isolated conversation threads without data loss.

### 3. 🛑 Human-in-the-Loop Supervision
Using `interrupt_before` triggers, high-impact actions (such as publishing, external API writes, or tool invocations) are halted until an operator inspects or approves the pending state.

### 4. ⏳ Time Travel & State Forking
Because every step is saved as a distinct checkpoint, you can rewind the agent's timeline to any historical point, modify variables (e.g., rephrasing a query or changing state inputs), and branch execution into a new timeline.

### 5. 🌐 Agentic Search (Tavily AI)
Integrates specialized search APIs optimized specifically for LLMs, producing clean, hallucination-resistant, cited research content.

---

## 🏗 Architecture & Workflow

```
                   ┌──────────────────────────────────┐
                   │           User Request           │
                   └─────────────────┬────────────────┘
                                     │
                                     ▼
                   ┌──────────────────────────────────┐
                   │         [Planner Node]           │
                   │  Creates structured outline      │
                   └─────────────────┬────────────────┘
                                     │
                                     ▼
                   ┌──────────────────────────────────┐
                   │        [Researcher Node]         │
                   │  Generates queries & searches    │
                   └─────────────────┬────────────────┘
                                     │
                                     ▼
                   ┌──────────────────────────────────┐
                   │        [Generator Node]          │
                   │  Synthesizes draft with sources  │
                   └─────────────────┬────────────────┘
                                     │
                                     ▼
                   ┌──────────────────────────────────┐
                   │        [Reflector Node]          │
                   │  Reviews & grades quality        │
                   └─────────────────┬────────────────┘
                                     │
                             (Conditional Edge)
                             Need Revisions?
                                   /   \
                         YES      /     \    NO (Max Revisions Met)
                                 ▼       ▼
                       (Loop back)       [END Node / Final Response]
```

---

## 📁 Repository Structure

```
LoopCraft/
│
├── 01_Basics/
│   ├── langgraph_basics.py                 # Core nodes, edges, states, and graph compilation
│   └── simple_react_agent.py               # ReAct (Reason + Act) loop built from scratch
│
├── 02_State_Management/
│   └── persistence_and_streaming.py        # SqliteSaver checkpoints, threads, & token streaming
│
├── 03_Tool_Integration/
│   └── agentic_search_tavily.py            # Binding external search tools to graph agents
│
├── 04_Human_in_the_loop/
│   └── human_approval_and_time_travel.py   # Interrupt gates, state editing & timeline rewinding
│
├── 05_Use_Cases/
│   └── essay_writer.py                     # Full multi-node essay writer with self-critique loop
│
├── app.py                                  # Modular runner showcasing the end-to-end agent
├── requirements.txt                        # Clean project dependencies
├── .env.example                            # Template for API keys
├── LICENCE                                 # MIT License
└── README.md                               # Project documentation
```

---

## 📖 Module Deep Dive

| Directory | Core Script | What It Demonstrates |
|---|---|---|
| `01_Basics` | `langgraph_basics.py` | Fundamental LangGraph building blocks: TypedDict states, nodes, and conditional edges. |
| `01_Basics` | `simple_react_agent.py` | Building an end-to-end ReAct cycle (thought $\rightarrow$ action $\rightarrow$ observation) without blackbox frameworks. |
| `02_State_Management` | `persistence_and_streaming.py` | Checkpointing conversation state with SQLite, thread isolation, and real-time streaming. |
| `03_Tool_Integration` | `agentic_search_tavily.py` | Seamless integration of Tavily Search API with model tool calling. |
| `04_Human_in_the_loop` | `human_approval_and_time_travel.py` | Breakpoints for human review before execution and state modifications mid-flight. |
| `05_Use_Cases` | `essay_writer.py` | Production-grade multi-agent workflow: Outline $\rightarrow$ Research $\rightarrow$ Draft $\rightarrow$ Critique $\rightarrow$ Revision. |

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- OpenAI API Key
- Tavily Search API Key ([Get a free key](https://tavily.com))

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Manish-66/LoopCraft.git
cd LoopCraft

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Populate your `.env` file with your credentials:
```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 4. Running the Examples

Run the complete end-to-end agent:
```bash
python app.py
```

Run individual module components:
```bash
# 1. Basic ReAct Agent
python 01_Basics/simple_react_agent.py

# 2. Persistence & Multi-Thread Checkpointing
python 02_State_Management/persistence_and_streaming.py

# 3. Search Tool Integration
python 03_Tool_Integration/agentic_search_tavily.py

# 4. Human-in-the-Loop & Time Travel
python 04_Human_in_the_loop/human_approval_and_time_travel.py

# 5. Autonomous Essay Writer
python 05_Use_Cases/essay_writer.py
```

---

## 🛠 Tech Stack

- **Orchestration:** [LangGraph](https://github.com/langchain-ai/langgraph)
- **Agent Ecosystem:** [LangChain](https://github.com/langchain-ai/langchain)
- **Language Models:** OpenAI GPT-4o / GPT-4o-mini
- **Search Tooling:** [Tavily Search API](https://tavily.com)
- **Persistence Engine:** SQLite & In-Memory Checkpointers

---

## 📄 License

This project is licensed under the MIT License — see the [LICENCE](LICENCE) file for details.

---

<div align="center">
  Crafted by <a href="https://github.com/Manish-66"><strong>Manish Kumar</strong></a>
</div>
