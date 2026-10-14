import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from typing import List, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# Load API keys from .env
load_dotenv()


import time

class ResilientGeminiChat:
    """Wraps Gemini with automatic exponential backoff retry and fallback models to handle 503 load spikes."""
    def __init__(self, api_key: str, models=None, temperature=0.2):
        self.api_key = api_key
        self.models = models or ["gemini-3.1-flash-lite", "gemini-2.5-pro", "gemini-flash-lite-latest"]
        self.temperature = temperature

    def _get_llm(self, model_name: str):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=self.temperature
        )

    def invoke(self, *args, **kwargs):
        last_err = None
        for model_name in self.models:
            for attempt in range(1, 4):
                try:
                    llm = self._get_llm(model_name)
                    return llm.invoke(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    err_msg = str(e)
                    if "503" in err_msg or "UNAVAILABLE" in err_msg or "demand" in err_msg or "10053" in err_msg:
                        print(f"⚠️ [{model_name}] busy (attempt {attempt}/3). Retrying in 2s...")
                        time.sleep(2)
                    else:
                        break
        raise last_err

    def with_structured_output(self, schema):
        outer = self
        class StructuredWrapper:
            def invoke(self, *args, **kwargs):
                last_err = None
                for model_name in outer.models:
                    for attempt in range(1, 4):
                        try:
                            llm = outer._get_llm(model_name)
                            structured = llm.with_structured_output(schema)
                            return structured.invoke(*args, **kwargs)
                        except Exception as e:
                            last_err = e
                            err_msg = str(e)
                            if "503" in err_msg or "UNAVAILABLE" in err_msg or "demand" in err_msg or "10053" in err_msg:
                                print(f"⚠️ [{model_name}] structured output busy (attempt {attempt}/3). Retrying in 2s...")
                                time.sleep(2)
                            else:
                                break
                raise last_err
        return StructuredWrapper()


def get_model():
    """Returns an LLM model instance based on available API keys (Gemini or OpenAI)."""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if gemini_key:
        print("🤖 Using Google Gemini with Resilient Multi-Model Fallback...")
        return ResilientGeminiChat(api_key=gemini_key, temperature=0.2)
    elif openai_key:
        from langchain_openai import ChatOpenAI
        print("🤖 Using OpenAI (gpt-4o-mini)...")
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_key,
            temperature=0.2
        )
    else:
        raise ValueError(
            "No API key found! Please define GEMINI_API_KEY or OPENAI_API_KEY in your .env file."
        )


def get_search_tool():
    """Returns Tavily search if TAVILY_API_KEY exists, otherwise falls back to DuckDuckGo (free, no key)."""
    tavily_key = os.getenv("TAVILY_API_KEY")

    if tavily_key:
        from langchain_community.tools.tavily_search import TavilySearchResults
        print("🔍 Web search powered by Tavily AI...")
        return TavilySearchResults(max_results=2)
    else:
        from langchain_community.tools import DuckDuckGoSearchRun
        print("🔍 Web search powered by DuckDuckGo (zero-config, free)...")
        ddg = DuckDuckGoSearchRun()

        class DuckDuckGoAdapter:
            def invoke(self, input_data):
                if isinstance(input_data, dict) and "query" in input_data:
                    q = input_data["query"]
                else:
                    q = str(input_data)
                try:
                    return ddg.invoke(q)
                except Exception as e:
                    return f"Search summary for '{q}': (Search lookup returned: {e})"

        return DuckDuckGoAdapter()


# ------------------ SCHEMA ------------------

class Queries(BaseModel):
    queries: List[str] = Field(description="Search queries list")


class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


# ------------------ AGENT ------------------

class MasterEssayAgent:
    def __init__(self, model, search_tool):
        self.model = model
        self.search = search_tool

    def plan_node(self, state: AgentState):
        prompt = "You are an expert technical writer. Create a clear, high-level outline for an article on the topic."
        res = self.model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=state["task"])
        ])
        return {"plan": res.content}

    def research_node(self, state: AgentState):
        prompt = "Generate 2 specific search queries to find real-world facts for this plan."

        structured_llm = self.model.with_structured_output(Queries)
        query_results = structured_llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=state["plan"])
        ])

        current_content = state.get("content", [])
        for q in query_results.queries:
            result = self.search.invoke({"query": q})
            current_content.append(str(result))

        return {"content": current_content}

    def generation_node(self, state: AgentState):
        prompt = "Write an insightful, well-structured, professional article synthesizing the outline and research."

        research_blob = "\n\n".join(state.get("content", []))
        msg = f"""TOPIC:
{state['task']}

OUTLINE:
{state['plan']}

RESEARCH FINDINGS:
{research_blob}
"""

        res = self.model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=msg)
        ])

        return {
            "draft": res.content,
            "revision_number": state.get("revision_number", 0) + 1
        }


# ------------------ GRAPH ------------------

def build_graph():
    model = get_model()
    search = get_search_tool()

    agent = MasterEssayAgent(model, search)

    workflow = StateGraph(AgentState)

    workflow.add_node("planner", agent.plan_node)
    workflow.add_node("researcher", agent.research_node)
    workflow.add_node("generator", agent.generation_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "generator")

    def router(state: AgentState):
        if state["revision_number"] > state["max_revisions"]:
            return "end"
        return "researcher"

    workflow.add_conditional_edges(
        "generator",
        router,
        {"researcher": "researcher", "end": END}
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ------------------ RUN ------------------

if __name__ == "__main__":
    app = build_graph()

    config = RunnableConfig(configurable={"thread_id": "loopcraft_run_01"})

    default_topic = "Future of Autonomous AI Agents in Software Development"
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        try:
            user_in = input(f"Enter research topic [press Enter for '{default_topic}']: ").strip()
            task = user_in if user_in else default_topic
        except EOFError:
            task = default_topic

    initial_input: AgentState = {
        "task": task,
        "plan": "",
        "draft": "",
        "critique": "",
        "content": [],
        "revision_number": 0,
        "max_revisions": 1,
    }

    print("\n" + "=" * 60)
    print("🚀 LoopCraft Agentic Workflow Started...")
    print(f"📌 Task: {initial_input['task']}")
    print("=" * 60)

    for event in app.stream(initial_input, config=config):
        node = list(event.keys())[0]
        data = event[node]
        print(f"\n👉 Completed Node: [{node.upper()}]")

        if node == "planner" and "plan" in data:
            plan_text = data["plan"]
            if isinstance(plan_text, list) and len(plan_text) > 0 and "text" in plan_text[0]:
                plan_text = plan_text[0]["text"]
            print("📝 [Generated Outline]:")
            print(str(plan_text).strip())

        elif node == "researcher" and "content" in data:
            sources = data.get("content", [])
            print(f"🔎 [Live Web Research]: Gathered {len(sources)} source snippets via search.")

        elif node == "generator" and "draft" in data:
            rev = data.get("revision_number", 1)
            print(f"✍️ [Drafting]: Finished revision #{rev}.")

    print("\n" + "=" * 60)
    print("🧠 Final Synthesized Draft:")
    print("=" * 60 + "\n")
    final_state = app.get_state(config)
    draft = final_state.values.get("draft", "")
    if isinstance(draft, list) and len(draft) > 0 and isinstance(draft[0], dict) and "text" in draft[0]:
        print(draft[0]["text"])
    else:
        print(draft)
    print("\n" + "=" * 60)
    print("✅ LoopCraft execution finished successfully!")
    print("=" * 60 + "\n")
