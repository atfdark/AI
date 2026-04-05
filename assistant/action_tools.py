"""Action wrappers exposed as tools for the Jarvis agent runtime."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from .tool_registry import Tool


def create_action_tools(
    actions,
    memory_store=None,
    file_search_engine=None,
    qa_callback=None,
    knowledge_brain=None,
    corpus_manager=None,
) -> List[Tool]:
    """Create tool wrappers from existing Actions methods."""

    def open_app(app_name: str) -> Dict[str, Any]:
        success = actions.launch_app(app_name)
        return {"success": bool(success), "message": f"Opened {app_name}" if success else f"Failed to open {app_name}"}

    def open_url(url: str) -> Dict[str, Any]:
        success = actions.open_url(url)
        return {"success": bool(success), "message": f"Opened {url}" if success else f"Failed to open {url}"}

    def search_web(query: str) -> Dict[str, Any]:
        summary = actions.perform_search(query)
        return {
            "success": bool(summary),
            "summary": summary,
            "message": summary or f"No summary found for {query}",
        }

    def take_screenshot() -> Dict[str, Any]:
        path = actions.take_screenshot()
        return {"success": bool(path), "path": path, "message": f"Screenshot saved to {path}" if path else "Failed to take screenshot"}

    def volume_up(steps: int = 2) -> Dict[str, Any]:
        actions.volume_up(steps=steps)
        return {"success": True, "message": f"Volume increased by {steps} steps"}

    def volume_down(steps: int = 2) -> Dict[str, Any]:
        actions.volume_down(steps=steps)
        return {"success": True, "message": f"Volume decreased by {steps} steps"}

    def get_weather(location: str) -> Dict[str, Any]:
        response = actions.get_weather(location)
        return {"success": bool(response), "message": response or f"Weather unavailable for {location}"}

    def wikipedia_summary(topic: str) -> Dict[str, Any]:
        response = actions.get_wikipedia_summary(topic)
        return {"success": bool(response), "message": response or f"No Wikipedia summary for {topic}"}

    def fetch_news() -> Dict[str, Any]:
        response = actions.fetch_news()
        return {"success": bool(response), "message": response or "Could not fetch news"}

    def create_todo(list_name: str) -> Dict[str, Any]:
        success = actions.create_todo_list(list_name)
        return {"success": bool(success), "message": f"Created todo list '{list_name}'" if success else f"Could not create list '{list_name}'"}

    def add_todo_task(list_name: str, task: str) -> Dict[str, Any]:
        success = actions.add_todo_task(list_name, task)
        return {"success": bool(success), "message": f"Added task to '{list_name}'" if success else f"Could not add task to '{list_name}'"}

    def list_todos() -> Dict[str, Any]:
        todos = actions.get_todo_lists()
        names = list(todos.keys())
        return {"success": True, "lists": names, "message": "Todo lists: " + ", ".join(names) if names else "No todo lists found"}

    def search_files(query: str, top_k: int = 5) -> Dict[str, Any]:
        if not file_search_engine:
            return {"success": False, "matches": [], "message": "Semantic file search is unavailable"}
        matches = file_search_engine.search(query=query, top_k=top_k)
        first_path = matches[0]["path"] if matches else ""
        if not matches:
            return {"success": False, "matches": [], "message": f"No files matched: {query}"}
        return {
            "success": True,
            "matches": matches,
            "first_path": first_path,
            "message": f"Found {len(matches)} file matches for '{query}'",
        }

    def open_path(path: str) -> Dict[str, Any]:
        if hasattr(actions, "open_path"):
            success = bool(actions.open_path(path))
            return {"success": success, "message": f"Opened path: {path}" if success else f"Could not open path: {path}"}

        expanded = os.path.expandvars(os.path.expanduser(path))
        success = os.path.exists(expanded)
        return {"success": success, "message": f"Path exists: {expanded}" if success else f"Path not found: {expanded}"}

    def remember_fact(text: str, category: str = "preference") -> Dict[str, Any]:
        if not memory_store:
            return {"success": False, "message": "Memory store unavailable"}
        memory_id = memory_store.remember(text=text, category=category, source="user")
        return {
            "success": memory_id > 0,
            "memory_id": memory_id,
            "message": "I will remember that." if memory_id > 0 else "I could not store that memory.",
        }

    def recall_memory(query: str, top_k: int = 5) -> Dict[str, Any]:
        if not memory_store:
            return {"success": False, "matches": [], "message": "Memory store unavailable"}
        matches = memory_store.search(query=query, top_k=top_k)
        if not matches:
            return {"success": False, "matches": [], "message": "No relevant memory found"}
        summary = "; ".join(item["text"] for item in matches[:3])
        return {
            "success": True,
            "matches": matches,
            "message": f"I found related memories: {summary}",
        }

    def answer_offline(question: str) -> Dict[str, Any]:
        if not qa_callback:
            return {"success": False, "message": "Offline QA is unavailable"}
        answer = qa_callback(question)
        return {"success": bool(answer), "message": answer or "I could not generate an offline answer."}

    def ingest_knowledge(path: str, max_files: int = 200) -> Dict[str, Any]:
        if not knowledge_brain:
            return {"success": False, "message": "Knowledge ingestion is unavailable"}

        expanded = os.path.expandvars(os.path.expanduser(path))
        if os.path.isdir(expanded):
            result = knowledge_brain.ingest_directory(root_path=expanded, max_files=max_files)
            success = bool(result.get("files_ingested", 0) > 0)
            return {
                "success": success,
                "result": result,
                "message": (
                    f"Ingested {result.get('files_ingested', 0)} files and {result.get('chunks', 0)} chunks"
                    if success
                    else "No knowledge files were ingested"
                ),
            }

        result = knowledge_brain.ingest_file(expanded)
        success = bool(result.get("success", False))
        return {
            "success": success,
            "result": result,
            "message": (
                f"Ingested {result.get('chunks', 0)} chunks from {os.path.basename(expanded)}"
                if success
                else "Knowledge file ingestion failed"
            ),
        }

    def knowledge_stats() -> Dict[str, Any]:
        if not knowledge_brain:
            return {"success": False, "message": "Knowledge brain unavailable"}
        stats = knowledge_brain.stats()
        registry_stats = corpus_manager.stats() if corpus_manager else {}
        return {
            "success": True,
            "stats": stats,
            "registry": registry_stats,
            "message": (
                f"Knowledge sources: {stats.get('sources', 0)}, "
                f"chunks: {stats.get('chunks', 0)}, avg trust: {stats.get('avg_trust', 0.0):.2f}, "
                f"registered corpora: {registry_stats.get('registered_sources', 0)}"
            ),
        }

    def knowledge_catalog(bundle: str = "starter") -> Dict[str, Any]:
        if not corpus_manager:
            return {"success": False, "message": "Knowledge corpus manager unavailable"}
        catalog = corpus_manager.catalog(bundle=bundle)
        return {
            "success": True,
            "bundle": bundle,
            "datasets": catalog,
            "message": f"Catalog bundle '{bundle}' has {len(catalog)} datasets",
        }

    def knowledge_plan(bundle: str = "starter", output_path: str = "") -> Dict[str, Any]:
        if not corpus_manager:
            return {"success": False, "message": "Knowledge corpus manager unavailable"}
        plan = corpus_manager.create_plan(bundle=bundle, output_path=output_path or None)
        return {
            "success": True,
            "plan": plan,
            "message": f"Knowledge plan created at {plan.get('plan_path', '')}",
        }

    def register_knowledge_source(dataset_id: str, path: str, notes: str = "") -> Dict[str, Any]:
        if not corpus_manager:
            return {"success": False, "message": "Knowledge corpus manager unavailable"}
        try:
            item = corpus_manager.register_local_source(dataset_id=dataset_id, local_path=path, notes=notes)
            return {
                "success": True,
                "source": item,
                "message": f"Registered {item.get('dataset_id')} -> {item.get('path')}",
            }
        except FileNotFoundError:
            return {"success": False, "message": f"Path not found: {path}"}
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    def ingest_registered_knowledge(max_files_per_dataset: int = 2000) -> Dict[str, Any]:
        if not corpus_manager or not knowledge_brain:
            return {"success": False, "message": "Knowledge ingestion infrastructure unavailable"}
        result = corpus_manager.ingest_registered(
            knowledge_brain=knowledge_brain,
            max_files_per_dataset=max_files_per_dataset,
        )
        return {
            "success": bool(result.get("success", False)),
            "result": result,
            "message": (
                f"Ingested {result.get('ingested_sources', 0)} registered sources, "
                f"files: {result.get('files_ingested', 0)}, chunks: {result.get('chunks', 0)}"
            ),
        }

    def list_registered_knowledge() -> Dict[str, Any]:
        if not corpus_manager:
            return {"success": False, "message": "Knowledge corpus manager unavailable"}
        items = corpus_manager.list_registered()
        return {
            "success": True,
            "sources": items,
            "message": f"Registered knowledge sources: {len(items)}",
        }

    return [
        Tool(
            name="open_app",
            description="Open an installed application by name.",
            parameters={"app_name": "str"},
            required=["app_name"],
            handler=open_app,
            safety_level="low",
        ),
        Tool(
            name="open_url",
            description="Open a URL in the default browser.",
            parameters={"url": "str"},
            required=["url"],
            handler=open_url,
            safety_level="low",
        ),
        Tool(
            name="search_web",
            description="Search the web and return a concise summary.",
            parameters={"query": "str"},
            required=["query"],
            handler=search_web,
            safety_level="low",
        ),
        Tool(
            name="take_screenshot",
            description="Capture a screenshot and save it to disk.",
            parameters={},
            required=[],
            handler=take_screenshot,
            safety_level="low",
        ),
        Tool(
            name="volume_up",
            description="Increase system volume.",
            parameters={"steps": "int"},
            required=[],
            handler=volume_up,
            safety_level="low",
        ),
        Tool(
            name="volume_down",
            description="Decrease system volume.",
            parameters={"steps": "int"},
            required=[],
            handler=volume_down,
            safety_level="low",
        ),
        Tool(
            name="get_weather",
            description="Get weather information for a location.",
            parameters={"location": "str"},
            required=["location"],
            handler=get_weather,
            safety_level="low",
        ),
        Tool(
            name="wikipedia_summary",
            description="Fetch a Wikipedia summary for a topic.",
            parameters={"topic": "str"},
            required=["topic"],
            handler=wikipedia_summary,
            safety_level="low",
        ),
        Tool(
            name="fetch_news",
            description="Fetch top headlines.",
            parameters={},
            required=[],
            handler=fetch_news,
            safety_level="low",
        ),
        Tool(
            name="create_todo",
            description="Create a todo list by name.",
            parameters={"list_name": "str"},
            required=["list_name"],
            handler=create_todo,
            safety_level="medium",
        ),
        Tool(
            name="add_todo_task",
            description="Add a task to an existing todo list.",
            parameters={"list_name": "str", "task": "str"},
            required=["list_name", "task"],
            handler=add_todo_task,
            safety_level="medium",
        ),
        Tool(
            name="list_todos",
            description="List all todo lists.",
            parameters={},
            required=[],
            handler=list_todos,
            safety_level="low",
        ),
        Tool(
            name="search_files",
            description="Search files semantically by query and return best matches.",
            parameters={"query": "str", "top_k": "int"},
            required=["query"],
            handler=search_files,
            safety_level="low",
        ),
        Tool(
            name="open_path",
            description="Open a local file or folder path.",
            parameters={"path": "str"},
            required=["path"],
            handler=open_path,
            safety_level="medium",
        ),
        Tool(
            name="remember_fact",
            description="Store a persistent user fact or preference.",
            parameters={"text": "str", "category": "str"},
            required=["text"],
            handler=remember_fact,
            safety_level="low",
        ),
        Tool(
            name="recall_memory",
            description="Retrieve relevant long-term memories for a query.",
            parameters={"query": "str", "top_k": "int"},
            required=["query"],
            handler=recall_memory,
            safety_level="low",
        ),
        Tool(
            name="answer_offline",
            description="Answer a factual question using local model and memory context.",
            parameters={"question": "str"},
            required=["question"],
            handler=answer_offline,
            safety_level="low",
        ),
        Tool(
            name="ingest_knowledge",
            description="Ingest local files or folders into the offline knowledge base.",
            parameters={"path": "str", "max_files": "int"},
            required=["path"],
            handler=ingest_knowledge,
            safety_level="medium",
        ),
        Tool(
            name="knowledge_stats",
            description="Return offline knowledge base statistics.",
            parameters={},
            required=[],
            handler=knowledge_stats,
            safety_level="low",
        ),
        Tool(
            name="knowledge_catalog",
            description="List curated datasets for building the offline knowledge corpus.",
            parameters={"bundle": "str"},
            required=[],
            handler=knowledge_catalog,
            safety_level="low",
        ),
        Tool(
            name="knowledge_plan",
            description="Generate a dataset acquisition plan file for a bundle.",
            parameters={"bundle": "str", "output_path": "str"},
            required=[],
            handler=knowledge_plan,
            safety_level="low",
        ),
        Tool(
            name="register_knowledge_source",
            description="Register a local dataset path under a known dataset id.",
            parameters={"dataset_id": "str", "path": "str", "notes": "str"},
            required=["dataset_id", "path"],
            handler=register_knowledge_source,
            safety_level="medium",
        ),
        Tool(
            name="list_registered_knowledge",
            description="List dataset paths currently registered for corpus ingestion.",
            parameters={},
            required=[],
            handler=list_registered_knowledge,
            safety_level="low",
        ),
        Tool(
            name="ingest_registered_knowledge",
            description="Ingest all registered dataset sources into the offline knowledge base.",
            parameters={"max_files_per_dataset": "int"},
            required=[],
            handler=ingest_registered_knowledge,
            safety_level="medium",
        ),
    ]
