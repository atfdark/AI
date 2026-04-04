"""Action wrappers exposed as tools for the Jarvis agent runtime."""

from __future__ import annotations

from typing import Any, Dict, List

from .tool_registry import Tool


def create_action_tools(actions) -> List[Tool]:
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
    ]
