"""Jarvis agent runtime: tool registry, router, executor, and speech feedback."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict

from .action_tools import create_action_tools
from .audit_logger import AuditLogger
from .knowledge_bootstrap import KnowledgeBootstrapManager
from .llm_router import LLMRouter
from .knowledge_brain import KnowledgeBrain
from .plan_executor import PlanExecutor
from .safety_policy import SafetyPolicy
from .semantic_file_search import SemanticFileSearchEngine
from .semantic_memory import SemanticMemoryStore
from .tool_registry import ToolRegistry


class AgentRuntime:
    """Coordinates routing and execution for low-confidence or multi-step requests."""

    def __init__(self, actions, tts=None, config_path: str | None = None):
        self.actions = actions
        self.tts = tts
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "..", "config.json")
        self.config = self._load_config()
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        agent_cfg = self.config.get("agent", {})
        self.enabled = bool(agent_cfg.get("enabled", True))
        self.speak_responses = bool(agent_cfg.get("speak_responses", True))
        self.low_confidence_threshold = float(agent_cfg.get("low_confidence_threshold", 0.55))
        self._knowledge_scheduler_thread = None
        self._knowledge_scheduler_stop = threading.Event()
        self._last_knowledge_ingest_at = 0.0
        self._last_knowledge_ingest_result: Dict[str, Any] = {}
        self._last_registered_ingest_result: Dict[str, Any] = {}

        self.intent_tool_map = {
            "open_application": "open_app",
            "web_browsing": "open_url",
            "search": "search_web",
            "screenshot": "take_screenshot",
            "weather": "get_weather",
            "wikipedia": "wikipedia_summary",
            "news_reporting": "fetch_news",
            "todo_generation": "create_todo",
            "todo_management": "list_todos",
            "volume_control": "volume_up",
        }

        memory_cfg = self.config.get("memory", {})
        memory_db_path = memory_cfg.get("db_path")
        if memory_db_path and not os.path.isabs(memory_db_path):
            memory_db_path = os.path.join(self.root_dir, memory_db_path)
        self.memory_store = SemanticMemoryStore(
            db_path=memory_db_path,
            embedding_model=memory_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
            use_embeddings=bool(memory_cfg.get("enable_embeddings", True)),
        )

        file_search_cfg = self.config.get("semantic_file_search", {})
        file_db_path = file_search_cfg.get("db_path")
        if file_db_path and not os.path.isabs(file_db_path):
            file_db_path = os.path.join(self.root_dir, file_db_path)
        self.file_search_engine = SemanticFileSearchEngine(
            workspace_root=self.root_dir,
            db_path=file_db_path,
            embedding_model=file_search_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
            use_embeddings=bool(file_search_cfg.get("enable_embeddings", True)),
            max_chars_per_file=int(file_search_cfg.get("max_chars_per_file", 3000)),
        )

        knowledge_cfg = self.config.get("knowledge_brain", {})
        knowledge_db_path = knowledge_cfg.get("db_path")
        if knowledge_db_path and not os.path.isabs(knowledge_db_path):
            knowledge_db_path = os.path.join(self.root_dir, knowledge_db_path)
        self.knowledge_brain = KnowledgeBrain(
            db_path=knowledge_db_path or os.path.join(self.root_dir, "data_versions", "knowledge_brain.db"),
            embedding_model=knowledge_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
            use_embeddings=bool(knowledge_cfg.get("enable_embeddings", True)),
            chunk_size_chars=int(knowledge_cfg.get("chunk_size_chars", 1200)),
            chunk_overlap_chars=int(knowledge_cfg.get("chunk_overlap_chars", 180)),
            ocr_enabled=bool(knowledge_cfg.get("ocr_enabled", True)),
            trust_overrides=knowledge_cfg.get("trust_overrides", {}),
        )

        corpus_cfg = self.config.get("knowledge_corpus", {})
        registry_path = corpus_cfg.get("registry_path")
        if registry_path and not os.path.isabs(registry_path):
            registry_path = os.path.join(self.root_dir, registry_path)
        self.knowledge_corpus = KnowledgeBootstrapManager(
            workspace_root=self.root_dir,
            registry_path=registry_path,
        )

        if bool(file_search_cfg.get("auto_index_on_start", False)):
            try:
                self.file_search_engine.refresh_index(
                    max_files=int(file_search_cfg.get("max_files_per_index", 3000)),
                    force=False,
                )
            except Exception:
                pass

        if bool(knowledge_cfg.get("auto_ingest_on_start", True)):
            source_dir = knowledge_cfg.get("default_source_dir", "knowledge_sources")
            if not os.path.isabs(source_dir):
                source_dir = os.path.join(self.root_dir, source_dir)
            try:
                self._last_knowledge_ingest_result = self.knowledge_brain.ingest_directory(
                    root_path=source_dir,
                    max_files=int(knowledge_cfg.get("max_files_per_ingest", 500)),
                )
                self._last_knowledge_ingest_at = time.time()
            except Exception:
                pass

        if bool(corpus_cfg.get("auto_ingest_registered_on_start", False)):
            try:
                self._last_registered_ingest_result = self.knowledge_corpus.ingest_registered(
                    knowledge_brain=self.knowledge_brain,
                    max_files_per_dataset=int(corpus_cfg.get("max_files_per_dataset", 2000)),
                )
                self._last_knowledge_ingest_at = time.time()
            except Exception:
                pass

        schedule_cfg = knowledge_cfg.get("schedule", {})
        if bool(schedule_cfg.get("enabled", False)):
            interval_seconds = max(300, int(schedule_cfg.get("interval_minutes", 30)) * 60)
            source_dir = knowledge_cfg.get("default_source_dir", "knowledge_sources")
            if not os.path.isabs(source_dir):
                source_dir = os.path.join(self.root_dir, source_dir)
            max_files = int(knowledge_cfg.get("max_files_per_ingest", 500))
            self._knowledge_scheduler_thread = threading.Thread(
                target=self._knowledge_scheduler_loop,
                args=(source_dir, max_files, interval_seconds),
                daemon=True,
            )
            self._knowledge_scheduler_thread.start()

        self.registry = ToolRegistry()
        self.registry.register_many(
            create_action_tools(
                actions,
                memory_store=self.memory_store,
                file_search_engine=self.file_search_engine,
                qa_callback=self._answer_offline,
                knowledge_brain=self.knowledge_brain,
                corpus_manager=self.knowledge_corpus,
            )
        )

        self.safety_policy = SafetyPolicy(config_path=self.config_path)
        self.audit_logger = AuditLogger()
        self.router = LLMRouter(config=self.config)
        self.executor = PlanExecutor(self.registry, self.safety_policy, self.audit_logger)

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}

    def is_candidate(self, text: str, parsed_intent: str, parsed_confidence: float, mode: str) -> bool:
        """Return whether this command should be handled by the agent layer."""
        if not self.enabled or mode != "command":
            return False

        lowered = text.lower().strip()
        if lowered.startswith("agent "):
            return True
        if " and " in lowered:
            return True
        if parsed_intent == "unknown":
            return True
        if parsed_confidence < self.low_confidence_threshold:
            return True

        return False

    def process(self, text: str, parsed_result, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Process a command through router + executor."""
        if not self.enabled:
            return {"handled": False}

        parsed_intent = parsed_result.intent.value
        parsed_confidence = float(parsed_result.confidence)
        parsed_parameters = parsed_result.parameters or {}

        if not self.is_candidate(text, parsed_intent, parsed_confidence, mode="command"):
            return {"handled": False}

        expanded_text = self._apply_alias(text)
        enriched_context = self._build_context(expanded_text, context or {})

        decision = self.router.decide(
            user_text=expanded_text,
            parsed_intent=parsed_intent,
            parsed_confidence=parsed_confidence,
            parsed_parameters=parsed_parameters,
            tools=self.registry.list_tool_summaries(),
            context=enriched_context,
            intent_tool_map=self.intent_tool_map,
        )

        executed_tool = ""
        if decision.route == "direct" and decision.tool:
            result = self.executor.execute_direct(decision.tool, decision.parameters)
            executed_tool = decision.tool
        elif decision.route == "plan":
            result = self.executor.execute_plan(decision.steps)
            if decision.steps:
                executed_tool = decision.steps[0].get("tool", "")
        elif decision.route == "clarify":
            result = {"success": False, "message": decision.reply or "Please clarify your request."}
        else:
            return {"handled": False}

        message = result.get("message", "Done")
        if self.tts and self.speak_responses and message:
            self.tts.say(message)

        success = bool(result.get("success", False))
        self.memory_store.track_interaction(
            user_text=expanded_text,
            decision_route=decision.route,
            tool_name=executed_tool or "none",
            success=success,
            response=message,
        )

        self.memory_store.learn_alias_from_command(
            user_text=expanded_text,
            tool_name=executed_tool,
            params=decision.parameters if decision.route == "direct" else {},
            success=success,
        )

        return {
            "handled": True,
            "success": success,
            "response": message,
            "decision": {
                "route": decision.route,
                "reason": decision.reason,
            },
            "result": result,
        }

    def health_check(self) -> Dict[str, Any]:
        """Return runtime diagnostics without invoking voice loop."""
        return {
            "enabled": self.enabled,
            "tool_count": len(self.registry.list_tools()),
            "ollama_enabled": self.router.enable_ollama,
            "ollama_model": self.router.ollama_model,
            "cloud_llm_enabled": self.router.enable_cloud_llm,
            "cloud_llm_provider": self.router.cloud_provider,
            "memory_db": self.memory_store.db_path,
            "file_index": self.file_search_engine.stats(),
            "knowledge": self.knowledge_brain.stats(),
            "knowledge_registry": self.knowledge_corpus.stats(),
            "knowledge_scheduler_enabled": self._knowledge_scheduler_thread is not None,
            "last_knowledge_ingest_at": self._last_knowledge_ingest_at,
            "last_knowledge_ingest_result": self._last_knowledge_ingest_result,
            "last_registered_ingest_result": self._last_registered_ingest_result,
        }

    def _knowledge_scheduler_loop(self, source_dir: str, max_files: int, interval_seconds: int) -> None:
        while not self._knowledge_scheduler_stop.wait(interval_seconds):
            try:
                self._last_knowledge_ingest_result = self.knowledge_brain.ingest_directory(
                    root_path=source_dir,
                    max_files=max_files,
                )
                self._last_knowledge_ingest_at = time.time()
            except Exception:
                continue

    def _build_context(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(context)
        enriched["recent_memories"] = self.memory_store.recall_recent(limit=5)
        enriched["relevant_memories"] = self.memory_store.search(query=text, top_k=4)
        enriched["knowledge_preview"] = self.knowledge_brain.search(query=text, top_k=3)
        return enriched

    def _apply_alias(self, text: str) -> str:
        lowered = text.lower().strip()
        match = None
        for verb in ("open", "launch", "start", "run"):
            candidate = re_match_command(lowered, verb)
            if candidate:
                match = (verb, candidate)
                break

        if not match:
            return text

        verb, phrase = match
        resolved = self.memory_store.resolve_alias(phrase)
        if not resolved:
            return text
        return f"{verb} {resolved}"

    def _answer_offline(self, question: str) -> str:
        relevant = self.memory_store.search(query=question, top_k=4)
        context_lines = [item["text"] for item in relevant]
        memory_context = "\n".join(context_lines)

        cited = self.knowledge_brain.build_cited_context(
            query=question,
            top_k=6,
            max_chars=3600,
        )
        knowledge_context = cited.get("context", "")
        citations = cited.get("citations", [])

        has_context = bool(memory_context.strip() or knowledge_context.strip())

        answer = self.router.answer_question(
            question=question,
            memory_context=memory_context,
            knowledge_context=knowledge_context,
        )
        if answer:
            if citations:
                citation_lines = []
                for item in citations[:4]:
                    page_text = f", page {item['page']}" if item.get("page") else ""
                    citation_lines.append(f"[{item['id']}] {item['source_title']}{page_text}")
                answer = answer.rstrip() + "\n\nSources:\n" + "\n".join(citation_lines)
            return answer

        if has_context:
            summary_parts = []
            if knowledge_context:
                summary_parts.append("I found relevant knowledge context in my local knowledge base.")
            if context_lines:
                summary_parts.append("I also found related memories from past interactions.")
            return " ".join(summary_parts)

        if context_lines:
            return "From memory: " + "; ".join(context_lines[:3])

        return "I do not have enough local context to answer that right now."


def re_match_command(text: str, verb: str) -> str | None:
    prefix = f"{verb} "
    if text.startswith(prefix):
        phrase = text[len(prefix) :].strip()
        return phrase or None
    return None
