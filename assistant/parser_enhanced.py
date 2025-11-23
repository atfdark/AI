import time
import re
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Intent(Enum):
    """Enumeration of possible command intents."""
    OPEN_APPLICATION = "open_application"
    CLOSE_WINDOW = "close_window"
    SYSTEM_CONTROL = "system_control"
    FILE_OPERATION = "file_operation"
    TEXT_OPERATION = "text_operation"
    VOLUME_CONTROL = "volume_control"
    SCREENSHOT = "screenshot"
    SEARCH = "search"
    WEB_BROWSING = "web_browsing"
    DICTATION = "dictation"
    SWITCH_MODE = "switch_mode"
    NEWS_REPORTING = "news_reporting"
    TODO_GENERATION = "todo_generation"
    TODO_MANAGEMENT = "todo_management"
    UNKNOWN = "unknown"


@dataclass
class CommandResult:
    """Result of command parsing and execution."""
    intent: Intent
    confidence: float
    action: str
    parameters: Dict[str, any]
    success: bool = True
    message: str = ""


class EnhancedCommandParser:
    """Enhanced command parser with intent recognition and natural language understanding."""
    
    def __init__(self, actions, tts, config_path: str = None):
        self.actions = actions
        self.tts = tts
        self.mode = 'command'  # 'command' or 'dictation'
        
        # Load configuration
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), '..', 'config.json'
        )
        self.config = self._load_config()
        
        # Safety settings
        self.safety = self.config.get('safety', {})
        
        # Command patterns for intent recognition
        self.command_patterns = self._initialize_patterns()
        # Reorder patterns to check more specific ones first
        self._reorder_patterns()
        
        # Learning data for personalization
        self.learning_data = self._load_learning_data()
        
        # Statistics
        self.stats = {
            'commands_processed': 0,
            'successful_commands': 0,
            'failed_commands': 0,
            'intent_accuracy': {}
        }

    def _load_config(self) -> dict:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_learning_data(self) -> dict:
        """Load user learning and personalization data."""
        learning_file = "user_learning.json"
        try:
            with open(learning_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {
                'frequent_commands': {},
                'user_preferences': {},
                'custom_commands': {},
                'corrected_intents': {}
            }

    def _initialize_patterns(self) -> Dict[Intent, List[Tuple[str, float]]]:
        """Initialize command patterns for intent recognition."""
        return {
            Intent.OPEN_APPLICATION: [
                (r'\b(open|launch|start|run)\s+(.+?)\s*(?:please|now)?$', 0.9),
                (r'\b(want|need)\s+(?:to\s+)?(open|launch|start|run)\s+(.+?)$', 0.8),
                (r'\b(please\s+)?(open|launch|start|run)\s+(.+?)$', 0.85),
            ],
            
            Intent.CLOSE_WINDOW: [
                (r'\b(close|shutdown|exit)\s+(?:this\s+)?(?:window|application|app)?$', 0.9),
                (r'\b(shut\s+down|exit\s+out\s+of)\s+(?:this\s+)?(?:window|app)?$', 0.8),
                (r'\b(quit|kill)\s+(?:this\s+)?(?:app|application)?$', 0.85),
            ],
            
            Intent.SYSTEM_CONTROL: [
                (r'\b(minimize|hide|maximize|show)\s+(?:this\s+)?(?:window|application)?$', 0.8),
                (r'\b(switch\s+to|switch\s+focus\s+to|go\s+to)\s+(.+?)$', 0.75),
                (r'\b(next\s+window|previous\s+window|alt\s+tab)$', 0.9),
            ],
            
            Intent.VOLUME_CONTROL: [
                (r'\b(volume|turn|increase|decrease)\s+(?:the\s+)?(?:volume|sound)\s*(?:up|down|higher|lower)?$', 0.9),
                (r'\b(make\s+it|turn\s+it)\s+(?:louder|quieter|higher|lower)$', 0.7),
                (r'\b(mute|unmute|silence)\s*(?:the\s+)?(?:volume|sound|computer)?$', 0.85),
            ],
            
            Intent.SCREENSHOT: [
                (r'\b(take\s+)?(a\s+)?screenshot\s*(?:please|now)?$', 0.9),
                (r'\b(capture\s+screen|screen\s+capture|screen\s+shot)$', 0.85),
                (r'\b(save\s+screen|save\s+this\s+screen)$', 0.7),
            ],
            
            Intent.TEXT_OPERATION: [
                (r'\b(copy|paste|cut|select\s+all)\s*(?:this|that|everything|all)?$', 0.9),
                (r'\b(undo|redo)\s*(?:that|last\s+action)?$', 0.8),
                (r'\b(save|save\s+as|print)\s*(?:this|that|document|file)?$', 0.85),
            ],
            
            Intent.SEARCH: [
                (r'\b(search\s+for|find|look\s+for)\s+(.+?)$', 0.9),
                (r'\b(what\s+is|who\s+is|when\s+is|where\s+is|why\s+is|how\s+to)\s+(.+?)$', 0.8),
                (r'\b(google\s+|bing\s+|yahoo\s+)\s*(.+?)$', 0.85),
            ],
            
            Intent.WEB_BROWSING: [
                (r'\b(go\s+to|open|visit|navigate\s+to)\s+(?:website\s+)?(.+?)$', 0.8),
                (r'\b(browse\s+to|surf\s+to)\s+(.+?)$', 0.75),
                (r'\b(check\s+out|look\s+at)\s+(.+?)$', 0.6),
            ],
            
            Intent.SWITCH_MODE: [
                (r'\b(start|begin|enter)\s+(?:dictation|dictation\s+mode)$', 1.0),
                (r'\b(stop|end|exit)\s+(?:dictation|dictation\s+mode)$', 1.0),
                (r'\b(switch\s+to|change\s+to)\s+(command|dictation)\s+mode$', 0.9),
            ],

            Intent.NEWS_REPORTING: [
                (r'\b(what\'s\s+the\s+news|tell\s+me\s+the\s+news|get\s+news|news\s+update|latest\s+news)\s*(?:please|now)?$', 0.9),
                (r'\b(give\s+me\s+news|fetch\s+news|news\s+report)\s*(?:please|now)?$', 0.85),
                (r'\b(what\'s\s+happening|what\'s\s+going\s+on|current\s+events)\s*(?:in\s+the\s+world)?$', 0.7),
            ],

            Intent.TODO_GENERATION: [
                (r'\b(create|make|generate)\s+(?:a\s+)?(?:todo|to-do|task)\s+list\s+(?:for\s+)?(.+?)$', 0.9),
                (r'\b(start\s+)?(?:a\s+)?(?:todo|to-do|task)\s+list\s+(?:called|named)\s+(.+?)$', 0.8),
            ],

            Intent.TODO_MANAGEMENT: [
                (r'\b(show|list|display|get)\s+(?:my\s+)?(?:todo|to-do|task)\s+lists?$', 0.9),
                (r'\b(add\s+task|add\s+to\s+todo|add\s+to\s+list)\s+(.+?)$', 0.85),
                (r'\b(remove|delete)\s+(?:task\s+)?(.+?)\s+from\s+(?:todo|to-do|list)$', 0.8),
                (r'\b(mark\s+(.+?)\s+as\s+(?:done|completed)|complete\s+(.+?))\s*$', 0.85),
                (r'\b(what\s+are\s+my\s+tasks|what\s+do\s+i\s+have\s+to\s+do)\s*(?:today|now)?$', 0.8),
            ]
        }

    def _reorder_patterns(self):
        """Reorder patterns to check more specific ones first."""
        # Create a new ordered dict with management patterns first
        ordered_patterns = {}

        # Add management patterns first (more specific)
        if Intent.TODO_MANAGEMENT in self.command_patterns:
            ordered_patterns[Intent.TODO_MANAGEMENT] = self.command_patterns[Intent.TODO_MANAGEMENT]

        # Add generation patterns
        if Intent.TODO_GENERATION in self.command_patterns:
            ordered_patterns[Intent.TODO_GENERATION] = self.command_patterns[Intent.TODO_GENERATION]

        # Add all other patterns
        for intent, patterns in self.command_patterns.items():
            if intent not in [Intent.TODO_MANAGEMENT, Intent.TODO_GENERATION]:
                ordered_patterns[intent] = patterns

        self.command_patterns = ordered_patterns

    def parse_intent(self, text: str) -> CommandResult:
        """Parse text to determine intent and extract parameters."""
        text = text.strip().lower()
        
        # Check mode switching first
        if 'start dictation' in text or 'begin dictation' in text:
            return CommandResult(
                intent=Intent.SWITCH_MODE,
                confidence=1.0,
                action="start_dictation",
                parameters={}
            )
        
        if 'stop dictation' in text or 'end dictation' in text:
            return CommandResult(
                intent=Intent.SWITCH_MODE,
                confidence=1.0,
                action="stop_dictation",
                parameters={}
            )
        
        # Dictation mode
        if self.mode == 'dictation':
            return CommandResult(
                intent=Intent.DICTATION,
                confidence=1.0,
                action="type_text",
                parameters={'text': text}
            )
        
        # Intent recognition for command mode
        best_match = None
        best_confidence = 0.0
        
        for intent, patterns in self.command_patterns.items():
            for pattern, confidence in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match and confidence > best_confidence:
                    best_match = (intent, match, confidence)
                    best_confidence = confidence
        
        if best_match:
            intent, match, confidence = best_match
            parameters = self._extract_parameters(intent, match, text)
            
            return CommandResult(
                intent=intent,
                confidence=confidence,
                action=intent.value,
                parameters=parameters
            )
        
        # Fallback to simple keyword matching for backward compatibility
        return self._fallback_parse(text)

    def _extract_parameters(self, intent: Intent, match: re.Match, text: str) -> Dict[str, any]:
        """Extract parameters based on intent and regex match."""
        parameters = {}
        
        if intent == Intent.OPEN_APPLICATION:
            # Extract application name from the match
            if match.lastindex >= 2:
                app_name = match.group(2) if match.lastindex >= 2 else match.group(1)
            else:
                app_name = match.group(1)
            parameters['application'] = app_name.strip()
        
        elif intent == Intent.SEARCH:
            # Extract search query
            if match.lastindex >= 2:
                query = match.group(2)
            else:
                query = match.group(1)
            parameters['query'] = query.strip()
            parameters['engine'] = 'google'  # Default search engine
        
        elif intent == Intent.WEB_BROWSING:
            # Extract website URL or name
            if match.lastindex >= 2:
                url = match.group(2)
            else:
                url = match.group(1)
            parameters['url'] = url.strip()
        
        elif intent == Intent.SYSTEM_CONTROL:
            # Extract target application/window
            if match.lastindex >= 2 and len(match.groups()) >= 2:
                parameters['target'] = match.group(2).strip()

        elif intent == Intent.TODO_GENERATION:
            # Extract list name and tasks
            if match.lastindex >= 2:
                list_name = match.group(2).strip()
                # For simple list creation, don't extract tasks from the list name
                parameters['list_name'] = list_name
                parameters['tasks'] = []  # Empty list, tasks can be added later

        elif intent == Intent.TODO_MANAGEMENT:
            # Extract action and task/list details
            if 'show' in text or 'list' in text or 'display' in text or 'get' in text:
                parameters['action'] = 'list'
            elif 'add' in text:
                parameters['action'] = 'add'
                # Extract task from "add task X" or "add to todo X"
                task_match = re.search(r'add\s+(?:task\s+)?(.+?)(?:\s+to\s+(?:todo|list))?$', text, re.IGNORECASE)
                if task_match:
                    task = task_match.group(1).strip()
                    parameters['task'] = task
            elif 'remove' in text or 'delete' in text:
                parameters['action'] = 'remove'
                # Extract task from "remove X from todo"
                task_match = re.search(r'(?:remove|delete)\s+(?:task\s+)?(.+?)\s+from\s+(?:todo|list)', text, re.IGNORECASE)
                if task_match:
                    task = task_match.group(1).strip()
                    parameters['task'] = task
            elif match and (match.group(2) or match.group(3)):  # This pattern matches mark X as done or complete X
                parameters['action'] = 'complete'
                # The regex captured the task in group 2 (for mark) or group 3 (for complete)
                task = match.group(2) if match.group(2) else match.group(3)
                if task:
                    parameters['task'] = task.strip()

        return parameters

    def _extract_tasks_from_text(self, text: str) -> List[str]:
        """Extract tasks from natural language text."""
        # For todo generation, look for task lists after keywords
        # Remove the command part first
        text = re.sub(r'^(create|make|generate|start)\s+(a\s+)?todo\s+list\s+(for|called|named)?\s*', '', text, flags=re.IGNORECASE)

        tasks = []

        # Split by commas
        if ',' in text:
            parts = text.split(',')
            for part in parts:
                task = part.strip()
                if task and len(task) > 1:  # Avoid single chars
                    # Clean up common prefixes
                    task = re.sub(r'^(and|or|also)\s+', '', task, flags=re.IGNORECASE)
                    task = re.sub(r'^(buy|get|do)\s+', '', task, flags=re.IGNORECASE)
                    tasks.append(task)

        # If no commas, try to find multiple tasks with "and"
        elif ' and ' in text.lower():
            parts = text.split(' and ')
            for part in parts:
                task = part.strip()
                if task and len(task) > 1:
                    task = re.sub(r'^(buy|get|do)\s+', '', task, flags=re.IGNORECASE)
                    tasks.append(task)

        # Single task - but only if it looks like a task, not the whole command
        elif text and len(text) > 3 and not text.startswith(('add', 'remove', 'mark', 'show')):
            task = text.strip()
            task = re.sub(r'^(buy|get|do)\s+', '', task, flags=re.IGNORECASE)
            if task:
                tasks.append(task)

        return [t for t in tasks if t]  # Filter empty

    def _extract_task_from_text(self, text: str) -> str:
        """Extract a single task from text."""
        # Remove action words and extract the task
        task = re.sub(r'^(add|remove|delete|complete|mark as done|mark as completed)\s+', '', text, flags=re.IGNORECASE)
        task = re.sub(r'\s+from\s+(todo|to-do|list).*$', '', task, flags=re.IGNORECASE)
        task = re.sub(r'\s+to\s+(todo|to-do|list).*$', '', task, flags=re.IGNORECASE)
        return task.strip()

    def _fallback_parse(self, text: str) -> CommandResult:
        """Fallback keyword-based parsing for compatibility."""
        # Simple keyword matching (preserving original behavior)
        lower = text.lower()
        
        if lower.startswith('open ') or lower.startswith('launch '):
            for app in self.actions.get_known_apps():
                if app.lower() in lower:
                    return CommandResult(
                        intent=Intent.OPEN_APPLICATION,
                        confidence=0.7,
                        action="open_application",
                        parameters={'application': app}
                    )
        
        # Add more fallback patterns as needed...
        
        return CommandResult(
            intent=Intent.UNKNOWN,
            confidence=0.0,
            action="unknown",
            parameters={'text': text}
        )

    def handle_text(self, text: str):
        """Main text processing handler."""
        self.stats['commands_processed'] += 1

        # Parse the intent
        result = self.parse_intent(text)

        # Execute the command
        success = self.execute_command(result)

        # Update statistics
        if success:
            self.stats['successful_commands'] += 1
        else:
            self.stats['failed_commands'] += 1

        # Update intent accuracy tracking
        intent_str = result.intent.value
        if intent_str not in self.stats['intent_accuracy']:
            self.stats['intent_accuracy'][intent_str] = {'correct': 0, 'total': 0}

        self.stats['intent_accuracy'][intent_str]['total'] += 1
        if success:
            self.stats['intent_accuracy'][intent_str]['correct'] += 1

    def execute_command(self, result: CommandResult) -> bool:
        """Execute the parsed command."""
        try:
            if result.intent == Intent.SWITCH_MODE:
                return self._handle_mode_switch(result)
            
            elif result.intent == Intent.DICTATION:
                return self._handle_dictation(result)
            
            elif result.intent == Intent.OPEN_APPLICATION:
                return self._handle_open_application(result)
            
            elif result.intent == Intent.CLOSE_WINDOW:
                return self._handle_close_window(result)
            
            elif result.intent == Intent.VOLUME_CONTROL:
                return self._handle_volume_control(result)
            
            elif result.intent == Intent.SCREENSHOT:
                return self._handle_screenshot(result)
            
            elif result.intent == Intent.TEXT_OPERATION:
                return self._handle_text_operation(result)
            
            elif result.intent == Intent.SEARCH:
                return self._handle_search(result)
            
            elif result.intent == Intent.WEB_BROWSING:
                return self._handle_web_browsing(result)

            elif result.intent == Intent.NEWS_REPORTING:
                return self._handle_news_reporting(result)

            elif result.intent == Intent.TODO_GENERATION:
                return self._handle_todo_generation(result)

            elif result.intent == Intent.TODO_MANAGEMENT:
                return self._handle_todo_management(result)

            else:
                responses = [
                    "Sorry, I didn't understand that command",
                    "I didn't catch that",
                    "Could you repeat that?",
                    "I'm not sure what you mean"
                ]
                import random
                self.tts.say(random.choice(responses))
                return False
                
        except Exception as e:
            print(f"[ERROR] Command execution failed: {e}")
            self.tts.say("Sorry, there was an error executing that command")
            return False

    def _handle_mode_switch(self, result: CommandResult) -> bool:
        """Handle mode switching commands."""
        if result.action == "start_dictation":
            self.mode = 'dictation'
            self.tts.say("Dictation started")
            return True
        elif result.action == "stop_dictation":
            self.mode = 'command'
            self.tts.say("Dictation stopped")
            return True
        return False

    def _handle_dictation(self, result: CommandResult) -> bool:
        """Handle dictation mode."""
        text = result.parameters.get('text', '')
        self.actions.type_text(text)
        return True

    def _handle_open_application(self, result: CommandResult) -> bool:
        """Handle application opening commands."""
        app_name = result.parameters.get('application', '')

        # Find best match for application name
        known_apps = self.actions.get_known_apps()
        best_match = None

        for app in known_apps:
            if app_name.lower() in app.lower() or app.lower() in app_name.lower():
                best_match = app
                break

        if best_match:
            success = self.actions.launch_app(best_match)
            if success:
                responses = [
                    f"Opening {best_match}",
                    f"Launching {best_match} for you",
                    f"Sure, opening {best_match}",
                    f"{best_match} is now open"
                ]
                import random
                self.tts.say(random.choice(responses))
                return True
            else:
                self.tts.say(f"Sorry, I couldn't open {best_match}")
                return False
        else:
            self.tts.say(f"I don't know how to open {app_name}")
            return False

    def _handle_close_window(self, result: CommandResult) -> bool:
        """Handle window closing commands."""
        self.actions.close_window()
        responses = ["Window closed", "Done", "Closed"]
        import random
        self.tts.say(random.choice(responses))
        return True

    def _handle_volume_control(self, result: CommandResult) -> bool:
        """Handle volume control commands."""
        text = result.parameters.get('text', '').lower()

        if 'up' in text or 'higher' in text or 'increase' in text:
            self.actions.volume_up(steps=2)
            responses = ["Volume increased", "Louder", "Turning it up"]
            import random
            self.tts.say(random.choice(responses))
            return True
        elif 'down' in text or 'lower' in text or 'decrease' in text:
            self.actions.volume_down(steps=2)
            responses = ["Volume decreased", "Quieter", "Turning it down"]
            import random
            self.tts.say(random.choice(responses))
            return True
        elif 'mute' in text:
            self.actions.volume_down(steps=10)  # Effectively mute
            responses = ["Volume muted", "Muted", "Shh"]
            import random
            self.tts.say(random.choice(responses))
            return True

        return False

    def _handle_screenshot(self, result: CommandResult) -> bool:
        """Handle screenshot commands."""
        filename = self.actions.take_screenshot()
        if filename:
            responses = ["Screenshot taken", "Captured", "Got it", "Screen captured"]
            import random
            self.tts.say(random.choice(responses))
            return True
        else:
            self.tts.say("Sorry, couldn't take screenshot")
            return False

    def _handle_text_operation(self, result: CommandResult) -> bool:
        """Handle text operation commands."""
        text = result.parameters.get('text', '').lower()

        if 'copy' in text and 'paste' not in text:
            self.actions.copy()
            responses = ["Copied", "Copied to clipboard", "Done"]
            import random
            self.tts.say(random.choice(responses))
            return True
        elif 'paste' in text:
            self.actions.paste()
            responses = ["Pasted", "Pasted from clipboard", "Done"]
            import random
            self.tts.say(random.choice(responses))
            return True
        elif 'save' in text:
            self.actions.save()
            responses = ["Saved", "File saved", "Done"]
            import random
            self.tts.say(random.choice(responses))
            return True
        elif 'select all' in text:
            self.actions.select_all()
            responses = ["Selected all", "Everything selected", "Done"]
            import random
            self.tts.say(random.choice(responses))
            return True

        return False

    def _handle_search(self, result: CommandResult) -> bool:
        """Handle search commands."""
        query = result.parameters.get('query', '')
        if query:
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            success = self.actions.open_url(url)
            if success:
                responses = [f"Searching for {query}", f"Let me search for {query}", f"Looking up {query}"]
                import random
                self.tts.say(random.choice(responses))
                return True
            else:
                self.tts.say(f"Sorry, couldn't search for {query}")
                return False
        return False

    def _handle_web_browsing(self, result: CommandResult) -> bool:
        """Handle web browsing commands."""
        url = result.parameters.get('url', '')
        if url:
            success = self.actions.open_url(url)
            if success:
                responses = [f"Opening {url}", f"Going to {url}", f"Navigating to {url}"]
                import random
                self.tts.say(random.choice(responses))
                return True
            else:
                self.tts.say(f"Sorry, couldn't open {url}")
                return False
        return False

    def _handle_news_reporting(self, result: CommandResult) -> bool:
        """Handle news reporting commands."""
        try:
            news_data = self.actions.fetch_news()
            if news_data:
                # Summarize and speak the news
                summary = f"Here are the latest news headlines: {news_data}"
                self.tts.say(summary)
                return True
            else:
                self.tts.say("Sorry, I couldn't fetch the news right now")
                return False
        except Exception as e:
            print(f"[ERROR] News reporting failed: {e}")
            self.tts.say("Sorry, there was an error getting the news")
            return False

    def _handle_todo_generation(self, result: CommandResult) -> bool:
        """Handle todo list generation commands."""
        list_name = result.parameters.get('list_name', 'default')
        tasks = result.parameters.get('tasks', [])

        success = self.actions.create_todo_list(list_name, tasks)
        if success:
            if tasks:
                self.tts.say(f"Created todo list '{list_name}' with {len(tasks)} tasks")
            else:
                self.tts.say(f"Created empty todo list '{list_name}'")
            return True
        else:
            self.tts.say("Sorry, couldn't create the todo list")
            return False

    def _handle_todo_management(self, result: CommandResult) -> bool:
        """Handle todo list management commands."""
        action = result.parameters.get('action', '')

        if action == 'list':
            lists = self.actions.get_todo_lists()
            if lists:
                list_names = list(lists.keys())
                self.tts.say(f"You have {len(list_names)} todo lists: {', '.join(list_names)}")
                return True
            else:
                self.tts.say("You don't have any todo lists yet")
                return True

        elif action == 'add':
            task = result.parameters.get('task', '')
            if task:
                success = self.actions.add_todo_task('default', task)  # Default list
                if success:
                    self.tts.say(f"Added task: {task}")
                    return True
                else:
                    self.tts.say("Sorry, couldn't add the task")
                    return False
            else:
                self.tts.say("No task specified")
                return False

        elif action == 'remove':
            task = result.parameters.get('task', '')
            if task:
                success = self.actions.remove_todo_task('default', task)
                if success:
                    self.tts.say(f"Removed task: {task}")
                    return True
                else:
                    self.tts.say("Sorry, couldn't remove the task")
                    return False
            else:
                self.tts.say("No task specified")
                return False

        elif action == 'complete':
            task = result.parameters.get('task', '')
            if task:
                success = self.actions.complete_todo_task('default', task)
                if success:
                    self.tts.say(f"Marked as completed: {task}")
                    return True
                else:
                    self.tts.say("Sorry, couldn't mark the task as completed")
                    return False
            else:
                self.tts.say("No task specified")
                return False

        return False

    def get_stats(self) -> dict:
        """Get parser statistics."""
        return self.stats.copy()

    def get_learning_data(self) -> dict:
        """Get user learning data."""
        return self.learning_data.copy()


# Backward compatibility
CommandParser = EnhancedCommandParser