import os
import subprocess
import json
import time
import pyautogui
import webbrowser
from urllib.parse import urlparse
try:
    from newsapi import NewsApiClient
except ImportError:
    NewsApiClient = None


class Actions:
    def __init__(self, config_path=None):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.config_path = config_path or os.path.join(root, 'config.json')
        self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception:
            self.config = {"apps": {}}

    def get_known_apps(self):
        return list(self.config.get('apps', {}).keys())

    def launch_app(self, name):
        apps = self.config.get('apps', {})
        if name not in apps:
            raise FileNotFoundError(f"Unknown app: {name}")
        path = apps[name]
        try:
            # Use shell=True for .lnk or simple paths on Windows
            subprocess.Popen(path, shell=True)
            return True
        except Exception as e:
            print(f"Failed to launch {name}: {e}")
            return False

    def type_text(self, text):
        # Slight pause to ensure focus
        time.sleep(0.05)
        try:
            pyautogui.write(text, interval=0.01)
        except Exception as e:
            print(f"Typing error: {e}")

    def hotkey(self, *keys):
        try:
            pyautogui.hotkey(*keys)
        except Exception as e:
            print(f"Hotkey error: {e}")

    def press(self, key):
        try:
            pyautogui.press(key)
        except Exception as e:
            print(f"Press key error: {e}")

    def take_screenshot(self, filename=None):
        filename = filename or os.path.join(os.getcwd(), 'screenshot.png')
        try:
            im = pyautogui.screenshot()
            im.save(filename)
            return filename
        except Exception as e:
            print(f"Screenshot error: {e}")
            return None

    def open_url(self, url):
        """Open a URL in the default web browser with validation and error handling."""
        try:
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                print(f"Invalid URL: {url}")
                return False

            # Open in browser
            webbrowser.open(url)
            return True
        except Exception as e:
            print(f"Failed to open URL {url}: {e}")
            return False

    def close_window(self):
        # Alt+F4
        self.hotkey('alt', 'f4')

    def copy(self):
        self.hotkey('ctrl', 'c')

    def paste(self):
        self.hotkey('ctrl', 'v')

    def save(self):
        self.hotkey('ctrl', 's')

    def select_all(self):
        self.hotkey('ctrl', 'a')

    def volume_up(self, steps=1):
        for _ in range(steps):
            try:
                pyautogui.press('volumeup')
            except Exception:
                pass

    def volume_down(self, steps=1):
        for _ in range(steps):
            try:
                pyautogui.press('volumedown')
            except Exception:
                pass

    def fetch_news(self):
        """Fetch latest news using NewsAPI and return a summary."""
        if NewsApiClient is None:
            print("[ERROR] NewsAPI library not installed")
            return None

        try:
            # Get API key from config
            news_config = self.config.get('newsapi', {})
            api_key = news_config.get('api_key')

            if not api_key:
                print("[ERROR] NewsAPI key not configured")
                return None

            # Initialize NewsAPI client
            newsapi = NewsApiClient(api_key=api_key)

            # Get top headlines
            top_headlines = newsapi.get_top_headlines(
                language='en',
                country='us',  # You can make this configurable
                page_size=5
            )

            if top_headlines['status'] != 'ok':
                print(f"[ERROR] NewsAPI returned status: {top_headlines['status']}")
                return None

            articles = top_headlines.get('articles', [])
            if not articles:
                return "No news articles found at the moment."

            # Create a summary of the top headlines
            summary_parts = []
            for i, article in enumerate(articles[:3], 1):  # Top 3 headlines
                title = article.get('title', 'No title')
                source = article.get('source', {}).get('name', 'Unknown source')
                summary_parts.append(f"{i}. {title} from {source}")

            summary = " ".join(summary_parts)
            return summary

        except Exception as e:
            print(f"[ERROR] Failed to fetch news: {e}")
            return None

    def create_todo_list(self, list_name: str, tasks: list = None) -> bool:
        """Create a new todo list with optional initial tasks."""
        try:
            todo_data = self._load_todo_data()
            if list_name in todo_data:
                return False  # List already exists

            todo_data[list_name] = {
                'tasks': [{'description': task, 'completed': False} for task in (tasks or [])],
                'created_at': time.time()
            }
            self._save_todo_data(todo_data)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to create todo list: {e}")
            return False

    def add_todo_task(self, list_name: str, task: str) -> bool:
        """Add a task to an existing todo list."""
        try:
            todo_data = self._load_todo_data()
            if list_name not in todo_data:
                # Create list if it doesn't exist
                todo_data[list_name] = {'tasks': [], 'created_at': time.time()}

            todo_data[list_name]['tasks'].append({
                'description': task,
                'completed': False
            })
            self._save_todo_data(todo_data)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to add todo task: {e}")
            return False

    def remove_todo_task(self, list_name: str, task: str) -> bool:
        """Remove a task from a todo list."""
        try:
            todo_data = self._load_todo_data()
            if list_name not in todo_data:
                return False

            tasks = todo_data[list_name]['tasks']
            # Find and remove the task
            for i, t in enumerate(tasks):
                if t['description'].lower() == task.lower():
                    tasks.pop(i)
                    self._save_todo_data(todo_data)
                    return True
            return False
        except Exception as e:
            print(f"[ERROR] Failed to remove todo task: {e}")
            return False

    def complete_todo_task(self, list_name: str, task: str) -> bool:
        """Mark a task as completed."""
        try:
            todo_data = self._load_todo_data()
            if list_name not in todo_data:
                return False

            tasks = todo_data[list_name]['tasks']
            for t in tasks:
                if t['description'].lower() == task.lower():
                    t['completed'] = True
                    self._save_todo_data(todo_data)
                    return True
            return False
        except Exception as e:
            print(f"[ERROR] Failed to complete todo task: {e}")
            return False

    def get_todo_lists(self) -> dict:
        """Get all todo lists."""
        try:
            return self._load_todo_data()
        except Exception as e:
            print(f"[ERROR] Failed to get todo lists: {e}")
            return {}

    def get_todo_list(self, list_name: str) -> list:
        """Get tasks from a specific todo list."""
        try:
            todo_data = self._load_todo_data()
            if list_name in todo_data:
                return todo_data[list_name]['tasks']
            return []
        except Exception as e:
            print(f"[ERROR] Failed to get todo list: {e}")
            return []

    def _load_todo_data(self) -> dict:
        """Load todo data from JSON file."""
        todo_file = os.path.join(os.path.dirname(self.config_path), 'todo_lists.json')
        try:
            with open(todo_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            print(f"[ERROR] Failed to load todo data: {e}")
            return {}

    def _save_todo_data(self, data: dict) -> None:
        """Save todo data to JSON file."""
        todo_file = os.path.join(os.path.dirname(self.config_path), 'todo_lists.json')
        try:
            with open(todo_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] Failed to save todo data: {e}")
