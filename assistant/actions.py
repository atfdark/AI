import os
import subprocess
import json
import time
import pyautogui
import webbrowser
from urllib.parse import urlparse, quote
try:
    import requests
except ImportError:
    requests = None

try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False

try:
    from newsapi import NewsApiClient
except ImportError:
    NewsApiClient = None

try:
    import wikipedia
except ImportError:
    wikipedia = None

try:
    import pyjokes
except ImportError:
    pyjokes = None

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    import geocoder
except ImportError:
    geocoder = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import win32api
    import win32service
    import win32serviceutil
    import win32eventlog
    import win32evtlogutil
    import win32con
    import win32security
    import win32file
    import win32net
    import win32process
    import win32gui
    import win32com.client
    import winreg
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

try:
    import comtypes
    import comtypes.client
    COMTYPES_AVAILABLE = True
except ImportError:
    COMTYPES_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    PYTTSX3_AVAILABLE = False


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
        print(f"[TYPE_TEXT] Received text: '{text}' (len: {len(text)}, repr: {repr(text)})")
        print(f"[TYPE_TEXT] Hex: {[hex(ord(c)) for c in text]}")

        # Try clipboard paste first (more reliable for Unicode)
        if CLIPBOARD_AVAILABLE and text:
            try:
                # Save current clipboard content
                current_clipboard = pyperclip.paste() if pyperclip.paste() else ""
                # Copy text to clipboard
                pyperclip.copy(text)
                # Paste
                self.hotkey('ctrl', 'v')
                print(f"[TYPE_TEXT] Successfully pasted text using clipboard")
                # Restore clipboard if it had content
                if current_clipboard:
                    time.sleep(0.1)  # Small delay
                    pyperclip.copy(current_clipboard)
                return
            except Exception as e:
                print(f"[TYPE_TEXT] Clipboard paste failed: {e}, falling back to typing")

        # Fallback to typing
        try:
            pyautogui.write(text, interval=0.01)
            print(f"[TYPE_TEXT] Successfully wrote text using pyautogui.write")
        except Exception as e:
            print(f"[TYPE_TEXT] Typing error: {e}")

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

    def init_pyttsx3_engine(self):
        """Initialize pyttsx3 TTS engine."""
        if not PYTTSX3_AVAILABLE:
            print("[ERROR] pyttsx3 library not installed")
            return None

        try:
            engine = pyttsx3.init()
            # Configure engine properties
            engine.setProperty('rate', 180)  # Default speech rate
            engine.setProperty('volume', 0.8)  # Default volume (0.0 to 1.0)
            return engine
        except Exception as e:
            print(f"[ERROR] Failed to initialize pyttsx3 engine: {e}")
            return None

    def get_available_voices(self):
        """Get list of available voices."""
        engine = self.init_pyttsx3_engine()
        if not engine:
            return []

        try:
            voices = engine.getProperty('voices')
            voice_list = []
            for voice in voices:
                voice_info = {
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages,
                    'gender': getattr(voice, 'gender', 'Unknown'),
                    'age': getattr(voice, 'age', 'Unknown')
                }
                voice_list.append(voice_info)
            return voice_list
        except Exception as e:
            print(f"[ERROR] Failed to get available voices: {e}")
            return []

    def set_voice(self, voice_id=None, gender=None):
        """Set TTS voice by ID or gender preference."""
        engine = self.init_pyttsx3_engine()
        if not engine:
            return False

        try:
            voices = engine.getProperty('voices')
            if not voices:
                print("[ERROR] No voices available")
                return False

            selected_voice = None

            if voice_id:
                # Find voice by ID
                for voice in voices:
                    if voice.id == voice_id:
                        selected_voice = voice
                        break
            elif gender:
                # Find voice by gender preference
                gender_lower = gender.lower()
                for voice in voices:
                    voice_gender = getattr(voice, 'gender', '').lower()
                    if gender_lower in voice_gender or gender_lower == voice_gender:
                        selected_voice = voice
                        break

            if selected_voice:
                engine.setProperty('voice', selected_voice.id)
                print(f"[TTS] Voice set to: {selected_voice.name}")
                return True
            else:
                print(f"[ERROR] Voice not found: {voice_id or gender}")
                return False

        except Exception as e:
            print(f"[ERROR] Failed to set voice: {e}")
            return False

    def set_speech_rate(self, rate):
        """Set speech rate (words per minute)."""
        engine = self.init_pyttsx3_engine()
        if not engine:
            return False

        try:
            # Clamp rate between reasonable bounds
            rate = max(50, min(400, rate))
            engine.setProperty('rate', rate)
            print(f"[TTS] Speech rate set to: {rate} WPM")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to set speech rate: {e}")
            return False

    def set_volume(self, volume):
        """Set TTS volume (0.0 to 1.0)."""
        engine = self.init_pyttsx3_engine()
        if not engine:
            return False

        try:
            # Clamp volume between 0.0 and 1.0
            volume = max(0.0, min(1.0, volume))
            engine.setProperty('volume', volume)
            print(f"[TTS] Volume set to: {volume}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to set volume: {e}")
            return False

    def speak_text_pyttsx3(self, text, voice_id=None, rate=None, volume=None):
        """Speak text using pyttsx3 with optional voice/rate/volume settings."""
        engine = self.init_pyttsx3_engine()
        if not engine:
            return False

        try:
            # Apply custom settings if provided
            if voice_id:
                voices = engine.getProperty('voices')
                for voice in voices:
                    if voice.id == voice_id:
                        engine.setProperty('voice', voice.id)
                        break

            if rate is not None:
                rate = max(50, min(400, rate))
                engine.setProperty('rate', rate)

            if volume is not None:
                volume = max(0.0, min(1.0, volume))
                engine.setProperty('volume', volume)

            # Speak the text
            engine.say(text)
            engine.runAndWait()
            return True

        except Exception as e:
            print(f"[ERROR] Failed to speak text with pyttsx3: {e}")
            return False

    def save_text_to_audio_file(self, text, filename, voice_id=None, rate=None, volume=None):
        """Save text to audio file using pyttsx3."""
        if not PYTTSX3_AVAILABLE:
            print("[ERROR] pyttsx3 library not installed")
            return False

        try:
            # Create a temporary engine for file saving
            engine = pyttsx3.init()

            # Apply custom settings if provided
            if voice_id:
                voices = engine.getProperty('voices')
                for voice in voices:
                    if voice.id == voice_id:
                        engine.setProperty('voice', voice.id)
                        break

            if rate is not None:
                rate = max(50, min(400, rate))
                engine.setProperty('rate', rate)

            if volume is not None:
                volume = max(0.0, min(1.0, volume))
                engine.setProperty('volume', volume)

            # Save to file
            engine.save_to_file(text, filename)
            engine.runAndWait()
            print(f"[TTS] Audio saved to: {filename}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to save text to audio file: {e}")
            return False

    def test_voices(self, test_text="Hello, this is a test of the text to speech system."):
        """Test all available voices by speaking the test text."""
        voices = self.get_available_voices()
        if not voices:
            print("[ERROR] No voices available for testing")
            return False

        print(f"[TTS] Testing {len(voices)} available voices...")

        for i, voice_info in enumerate(voices[:5], 1):  # Test first 5 voices
            print(f"[TTS] Testing voice {i}: {voice_info['name']}")
            success = self.speak_text_pyttsx3(
                f"Voice {i}: {voice_info['name']}. {test_text}",
                voice_id=voice_info['id']
            )
            if not success:
                print(f"[ERROR] Failed to test voice: {voice_info['name']}")
            time.sleep(1)  # Brief pause between tests

        return True

    def preview_voice(self, voice_id, test_text="This is a preview of the selected voice."):
        """Preview a specific voice."""
        success = self.speak_text_pyttsx3(test_text, voice_id=voice_id)
        return success

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

    def perform_search(self, query: str) -> str:
        """Perform a search using DuckDuckGo API and return a short summary."""
        if requests is None:
            print("[ERROR] Requests library not installed")
            return None

        try:
            # Use DuckDuckGo instant answers API
            url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Try to get instant answer or abstract
            if data.get('Answer'):
                return data['Answer']
            elif data.get('AbstractText'):
                return data['AbstractText']
            elif data.get('Definition'):
                return data['Definition']
            elif data.get('RelatedTopics'):
                # Get first related topic summary
                topics = data['RelatedTopics']
                if topics and isinstance(topics[0], dict) and 'Text' in topics[0]:
                    return topics[0]['Text']
            else:
                return f"I found information about {query}, but couldn't get a concise summary. You might want to search manually."

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Search request failed: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to perform search: {e}")
            return None

    def get_wikipedia_summary(self, topic: str) -> str:
        """Get Wikipedia summary for a topic, handling disambiguation."""
        if wikipedia is None:
            print("[ERROR] Wikipedia library not installed")
            return None

        try:
            # Set language to English
            wikipedia.set_lang("en")

            # Try to get the page
            page = wikipedia.page(topic, auto_suggest=True)

            # Get summary (first 3 sentences)
            summary = wikipedia.summary(topic, sentences=3, auto_suggest=True)

            return summary

        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation by listing options
            options = e.options[:5]  # Top 5 options
            return f"Multiple topics found for '{topic}'. Possible options: {', '.join(options)}. Please be more specific."

        except wikipedia.exceptions.PageError:
            return f"Sorry, I couldn't find a Wikipedia page for '{topic}'."

        except Exception as e:
            print(f"[ERROR] Failed to get Wikipedia summary: {e}")
            return None

    def get_wikipedia_page_info(self, topic: str) -> dict:
        """Get detailed page information from Wikipedia."""
        if wikipedia is None:
            print("[ERROR] Wikipedia library not installed")
            return None

        try:
            wikipedia.set_lang("en")
            page = wikipedia.page(topic, auto_suggest=True)

            info = {
                'title': page.title,
                'url': page.url,
                'summary': page.summary[:500] + '...' if len(page.summary) > 500 else page.summary,
                'categories': page.categories[:10],  # Top 10 categories
                'links': page.links[:10]  # Top 10 links
            }

            return info

        except wikipedia.exceptions.DisambiguationError as e:
            return {'error': 'disambiguation', 'options': e.options[:5]}

        except wikipedia.exceptions.PageError:
            return {'error': 'page_not_found'}

        except Exception as e:
            print(f"[ERROR] Failed to get Wikipedia page info: {e}")
            return None

    def handle_wikipedia_disambiguation(self, topic: str, choice: int = 0) -> str:
        """Handle disambiguation by selecting a specific option."""
        if wikipedia is None:
            print("[ERROR] Wikipedia library not installed")
            return None

        try:
            wikipedia.set_lang("en")
            # Get the disambiguation options
            search_results = wikipedia.search(topic)
            if choice < len(search_results):
                selected_topic = search_results[choice]
                return self.get_wikipedia_summary(selected_topic)
            else:
                return f"Invalid choice. Available options: {', '.join(search_results[:5])}"

        except Exception as e:
            print(f"[ERROR] Failed to handle disambiguation: {e}")
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

    def get_random_joke(self) -> str:
        """Get a random joke from all categories."""
        if pyjokes is None:
            print("[ERROR] pyjokes library not installed")
            return "Sorry, the joke library is not available."

        try:
            return pyjokes.get_joke()
        except Exception as e:
            print(f"[ERROR] Failed to get random joke: {e}")
            return "Sorry, I couldn't fetch a joke right now."

    def get_joke_by_category(self, category: str) -> str:
        """Get a random joke from a specific category (neutral, chuck, all)."""
        if pyjokes is None:
            print("[ERROR] pyjokes library not installed")
            return "Sorry, the joke library is not available."

        valid_categories = ['neutral', 'chuck', 'all']
        if category.lower() not in valid_categories:
            return f"Sorry, '{category}' is not a valid category. Available categories: {', '.join(valid_categories)}."

        try:
            return pyjokes.get_joke(category=category.lower())
        except Exception as e:
            print(f"[ERROR] Failed to get joke from category '{category}': {e}")
            return f"Sorry, I couldn't fetch a {category} joke right now."

    def get_programming_joke(self) -> str:
        """Get a programming-related joke."""
        # Programming jokes are typically in the 'neutral' category
        return self.get_joke_by_category('neutral')

    def search_youtube(self, query: str, max_results: int = 5) -> list:
        """Search YouTube videos using yt-dlp."""
        if yt_dlp is None:
            print("[ERROR] yt-dlp library not installed")
            return None

        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'force_generic_extractor': False,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Search using ytsearch
                search_url = f'ytsearch{max_results}:{query}'
                result = ydl.extract_info(search_url, download=False)

                if 'entries' in result:
                    results = []
                    for entry in result['entries'][:max_results]:
                        results.append({
                            'title': entry.get('title', 'Unknown'),
                            'url': entry.get('url', ''),
                            'author': entry.get('uploader', 'Unknown'),
                            'length': entry.get('duration', 0),
                            'views': entry.get('view_count', 0)
                        })
                    return results
                return None
        except Exception as e:
            print(f"[ERROR] YouTube search failed: {e}")
            return None

    def get_youtube_video_info(self, url_or_query: str) -> dict:
        """Get YouTube video information using yt-dlp."""
        if yt_dlp is None:
            print("[ERROR] yt-dlp library not installed")
            return None

        try:
            # If it's not a URL, try to search and get first result
            if not url_or_query.startswith(('http://', 'https://', 'www.youtube.com', 'youtu.be')):
                search_results = self.search_youtube(url_or_query, 1)
                if search_results:
                    url_or_query = search_results[0]['url']
                else:
                    return None

            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(url_or_query, download=False)

                info = {
                    'title': result.get('title', 'Unknown'),
                    'author': result.get('uploader', 'Unknown'),
                    'length': result.get('duration', 0),
                    'views': result.get('view_count', 0),
                    'description': result.get('description', '')[:200] + '...' if len(result.get('description', '')) > 200 else result.get('description', ''),
                    'url': result.get('webpage_url', url_or_query)
                }
                return info
        except Exception as e:
            print(f"[ERROR] Failed to get YouTube video info: {e}")
            return None

    def download_youtube_audio(self, url_or_query: str) -> bool:
        """Download YouTube audio with safety checks using yt-dlp."""
        if yt_dlp is None:
            print("[ERROR] yt-dlp library not installed")
            return False

        try:
            # If it's not a URL, try to search and get first result
            if not url_or_query.startswith(('http://', 'https://', 'www.youtube.com', 'youtu.be')):
                search_results = self.search_youtube(url_or_query, 1)
                if search_results:
                    url_or_query = search_results[0]['url']
                else:
                    return False

            # First get info to check duration
            ydl_opts_info = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }

            with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
                result = ydl.extract_info(url_or_query, download=False)

                # Safety checks
                duration = result.get('duration', 0)
                if duration > 3600:  # 1 hour limit
                    print("[ERROR] Video too long (>1 hour)")
                    return False

            # Safe download directory
            download_dir = os.path.join(os.getcwd(), 'downloads')
            os.makedirs(download_dir, exist_ok=True)

            # Download audio
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url_or_query])

            return True

        except Exception as e:
            print(f"[ERROR] YouTube audio download failed: {e}")
            return False

    def download_youtube_video(self, url_or_query: str) -> bool:
        """Download YouTube video with safety checks using yt-dlp."""
        if yt_dlp is None:
            print("[ERROR] yt-dlp library not installed")
            return False

        try:
            # If it's not a URL, try to search and get first result
            if not url_or_query.startswith(('http://', 'https://', 'www.youtube.com', 'youtu.be')):
                search_results = self.search_youtube(url_or_query, 1)
                if search_results:
                    url_or_query = search_results[0]['url']
                else:
                    return False

            # First get info to check duration
            ydl_opts_info = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }

            with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
                result = ydl.extract_info(url_or_query, download=False)

                # Safety checks
                duration = result.get('duration', 0)
                if duration > 1800:  # 30 minutes limit
                    print("[ERROR] Video too long (>30 minutes)")
                    return False

            # Safe download directory
            download_dir = os.path.join(os.getcwd(), 'downloads')
            os.makedirs(download_dir, exist_ok=True)

            # Download video (best quality but reasonable size)
            ydl_opts = {
                'format': 'best[height<=720]',  # Limit to 720p to control file size
                'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url_or_query])

            return True

        except Exception as e:
            print(f"[ERROR] YouTube video download failed: {e}")
            return False

    def get_current_location(self) -> dict:
        """Get current location using IP-based geolocation."""
        if geocoder is None:
            print("[ERROR] Geocoder library not installed")
            return None

        try:
            # Get location based on IP address
            g = geocoder.ip('me')

            if g.ok:
                location_data = {
                    'latitude': g.lat,
                    'longitude': g.lng,
                    'city': g.city,
                    'state': g.state,
                    'country': g.country,
                    'address': g.address
                }
                return location_data
            else:
                print(f"[ERROR] Geocoding failed: {g.status}")
                return None

        except Exception as e:
            print(f"[ERROR] Failed to get current location: {e}")
            return None

    def geocode_address(self, address: str) -> dict:
        """Convert address to coordinates using geocoding."""
        if geocoder is None:
            print("[ERROR] Geocoder library not installed")
            return None

        try:
            g = geocoder.osm(address)  # Using OpenStreetMap for free geocoding

            if g.ok:
                location_data = {
                    'latitude': g.lat,
                    'longitude': g.lng,
                    'address': g.address,
                    'city': g.city,
                    'state': g.state,
                    'country': g.country
                }
                return location_data
            else:
                print(f"[ERROR] Geocoding failed for address '{address}': {g.status}")
                return None

        except Exception as e:
            print(f"[ERROR] Failed to geocode address '{address}': {e}")
            return None

    def reverse_geocode(self, latitude: float, longitude: float) -> dict:
        """Convert coordinates to address using reverse geocoding."""
        if geocoder is None:
            print("[ERROR] Geocoder library not installed")
            return None

        try:
            g = geocoder.osm([latitude, longitude], method='reverse')

            if g.ok:
                location_data = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'address': g.address,
                    'city': g.city,
                    'state': g.state,
                    'country': g.country
                }
                return location_data
            else:
                print(f"[ERROR] Reverse geocoding failed for coordinates ({latitude}, {longitude}): {g.status}")
                return None

        except Exception as e:
            print(f"[ERROR] Failed to reverse geocode coordinates ({latitude}, {longitude}): {e}")
            return None

    def get_cpu_usage(self) -> str:
        """Get current CPU usage percentage."""
        if psutil is None:
            return "CPU monitoring is not available. Please install psutil."

        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            return f"Current CPU usage is {cpu_percent}%"
        except Exception as e:
            print(f"[ERROR] Failed to get CPU usage: {e}")
            return "Sorry, I couldn't retrieve CPU usage information."

    def get_memory_usage(self) -> str:
        """Get current memory usage information."""
        if psutil is None:
            return "Memory monitoring is not available. Please install psutil."

        try:
            memory = psutil.virtual_memory()
            used_gb = round(memory.used / (1024**3), 2)
            total_gb = round(memory.total / (1024**3), 2)
            percent = memory.percent
            return f"Memory usage: {used_gb} GB used out of {total_gb} GB total, which is {percent}%"
        except Exception as e:
            print(f"[ERROR] Failed to get memory usage: {e}")
            return "Sorry, I couldn't retrieve memory usage information."

    def get_disk_space(self) -> str:
        """Get disk space information for the main drive."""
        if psutil is None:
            return "Disk monitoring is not available. Please install psutil."

        try:
            disk = psutil.disk_usage('/')
            used_gb = round(disk.used / (1024**3), 2)
            total_gb = round(disk.total / (1024**3), 2)
            free_gb = round(disk.free / (1024**3), 2)
            percent = disk.percent
            return f"Disk space: {used_gb} GB used, {free_gb} GB free out of {total_gb} GB total, which is {percent}% used"
        except Exception as e:
            print(f"[ERROR] Failed to get disk space: {e}")
            return "Sorry, I couldn't retrieve disk space information."

    def get_battery_status(self) -> str:
        """Get battery status information (for laptops)."""
        if psutil is None:
            return "Battery monitoring is not available. Please install psutil."

        try:
            battery = psutil.sensors_battery()
            if battery is None:
                return "No battery detected. This might be a desktop computer."

            percent = battery.percent
            plugged = battery.power_plugged
            if plugged:
                status = "plugged in and charging"
            else:
                status = "on battery power"

            if battery.secsleft != psutil.POWER_TIME_UNLIMITED:
                hours, remainder = divmod(battery.secsleft, 3600)
                minutes, _ = divmod(remainder, 60)
                time_left = f"{hours} hours and {minutes} minutes"
            else:
                time_left = "unlimited (plugged in)"

            return f"Battery is at {percent}%, {status}. Time remaining: {time_left}"
        except Exception as e:
            print(f"[ERROR] Failed to get battery status: {e}")
            return "Sorry, I couldn't retrieve battery status information."

    def get_running_processes(self) -> str:
        """Get information about running processes."""
        if psutil is None:
            return "Process monitoring is not available. Please install psutil."

        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    info = proc.info
                    processes.append({
                        'pid': info['pid'],
                        'name': info['name'],
                        'cpu': info['cpu_percent'],
                        'memory': info['memory_percent']
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by CPU usage and get top 5
            processes.sort(key=lambda x: x['cpu'], reverse=True)
            top_processes = processes[:5]

            if not top_processes:
                return "No process information available."

            result = "Top 5 processes by CPU usage: "
            for proc in top_processes:
                result += f"{proc['name']} (PID: {proc['pid']}) using {proc['cpu']:.1f}% CPU and {proc['memory']:.1f}% memory; "

            return result.rstrip('; ')
        except Exception as e:
            print(f"[ERROR] Failed to get running processes: {e}")
            return "Sorry, I couldn't retrieve running process information."

    def get_network_info(self) -> str:
        """Get network interface information."""
        if psutil is None:
            return "Network monitoring is not available. Please install psutil."

        try:
            net_io = psutil.net_io_counters()
            sent_mb = round(net_io.bytes_sent / (1024**2), 2)
            recv_mb = round(net_io.bytes_recv / (1024**2), 2)

            # Get network interfaces
            interfaces = psutil.net_if_addrs()
            active_interfaces = []
            for name, addrs in interfaces.items():
                for addr in addrs:
                    if addr.family.name == 'AF_INET' and addr.address != '127.0.0.1':
                        active_interfaces.append(f"{name}: {addr.address}")

            if active_interfaces:
                interface_info = "; ".join(active_interfaces[:3])  # Top 3 interfaces
            else:
                interface_info = "No active network interfaces found"

            return f"Network: {sent_mb} MB sent, {recv_mb} MB received. Active interfaces: {interface_info}"
        except Exception as e:
            print(f"[ERROR] Failed to get network info: {e}")
            return "Sorry, I couldn't retrieve network information."

    def compare_prices(self, product: str) -> str:
        """Compare prices for a product from e-commerce sites."""
        if requests is None or BeautifulSoup is None:
            print("[ERROR] Requests or BeautifulSoup library not installed")
            return None

        try:
            import re
            # Use a reputable price comparison site like Google Shopping or similar
            # For demonstration, we'll use a simple approach with a search engine
            query = f"{product} price"
            url = f"https://www.google.com/search?q={quote(query)}&tbm=shop"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract price information (this is a simplified example)
            prices = []
            price_elements = soup.find_all(['span', 'div'], class_=re.compile(r'price|cost'))

            for element in price_elements[:5]:  # Limit to first 5 results
                price_text = element.get_text().strip()
                if '$' in price_text or '₹' in price_text or '£' in price_text:
                    prices.append(price_text)

            if prices:
                summary = f"Found prices ranging from {min(prices)} to {max(prices)}"
                return summary
            else:
                return f"Could not extract specific prices, but found information about {product}"

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Price comparison request failed: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to compare prices: {e}")
            return None

    def lookup_recipe(self, recipe_name: str) -> str:
        """Look up a recipe from cooking websites."""
        if requests is None or BeautifulSoup is None:
            print("[ERROR] Requests or BeautifulSoup library not installed")
            return None

        try:
            import re
            # Use AllRecipes or similar reputable site
            url = f"https://www.allrecipes.com/search?q={quote(recipe_name)}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract recipe information
            recipe_cards = soup.find_all('div', class_=re.compile(r'recipe|card'))

            if recipe_cards:
                first_recipe = recipe_cards[0]
                title = first_recipe.find(['h3', 'h2', 'a']).get_text().strip() if first_recipe.find(['h3', 'h2', 'a']) else recipe_name

                # Try to get ingredients and instructions
                ingredients = []
                instructions = []

                ingredient_elements = first_recipe.find_all(['li', 'span'], class_=re.compile(r'ingredient'))
                for ing in ingredient_elements[:8]:  # Limit ingredients
                    ingredients.append(ing.get_text().strip())

                instruction_elements = first_recipe.find_all(['li', 'p'], class_=re.compile(r'instruction|step'))
                for inst in instruction_elements[:5]:  # Limit steps
                    instructions.append(inst.get_text().strip())

                summary = f"Recipe for {title}. "
                if ingredients:
                    summary += f"Ingredients: {', '.join(ingredients[:5])}. "
                if instructions:
                    summary += f"Steps: {'; '.join(instructions[:3])}"

                return summary
            else:
                return f"Found recipe information for {recipe_name}"

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Recipe lookup request failed: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to lookup recipe: {e}")
            return None

    def define_word(self, word: str) -> str:
        """Get dictionary definition for a word."""
        if requests is None or BeautifulSoup is None:
            print("[ERROR] Requests or BeautifulSoup library not installed")
            return None

        try:
            import re
            # Use Merriam-Webster or similar dictionary site
            url = f"https://www.merriam-webster.com/dictionary/{quote(word)}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract definition
            definition_elements = soup.find_all(['span', 'p'], class_=re.compile(r'definition|meaning'))

            definitions = []
            for element in definition_elements[:3]:  # Get first 3 definitions
                text = element.get_text().strip()
                if text and len(text) > 10:  # Filter out short irrelevant text
                    definitions.append(text)

            if definitions:
                return '; '.join(definitions)
            else:
                # Fallback: try to get from the page content
                content = soup.find('div', class_=re.compile(r'content|entry'))
                if content:
                    text = content.get_text()[:300]  # Limit length
                    return text
                else:
                    return f"Definition found for {word}"

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Dictionary lookup request failed: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to define word: {e}")
            return None

    def get_stock_price(self, stock_symbol: str) -> str:
        """Get current stock price information."""
        if requests is None or BeautifulSoup is None:
            print("[ERROR] Requests or BeautifulSoup library not installed")
            return None

        try:
            import re
            # Use Yahoo Finance or similar
            url = f"https://finance.yahoo.com/quote/{quote(stock_symbol)}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract stock price
            price_element = soup.find(['span', 'div'], class_=re.compile(r'price|current'))
            if not price_element:
                # Try alternative selectors
                price_element = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})

            if price_element:
                price = price_element.get_text().strip()
                # Try to get company name
                name_element = soup.find(['h1', 'div'], class_=re.compile(r'name|symbol'))
                company_name = name_element.get_text().strip() if name_element else stock_symbol

                return f"{company_name} is trading at {price}"
            else:
                return f"Stock information found for {stock_symbol}"

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Stock price request failed: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to get stock price: {e}")
            return None

    def get_weather(self, location: str) -> str:
        """Get weather information for a location."""
        if requests is None or BeautifulSoup is None:
            print("[ERROR] Requests or BeautifulSoup library not installed")
            return None

        try:
            import re
            # Use Weather.com or similar reputable weather site
            url = f"https://weather.com/weather/today/l/{quote(location)}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract weather information
            temp_element = soup.find(['span', 'div'], class_=re.compile(r'temperature|temp'))
            condition_element = soup.find(['div', 'span'], class_=re.compile(r'condition|description'))

            temperature = temp_element.get_text().strip() if temp_element else "Unknown"
            condition = condition_element.get_text().strip() if condition_element else "Unknown"

            if temperature != "Unknown" or condition != "Unknown":
                return f"Temperature: {temperature}, Conditions: {condition}"
            else:
                return f"Weather information for {location}"

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Weather request failed: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to get weather: {e}")
            return None

    def get_windows_system_info(self) -> str:
        """Get Windows system information using win32api."""
        if not WIN32_AVAILABLE:
            return "Windows system information is not available. Please install pypiwin32."

        try:
            # Get OS version
            version = win32api.GetVersionEx()
            os_version = f"{version[0]}.{version[1]}.{version[2]}"

            # Get computer name
            computer_name = win32api.GetComputerName()

            # Get user name
            user_name = win32api.GetUserName()

            # Get system info
            system_info = win32api.GetSystemInfo()
            processor_count = system_info[5]  # dwNumberOfProcessors

            return f"OS Version: {os_version}, Computer Name: {computer_name}, User: {user_name}, Processors: {processor_count}"

        except Exception as e:
            print(f"[ERROR] Failed to get Windows system info: {e}")
            return "Sorry, I couldn't retrieve Windows system information."

    def create_file(self, file_path: str, content: str = "") -> bool:
        """Create a new file with optional content."""
        try:
            # Safety check: don't allow creating files in system directories
            system_dirs = ['C:\\Windows', 'C:\\Program Files', 'C:\\Program Files (x86)', 'C:\\System32']
            normalized_path = os.path.abspath(file_path)

            for sys_dir in system_dirs:
                if normalized_path.upper().startswith(sys_dir.upper()):
                    print(f"[ERROR] Cannot create files in system directory: {sys_dir}")
                    return False

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(normalized_path), exist_ok=True)

            with open(normalized_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return True

        except Exception as e:
            print(f"[ERROR] Failed to create file {file_path}: {e}")
            return False

    def delete_file(self, file_path: str) -> bool:
        """Delete a file safely."""
        try:
            # Safety check: don't allow deleting system files
            system_dirs = ['C:\\Windows', 'C:\\Program Files', 'C:\\Program Files (x86)', 'C:\\System32']
            normalized_path = os.path.abspath(file_path)

            for sys_dir in system_dirs:
                if normalized_path.upper().startswith(sys_dir.upper()):
                    print(f"[ERROR] Cannot delete files in system directory: {sys_dir}")
                    return False

            if os.path.exists(normalized_path):
                os.remove(normalized_path)
                return True
            else:
                print(f"[ERROR] File does not exist: {file_path}")
                return False

        except Exception as e:
            print(f"[ERROR] Failed to delete file {file_path}: {e}")
            return False

    def move_file(self, source_path: str, dest_path: str) -> bool:
        """Move a file from source to destination."""
        try:
            # Safety checks
            system_dirs = ['C:\\Windows', 'C:\\Program Files', 'C:\\Program Files (x86)', 'C:\\System32']
            normalized_source = os.path.abspath(source_path)
            normalized_dest = os.path.abspath(dest_path)

            for sys_dir in system_dirs:
                if normalized_source.upper().startswith(sys_dir.upper()) or normalized_dest.upper().startswith(sys_dir.upper()):
                    print(f"[ERROR] Cannot move files to/from system directory: {sys_dir}")
                    return False

            if os.path.exists(normalized_source):
                os.makedirs(os.path.dirname(normalized_dest), exist_ok=True)
                os.rename(normalized_source, normalized_dest)
                return True
            else:
                print(f"[ERROR] Source file does not exist: {source_path}")
                return False

        except Exception as e:
            print(f"[ERROR] Failed to move file from {source_path} to {dest_path}: {e}")
            return False

    def start_windows_service(self, service_name: str) -> bool:
        """Start a Windows service."""
        if not WIN32_AVAILABLE:
            print("[ERROR] Windows services management is not available. Please install pypiwin32.")
            return False

        try:
            win32serviceutil.StartService(service_name)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to start service {service_name}: {e}")
            return False

    def stop_windows_service(self, service_name: str) -> bool:
        """Stop a Windows service."""
        if not WIN32_AVAILABLE:
            print("[ERROR] Windows services management is not available. Please install pypiwin32.")
            return False

        try:
            win32serviceutil.StopService(service_name)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to stop service {service_name}: {e}")
            return False

    def read_registry_value(self, key_path: str, value_name: str) -> str:
        """Read a value from Windows registry safely."""
        if not WIN32_AVAILABLE:
            return "Windows registry access is not available. Please install pypiwin32."

        try:
            # Safety check: only allow access to safe registry keys
            safe_keys = [
                'HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run',
                'HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer',
                'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion',
                'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion'
            ]

            # Check if the key path starts with any safe key
            is_safe = False
            for safe_key in safe_keys:
                if key_path.upper().startswith(safe_key.upper()):
                    is_safe = True
                    break

            if not is_safe:
                return f"Access denied: Registry key {key_path} is not in the allowed list."

            # Open the registry key
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER if key_path.startswith('HKEY_CURRENT_USER') else winreg.HKEY_LOCAL_MACHINE,
                                key_path.split('\\', 1)[1], 0, winreg.KEY_READ)

            # Read the value
            value, value_type = winreg.QueryValueEx(key, value_name)
            winreg.CloseKey(key)

            return str(value)

        except FileNotFoundError:
            return f"Registry key or value not found: {key_path}\\{value_name}"
        except Exception as e:
            print(f"[ERROR] Failed to read registry value {key_path}\\{value_name}: {e}")
            return f"Error reading registry value: {str(e)}"

    def get_windows_event_logs(self, log_type: str = 'System', max_events: int = 10) -> str:
        """Read Windows event logs."""
        if not WIN32_AVAILABLE:
            return "Windows event log access is not available. Please install pypiwin32."

        try:
            # Open event log
            hand = win32eventlog.OpenEventLog(None, log_type)

            # Read events
            events = []
            flags = win32eventlog.EVENTLOG_BACKWARDS_READ | win32eventlog.EVENTLOG_SEQUENTIAL_READ
            total = win32eventlog.GetNumberOfEventLogRecords(hand)

            events_read = win32eventlog.ReadEventLog(hand, flags, 0)

            for event in events_read[:max_events]:
                try:
                    msg = win32evtlogutil.FormatMessage(event)
                    events.append(f"Event ID: {event.EventID}, Type: {event.EventType}, Message: {msg[:100]}...")
                except:
                    events.append(f"Event ID: {event.EventID}, Type: {event.EventType}")

            win32eventlog.CloseEventLog(hand)

            if events:
                return f"Recent {log_type} events: " + "; ".join(events)
            else:
                return f"No events found in {log_type} log."

        except Exception as e:
            print(f"[ERROR] Failed to read event logs: {e}")
            return f"Error reading event logs: {str(e)}"

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers using Haversine formula."""
        try:
            import math

            # Convert to radians
            lat1_rad = math.radians(lat1)
            lon1_rad = math.radians(lon1)
            lat2_rad = math.radians(lat2)
            lon2_rad = math.radians(lon2)

            # Haversine formula
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad

            a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

            # Earth's radius in kilometers
            radius = 6371.0

            distance = radius * c
            return round(distance, 2)

        except Exception as e:
            print(f"[ERROR] Failed to calculate distance: {e}")
            return None
