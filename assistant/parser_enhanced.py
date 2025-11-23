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
    WIKIPEDIA = "wikipedia"
    YOUTUBE = "youtube"
    TODO_GENERATION = "todo_generation"
    TODO_MANAGEMENT = "todo_management"
    JOKES = "jokes"
    LOCATION_SERVICES = "location_services"
    SYSTEM_MONITORING = "system_monitoring"
    PRICE_COMPARISON = "price_comparison"
    RECIPE_LOOKUP = "recipe_lookup"
    DICTIONARY = "dictionary"
    STOCK_PRICE = "stock_price"
    WEATHER = "weather"
    WINDOWS_SYSTEM_INFO = "windows_system_info"
    WINDOWS_SERVICES = "windows_services"
    WINDOWS_REGISTRY = "windows_registry"
    WINDOWS_EVENT_LOG = "windows_event_log"
    TTS_CONTROL = "tts_control"
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

        # Language settings
        self.language_config = self.config.get('language', {})
        self.current_language = self.language_config.get('default', 'en')
        self.supported_languages = self.language_config.get('supported', ['en'])

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
        patterns = {}

        # English patterns
        patterns.update({
            Intent.OPEN_APPLICATION: [
                (r'\b(open|launch|start|run)\s+(.+?)\s*(?:please|now)?$', 0.9),
                (r'\b(want|need)\s+(?:to\s+)?(open|launch|start|run)\s+(.+?)$', 0.8),
                (r'\b(please\s+)?(open|launch|start|run)\s+(.+?)$', 0.85),
            ],
        })

        # Hindi patterns
        if 'hi' in self.supported_languages:
            patterns[Intent.OPEN_APPLICATION].extend([
                (r'\b(खोलो|शुरू\s+करो|चलाओ)\s+(.+?)\s*(?:कृपया|अभी)?$', 0.9),
                (r'\b(मैं\s+चाहता\s+हूं|मुझे\s+चाहिए)\s+(?:कि\s+)?(खोलो|शुरू\s+करो|चलाओ)\s+(.+?)$', 0.8),
                (r'\b(कृपया\s+)?(खोलो|शुरू\s+करो|चलाओ)\s+(.+?)$', 0.85),
            ])

        # Continue with other intents...
        patterns.update({
            Intent.CLOSE_WINDOW: [
                (r'\b(close|shutdown|exit)\s+(?:this\s+)?(?:window|application|app)?$', 0.9),
                (r'\b(shut\s+down|exit\s+out\s+of)\s+(?:this\s+)?(?:window|app)?$', 0.8),
                (r'\b(quit|kill)\s+(?:this\s+)?(?:app|application)?$', 0.85),
            ],
        })

        if 'hi' in self.supported_languages:
            patterns[Intent.CLOSE_WINDOW].extend([
                (r'\b(बंद\s+करो|बंद)\s+(?:यह\s+)?(?:विंडो|एप्लिकेशन|ऐप)?$', 0.9),
                (r'\b(बंद\s+कर\s+दो|निकाल\s+दो)\s+(?:यह\s+)?(?:विंडो|ऐप)?$', 0.8),
                (r'\b(छोड़ो|मरो)\s+(?:यह\s+)?(?:ऐप|एप्लिकेशन)?$', 0.85),
            ])

        patterns.update({
            Intent.SYSTEM_CONTROL: [
                (r'\b(minimize|hide|maximize|show)\s+(?:this\s+)?(?:window|application)?$', 0.8),
                (r'\b(switch\s+to|switch\s+focus\s+to|go\s+to)\s+(.+?)$', 0.75),
                (r'\b(next\s+window|previous\s+window|alt\s+tab)$', 0.9),
            ],
        })

        if 'hi' in self.supported_languages:
            patterns[Intent.SYSTEM_CONTROL].extend([
                (r'\b(छोटा\s+करो|छिपाओ|बड़ा\s+करो|दिखाओ)\s+(?:यह\s+)?(?:विंडो|एप्लिकेशन)?$', 0.8),
                (r'\b(स्विच\s+करो|फोकस\s+करो|जाओ)\s+(.+?)$', 0.75),
                (r'\b(अगली\s+विंडो|पिछली\s+विंडो|ऑल्ट\s+टैब)$', 0.9),
            ])

        patterns.update({
            Intent.VOLUME_CONTROL: [
                (r'\b(volume|turn|increase|decrease)\s+(?:the\s+)?(?:volume|sound)\s*(?:up|down|higher|lower)?$', 0.9),
                (r'\b(make\s+it|turn\s+it)\s+(?:louder|quieter|higher|lower)$', 0.7),
                (r'\b(mute|unmute|silence)\s*(?:the\s+)?(?:volume|sound|computer)?$', 0.85),
            ],
        })

        if 'hi' in self.supported_languages:
            patterns[Intent.VOLUME_CONTROL].extend([
                (r'\b(वॉल्यूम|घुमाओ|बढ़ाओ|घटाओ)\s+(?:द\s+)?(?:वॉल्यूम|आवाज)\s*(?:ऊपर|नीचे|ज़्यादा|कम)?$', 0.9),
                (r'\b(बनाओ|करो)\s+(?:इसे\s+)?(?:ज़्यादा|कम|ऊँचा|नीचा)$', 0.7),
                (r'\b(म्यूट|अनम्यूट|चुप)\s*(?:करो\s+)?(?:वॉल्यूम|आवाज|कंप्यूटर)?$', 0.85),
            ])

        patterns.update({
            Intent.SCREENSHOT: [
                (r'\b(take\s+)?(a\s+)?screenshot\s*(?:please|now)?$', 0.9),
                (r'\b(capture\s+screen|screen\s+capture|screen\s+shot)$', 0.85),
                (r'\b(save\s+screen|save\s+this\s+screen)$', 0.7),
            ],
        })

        if 'hi' in self.supported_languages:
            patterns[Intent.SCREENSHOT].extend([
                (r'\b(लो\s+)?(एक\s+)?स्क्रीनशॉट\s*(?:कृपया|अभी)?$', 0.9),
                (r'\b(कैप्चर\s+स्क्रीन|स्क्रीन\s+कैप्चर|स्क्रीन\s+शॉट)$', 0.85),
                (r'\b(सेव\s+स्क्रीन|सेव\s+करो\s+यह\s+स्क्रीन)$', 0.7),
            ])

        patterns.update({
            Intent.TEXT_OPERATION: [
                (r'\b(copy|paste|cut|select\s+all)\s*(?:this|that|everything|all)?$', 0.9),
                (r'\b(undo|redo)\s*(?:that|last\s+action)?$', 0.8),
                (r'\b(save|save\s+as|print)\s*(?:this|that|document|file)?$', 0.85),
            ],
        })

        if 'hi' in self.supported_languages:
            patterns[Intent.TEXT_OPERATION].extend([
                (r'\b(कॉपी|पेस्ट|कट|सब\s+सेलेक्ट)\s*(?:यह|वह|सब|सब कुछ)?$', 0.9),
                (r'\b(अन्डू|रीडू)\s*(?:वह|आखिरी\s+एक्शन)?$', 0.8),
                (r'\b(सेव|सेव\s+ऐज़|प्रिंट)\s*(?:यह|वह|डॉक्यूमेंट|फाइल)?$', 0.85),
            ])

        patterns.update({
            Intent.WIKIPEDIA: [
                (r'\b(what\s+is|who\s+is|when\s+is|where\s+is|why\s+is|how\s+to)\s+(.+?)$', 0.8),
                (r'\b(tell\s+me\s+about|explain|describe)\s+(.+?)$', 0.85),
                (r'\b(wikipedia|wiki)\s+(.+?)$', 0.9),
                (r'\b(search\s+wikipedia\s+for|lookup)\s+(.+?)$', 0.85),
            ],
        })

        patterns.update({
            Intent.SEARCH: [
                (r'\b(search\s+for|find|look\s+for)\s+(.+?)$', 0.9),
                (r'\b(google\s+|bing\s+|yahoo\s+)\s*(.+?)$', 0.85),
            ],
        })

        if 'hi' in self.supported_languages:
            patterns[Intent.SEARCH].extend([
                (r'\b(खोजो|ढूंढो|देखो)\s+(.+?)$', 0.9),
                (r'\b(गूगल\s+|बिंग\s+|याहू\s+)\s*(.+?)$', 0.85),
            ])

        patterns.update({
            Intent.WEB_BROWSING: [
                (r'\b(go\s+to|open|visit|navigate\s+to)\s+(?:website\s+)?(.+?)$', 0.8),
                (r'\b(browse\s+to|surf\s+to)\s+(.+?)$', 0.75),
                (r'\b(check\s+out|look\s+at)\s+(.+?)$', 0.6),
            ],
        })

        if 'hi' in self.supported_languages:
            patterns[Intent.WEB_BROWSING].extend([
                (r'\b(जाओ|खोलो|विजिट\s+करो|नेविगेट\s+करो)\s+(?:वेबसाइट\s+)?(.+?)$', 0.8),
                (r'\b(ब्राउज़\s+करो|सर्फ\s+करो)\s+(.+?)$', 0.75),
                (r'\b(चेक\s+करो|देखो)\s+(.+?)$', 0.6),
            ])

        patterns.update({
            Intent.SWITCH_MODE: [
                (r'\b(start|begin|enter)\s+(?:dictation|dictation\s+mode)$', 1.0),
                (r'\b(stop|end|exit)\s+(?:dictation|dictation\s+mode)$', 1.0),
                (r'\b(switch\s+to|change\s+to)\s+(command|dictation)\s+mode$', 0.9),
            ],
        })

        if 'hi' in self.supported_languages:
            patterns[Intent.SWITCH_MODE].extend([
                (r'\b(शुरू|बEGIN|एंटर)\s+(?:डिक्टेशन|डिक्टेशन\s+मोड)$', 1.0),
                (r'\b(रोको|एंड|एग्जिट)\s+(?:डिक्टेशन|डिक्टेशन\s+मोड)$', 1.0),
                (r'\b(स्विच\s+करो|चेंज\s+करो)\s+(कमांड|डिक्टेशन)\s+मोड$', 0.9),
            ])

        patterns.update({
            Intent.NEWS_REPORTING: [
                (r'\b(what\'s\s+the\s+news|tell\s+me\s+the\s+news|get\s+news|news\s+update|latest\s+news)\s*(?:please|now)?$', 0.9),
                (r'\b(give\s+me\s+news|fetch\s+news|news\s+report)\s*(?:please|now)?$', 0.85),
                (r'\b(what\'s\s+happening|what\'s\s+going\s+on|current\s+events)\s*(?:in\s+the\s+world)?$', 0.7),
            ],
        })

        patterns.update({
            Intent.WIKIPEDIA: [
                (r'\b(what\s+is|who\s+is|when\s+is|where\s+is|why\s+is|how\s+to)\s+(.+?)$', 0.8),
                (r'\b(tell\s+me\s+about|explain|describe)\s+(.+?)$', 0.85),
                (r'\b(wikipedia|wiki)\s+(.+?)$', 0.9),
                (r'\b(search\s+wikipedia\s+for|lookup)\s+(.+?)$', 0.85),
            ],
        })

        patterns.update({
            Intent.YOUTUBE: [
                (r'\b(search\s+youtube\s+for|youtube\s+search)\s+(.+?)$', 0.9),
                (r'\b(find\s+(?:youtube\s+)?videos?\s+(?:about|for))\s+(.+?)$', 0.85),
                (r'\b(get\s+(?:video\s+)?info\s+(?:for|about))\s+(.+?)$', 0.9),
                (r'\b(download\s+(?:youtube\s+)?(?:audio|sound|music)\s+(?:from|for))\s+(.+?)$', 0.9),
                (r'\b(download\s+(?:youtube\s+)?video\s+(?:from|for))\s+(.+?)$', 0.9),
                (r'\b(youtube\s+(?:audio|video)\s+download)\s+(.+?)$', 0.85),
            ],
        })

        if 'hi' in self.supported_languages:
            patterns[Intent.YOUTUBE].extend([
                (r'\b(यूट्यूब\s+खोजो|यूट्यूब\s+में\s+ढूंढो)\s+(.+?)$', 0.9),
                (r'\b((?:यूट्यूब\s+)?वीडियो\s+ढूंढो\s+(?:के\s+बारे\s+में|के\s+लिए))\s+(.+?)$', 0.85),
                (r'\b((?:वीडियो\s+)?जानकारी\s+लो\s+(?:के\s+बारे\s+में|के\s+लिए))\s+(.+?)$', 0.9),
                (r'\b(डाउनलोड\s+करो\s+(?:यूट्यूब\s+)?(?:ऑडियो|साउंड|म्यूजिक)\s+(?:से|के\s+लिए))\s+(.+?)$', 0.9),
                (r'\b(डाउनलोड\s+करो\s+(?:यूट्यूब\s+)?वीडियो\s+(?:से|के\s+लिए))\s+(.+?)$', 0.9),
                (r'\b(यूट्यूब\s+(?:ऑडियो|वीडियो)\s+डाउनलोड)\s+(.+?)$', 0.85),
            ])

        if 'hi' in self.supported_languages:
            patterns[Intent.NEWS_REPORTING].extend([
                (r'\b(क्या\s+खबर\s+है|मुझे\s+खबर\s+बताओ|खबर\s+लाओ|खबर\s+अपडेट|ताज़ा\s+खबर)\s*(?:कृपया|अभी)?$', 0.9),
                (r'\b(मुझे\s+खबर\s+दो|खबर\s+फेच\s+करो|खबर\s+रिपोर्ट)\s*(?:कृपया|अभी)?$', 0.85),
                (r'\b(क्या\s+हो\s+रहा\s+है|क्या\s+चल\s+रहा\s+है|मौजूदा\s+घटनाएं)\s*(?:दुनिया\s+में)?$', 0.7),
            ])

        patterns.update({
            Intent.TODO_GENERATION: [
                (r'\b(create|make|generate)\s+(?:a\s+)?(?:todo|to-do|task)\s+list\s+(?:for\s+)?(.+?)$', 0.9),
                (r'\b(start\s+)?(?:a\s+)?(?:todo|to-do|task)\s+list\s+(?:called|named)\s+(.+?)$', 0.8),
            ],
        })

        if 'hi' in self.supported_languages:
            patterns[Intent.TODO_GENERATION].extend([
                (r'\b(बनाओ|करो|जनरेट\s+करो)\s+(?:एक\s+)?(?:टूडू|टू-डू|टास्क)\s+लिस्ट\s+(?:के\s+लिए\s+)?(.+?)$', 0.9),
                (r'\b(शुरू\s+)?(?:एक\s+)?(?:टूडू|टू-डू|टास्क)\s+लिस्ट\s+(?:कॉल्ड|नेम्ड)\s+(.+?)$', 0.8),
            ])

        patterns.update({
            Intent.TODO_MANAGEMENT: [
                (r'\b(show|list|display|get)\s+(?:my\s+)?(?:todo|to-do|task)\s+lists?$', 0.9),
                (r'\b(add\s+task|add\s+to\s+todo|add\s+to\s+list)\s+(.+?)$', 0.85),
                (r'\b(remove|delete)\s+(?:task\s+)?(.+?)\s+from\s+(?:todo|to-do|list)$', 0.8),
                (r'\b(mark\s+(.+?)\s+as\s+(?:done|completed)|complete\s+(.+?))\s*$', 0.85),
                (r'\b(what\s+are\s+my\s+tasks|what\s+do\s+i\s+have\s+to\s+do)\s*(?:today|now)?$', 0.8),
            ],
        })

        patterns.update({
            Intent.JOKES: [
                (r'\b(tell\s+me\s+a\s+joke|tell\s+a\s+joke|make\s+me\s+laugh)\s*(?:please|now)?$', 0.9),
                (r'\b(joke|jokes)\s*(?:please|now)?$', 0.8),
                (r'\b(programming\s+joke|tech\s+joke|computer\s+joke)\s*(?:please|now)?$', 0.85),
                (r'\b(chuck\s+norris\s+joke|chuck\s+joke)\s*(?:please|now)?$', 0.85),
                (r'\b(neutral\s+joke|clean\s+joke)\s*(?:please|now)?$', 0.85),
            ],
        })

        patterns.update({
            Intent.LOCATION_SERVICES: [
                (r'\breverse\s+geocode\s+(.+?)(?:\s+please|\s+now)?$', 0.85),
                (r'\bfind\s+address\s+(?:for|of)\s+coordinates?\s+(.+?)(?:\s+please|\s+now)?$', 0.85),
                (r'\bfind\s+coordinates\s+(?:for|of)\s+(.+?)(?:\s+please|\s+now)?$', 0.85),
                (r'\bgeocode\s+(?:address\s+)?(.+?)(?:\s+please|\s+now)?$', 0.85),
                (r'\bhow\s+far\s+is\s+(.+?)\s+from\s+(.+?)(?:\s+please|\s+now)?$', 0.9),
                (r'\bdistance\s+(?:between|from)\s+(.+?)\s+(?:to|and)\s+(.+?)(?:\s+please|\s+now)?$', 0.9),
                (r'\bcalculate\s+distance\s+(?:between|from)\s+(.+?)\s+(?:to|and)\s+(.+?)(?:\s+please|\s+now)?$', 0.85),
                (r'\b(where\s+am\s+i|what\'s\s+my\s+location|what\s+is\s+my\s+location)\s*(?:\?|please|now)?$', 0.9),
                (r'\b(find\s+(?:my\s+)?(?:current\s+)?location|get\s+(?:my\s+)?(?:current\s+)?location)\s*(?:please|now)?$', 0.9),
            ],
        })

        patterns.update({
            Intent.SYSTEM_MONITORING: [
                (r'\b(what\'s\s+my\s+cpu\s+usage|cpu\s+usage|how\s+much\s+cpu\s+is\s+being\s+used)\s*(?:\?|please|now)?$', 0.9),
                (r'\b(memory\s+usage|how\s+much\s+memory\s+is\s+(?:being\s+)?used|ram\s+usage)\s*(?:\?|please|now)?$', 0.9),
                (r'\b(how\s+much\s+memory\s+is\s+free|free\s+memory|available\s+memory)\s*(?:\?|please|now)?$', 0.85),
                (r'\b(disk\s+space|how\s+much\s+disk\s+space|storage\s+usage)\s*(?:\?|please|now)?$', 0.9),
                (r'\b(battery\s+status|battery\s+level|how\s+much\s+battery\s+is\s+left)\s*(?:\?|please|now)?$', 0.9),
                (r'\b(running\s+processes|what\s+processes\s+are\s+running|active\s+processes)\s*(?:\?|please|now)?$', 0.9),
                (r'\b(network\s+info|network\s+information|network\s+status)\s*(?:\?|please|now)?$', 0.9),
            ],
        })

        patterns.update({
            Intent.PRICE_COMPARISON: [
                (r'\b(compare\s+prices?\s+(?:for|of)\s+(.+?))\s*(?:please|now)?$', 0.9),
                (r'\b(price\s+comparison\s+(?:for|of)\s+(.+?))\s*(?:please|now)?$', 0.9),
                (r'\b(how\s+much\s+(?:does|is)\s+(.+?)\s+cost)\s*(?:\?|please|now)?$', 0.8),
                (r'\b(find\s+best\s+price\s+(?:for|of)\s+(.+?))\s*(?:please|now)?$', 0.85),
            ],
        })

        patterns.update({
            Intent.RECIPE_LOOKUP: [
                (r'\b(find\s+recipe\s+(?:for|of)\s+(.+?))\s*(?:please|now)?$', 0.9),
                (r'\b(recipe\s+(?:for|of)\s+(.+?))\s*(?:please|now)?$', 0.9),
                (r'\b(how\s+to\s+(?:make|cook|prepare)\s+(.+?))\s*(?:please|now)?$', 0.8),
                (r'\b(search\s+recipe\s+(?:for|of)\s+(.+?))\s*(?:please|now)?$', 0.85),
            ],
        })

        patterns.update({
            Intent.DICTIONARY: [
                (r'\b(define\s+(.+?))\s*(?:please|now)?$', 0.9),
                (r'\b(what\s+(?:does|is)\s+(.+?)\s+mean)\s*(?:\?|please|now)?$', 0.9),
                (r'\b(meaning\s+of\s+(.+?))\s*(?:please|now)?$', 0.85),
                (r'\b(lookup\s+(.+?)\s+in\s+dictionary)\s*(?:please|now)?$', 0.8),
            ],
        })

        patterns.update({
            Intent.STOCK_PRICE: [
                (r'\b(stock\s+price\s+(?:of|for)\s+(.+?))\s*(?:please|now)?$', 0.9),
                (r'\b(how\s+much\s+is\s+(.+?)\s+stock)\s*(?:\?|please|now)?$', 0.9),
                (r'\b(current\s+price\s+of\s+(.+?)\s+shares?)\s*(?:please|now)?$', 0.85),
                (r'\b(.+?)\s+stock\s+price\s*(?:please|now)?$', 0.8),
            ],
        })

        patterns.update({
            Intent.WEATHER: [
                (r'\b(weather\s+(?:in|for|at)\s+(.+?))\s*(?:please|now)?$', 0.9),
                (r'\b(what\'s\s+the\s+weather\s+(?:like\s+)?(?:in|for|at)\s+(.+?))\s*(?:\?|please|now)?$', 0.9),
                (r'\b(how\s+is\s+the\s+weather\s+(?:in|for|at)\s+(.+?))\s*(?:please|now)?$', 0.85),
                (r'\b(temperature\s+(?:in|for|at)\s+(.+?))\s*(?:please|now)?$', 0.8),
            ],
        })

        patterns.update({
            Intent.WINDOWS_SYSTEM_INFO: [
                (r'\b(system\s+info|windows\s+info|computer\s+info)\s*(?:please|now)?$', 0.9),
                (r'\b(what\s+is\s+my\s+(?:system|windows|computer)\s+info)\s*(?:\?|please|now)?$', 0.9),
                (r'\b(show\s+(?:system|windows|computer)\s+information)\s*(?:please|now)?$', 0.85),
            ],
        })

        patterns.update({
            Intent.FILE_OPERATION: [
                (r'\b(create\s+(?:a\s+)?(?:new\s+)?file\s+(?:called|named)\s+(.+?))\s*(?:please|now)?$', 0.9),
                (r'\b(delete\s+(?:the\s+)?file\s+(.+?))\s*(?:please|now)?$', 0.9),
                (r'\b(move\s+(?:the\s+)?file\s+(.+?)\s+to\s+(.+?))\s*(?:please|now)?$', 0.9),
                (r'\b(remove\s+(?:the\s+)?file\s+(.+?))\s*(?:please|now)?$', 0.85),
            ],
        })

        patterns.update({
            Intent.WINDOWS_SERVICES: [
                (r'\b(start\s+(?:the\s+)?(?:windows\s+)?service\s+(.+?))\s*(?:please|now)?$', 0.9),
                (r'\b(stop\s+(?:the\s+)?(?:windows\s+)?service\s+(.+?))\s*(?:please|now)?$', 0.9),
                (r'\b(restart\s+(?:the\s+)?(?:windows\s+)?service\s+(.+?))\s*(?:please|now)?$', 0.85),
            ],
        })

        patterns.update({
            Intent.WINDOWS_REGISTRY: [
                (r'\b(read\s+(?:registry\s+)?(?:key|value)\s+(.+?))\s*(?:please|now)?$', 0.9),
                (r'\b(get\s+(?:registry\s+)?(?:key|value)\s+(.+?))\s*(?:please|now)?$', 0.9),
                (r'\b(show\s+(?:registry\s+)?(?:key|value)\s+(.+?))\s*(?:please|now)?$', 0.85),
            ],
        })

        patterns.update({
            Intent.WINDOWS_EVENT_LOG: [
                (r'\b(show\s+(?:windows\s+)?(?:event\s+)?logs)\s*(?:please|now)?$', 0.9),
                (r'\b(get\s+(?:windows\s+)?(?:event\s+)?logs)\s*(?:please|now)?$', 0.9),
                (r'\b(display\s+(?:windows\s+)?(?:event\s+)?logs)\s*(?:please|now)?$', 0.85),
                (r'\b(system\s+logs|event\s+log)\s*(?:please|now)?$', 0.8),
            ],
        })

        patterns.update({
            Intent.TTS_CONTROL: [
                (r'\b(change\s+voice\s+to\s+(?:male|female|man|woman))\s*(?:please|now)?$', 0.9),
                (r'\b(set\s+voice\s+to\s+(?:male|female|man|woman))\s*(?:please|now)?$', 0.9),
                (r'\b(use\s+(?:male|female|man|woman)\s+voice)\s*(?:please|now)?$', 0.9),
                (r'\b(speak\s+faster|speak\s+slower|increase\s+speed|decrease\s+speed)\s*(?:please|now)?$', 0.9),
                (r'\b(set\s+speech\s+rate\s+to\s+(\d+))\s*(?:please|now)?$', 0.9),
                (r'\b(save\s+(?:this\s+)?(?:text|speech)\s+to\s+(?:audio\s+)?file\s*(?:called|named)?\s*(.+?))\s*(?:please|now)?$', 0.9),
                (r'\b(save\s+(?:this\s+)?(?:text|speech)\s+to\s+audio\s*(?:file)?\s*(.+?))\s*(?:please|now)?$', 0.85),
                (r'\b(test\s+voices|preview\s+voices|try\s+voices)\s*(?:please|now)?$', 0.9),
                (r'\b(show\s+available\s+voices|list\s+voices|get\s+voices)\s*(?:please|now)?$', 0.9),
                (r'\b(set\s+volume\s+to\s+(\d+(?:\.\d+)?)|increase\s+volume|decrease\s+volume)\s*(?:please|now)?$', 0.9),
                (r'\b(preview\s+voice|test\s+voice)\s+(\d+|\w+)\s*(?:please|now)?$', 0.8),
                (r'\b(speak\s+with\s+voice\s+(\d+|\w+))\s*(?:please|now)?$', 0.8),
            ],
        })

        if 'hi' in self.supported_languages:
            patterns[Intent.TODO_MANAGEMENT].extend([
                (r'\b(दिखाओ|लिस्ट|डिस्प्ले|गेट)\s+(?:मेरी\s+)?(?:टूडू|टू-डू|टास्क)\s+लिस्ट.?$', 0.9),
                (r'\b(ऐड\s+टास्क|ऐड\s+टू\s+टूडू|ऐड\s+टू\s+लिस्ट)\s+(.+?)$', 0.85),
                (r'\b(रिमूव|डिलीट)\s+(?:टास्क\s+)?(.+?)\s+फ्रॉम\s+(?:टूडू|टू-डू|लिस्ट)$', 0.8),
                (r'\b(मार्क\s+(.+?)\s+ऐज़\s+(?:डन|कंप्लीटेड)|कंप्लीट\s+(.+?))\s*$', 0.85),
                (r'\b(क्या\s+हैं\s+मेरे\s+टास्क|मुझे\s+क्या\s+करना\s+है)\s*(?:आज|अभी)?$', 0.8),
            ])

        return patterns

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

        elif intent == Intent.WIKIPEDIA:
            # Extract wikipedia topic
            if match.lastindex >= 2:
                topic = match.group(2)
            else:
                topic = match.group(1)
            parameters['topic'] = topic.strip()
        
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

        elif intent == Intent.JOKES:
            # Extract joke type
            if 'programming' in text or 'tech' in text or 'computer' in text:
                parameters['joke_type'] = 'programming'
            elif 'chuck' in text or 'norris' in text:
                parameters['joke_type'] = 'chuck'
            elif 'neutral' in text or 'clean' in text:
                parameters['joke_type'] = 'neutral'
            else:
                parameters['joke_type'] = 'random'

        elif intent == Intent.YOUTUBE:
            # Extract YouTube action and query/URL
            if 'search' in text or 'find' in text:
                parameters['action'] = 'search'
                if match.lastindex >= 2:
                    parameters['query'] = match.group(2).strip()
            elif 'info' in text:
                parameters['action'] = 'info'
                if match.lastindex >= 2:
                    parameters['query'] = match.group(2).strip()
            elif 'audio' in text or 'sound' in text or 'music' in text:
                parameters['action'] = 'download_audio'
                if match.lastindex >= 2:
                    parameters['query'] = match.group(2).strip()
            elif 'video' in text:
                parameters['action'] = 'download_video'
                if match.lastindex >= 2:
                    parameters['query'] = match.group(2).strip()

        elif intent == Intent.LOCATION_SERVICES:
            # Extract location service action and parameters
            if 'where am i' in text or 'what\'s my location' in text or 'what is my location' in text or 'find my location' in text or 'get my location' in text:
                parameters['action'] = 'current_location'
            elif 'reverse geocode' in text:
                parameters['action'] = 'reverse_geocode'
                if match and len(match.groups()) >= 1:
                    # Try to parse coordinates from the match
                    coord_text = match.group(1).strip()
                    # Look for latitude,longitude pattern
                    import re as regex
                    coord_match = regex.search(r'(-?\d+\.?\d*),\s*(-?\d+\.?\d*)', coord_text)
                    if coord_match:
                        parameters['latitude'] = float(coord_match.group(1))
                        parameters['longitude'] = float(coord_match.group(2))
                    else:
                        parameters['coordinates'] = coord_text
            elif 'find address' in text:
                parameters['action'] = 'reverse_geocode'
                if match and len(match.groups()) >= 1:
                    # Try to parse coordinates from the match
                    coord_text = match.group(1).strip()
                    # Look for latitude,longitude pattern
                    import re as regex
                    coord_match = regex.search(r'(-?\d+\.?\d*),\s*(-?\d+\.?\d*)', coord_text)
                    if coord_match:
                        parameters['latitude'] = float(coord_match.group(1))
                        parameters['longitude'] = float(coord_match.group(2))
                    else:
                        parameters['coordinates'] = coord_text
            elif 'find coordinates' in text:
                parameters['action'] = 'geocode'
                if match and len(match.groups()) >= 1:
                    parameters['address'] = match.group(1).strip()
            elif 'geocode' in text:
                parameters['action'] = 'geocode'
                if match and len(match.groups()) >= 1:
                    parameters['address'] = match.group(1).strip()
            elif 'how far' in text or 'distance' in text or 'calculate distance' in text:
                parameters['action'] = 'calculate_distance'
                if match and len(match.groups()) >= 2:
                    parameters['location1'] = match.group(1).strip()
                    parameters['location2'] = match.group(2).strip()

        elif intent == Intent.SYSTEM_MONITORING:
            # Extract system monitoring action
            if 'cpu' in text.lower():
                parameters['action'] = 'cpu_usage'
            elif 'memory' in text.lower() and 'free' in text.lower():
                parameters['action'] = 'memory_free'
            elif 'memory' in text.lower():
                parameters['action'] = 'memory_usage'
            elif 'disk' in text.lower() or 'storage' in text.lower():
                parameters['action'] = 'disk_space'
            elif 'battery' in text.lower():
                parameters['action'] = 'battery_status'
            elif 'process' in text.lower():
                parameters['action'] = 'running_processes'
            elif 'network' in text.lower():
                parameters['action'] = 'network_info'

        elif intent == Intent.PRICE_COMPARISON:
            # Extract product name
            if match.lastindex >= 2:
                parameters['product'] = match.group(2).strip()
            else:
                parameters['product'] = match.group(1).strip()

        elif intent == Intent.RECIPE_LOOKUP:
            # Extract recipe name
            if match.lastindex >= 2:
                parameters['recipe'] = match.group(2).strip()
            else:
                parameters['recipe'] = match.group(1).strip()

        elif intent == Intent.DICTIONARY:
            # Extract word to define
            if match.lastindex >= 2:
                parameters['word'] = match.group(2).strip()
            else:
                parameters['word'] = match.group(1).strip()

        elif intent == Intent.STOCK_PRICE:
            # Extract stock symbol or company name
            if match.lastindex >= 2:
                parameters['stock'] = match.group(2).strip()
            else:
                parameters['stock'] = match.group(1).strip()

        elif intent == Intent.WEATHER:
            # Extract location
            if match.lastindex >= 2:
                parameters['location'] = match.group(2).strip()
            else:
                parameters['location'] = match.group(1).strip()

        elif intent == Intent.WINDOWS_SYSTEM_INFO:
            # No parameters needed
            pass

        elif intent == Intent.FILE_OPERATION:
            # Extract file operation details
            if 'create' in text.lower():
                parameters['action'] = 'create'
                if match.lastindex >= 2:
                    parameters['file_path'] = match.group(2).strip()
            elif 'delete' in text.lower() or 'remove' in text.lower():
                parameters['action'] = 'delete'
                if match.lastindex >= 2:
                    parameters['file_path'] = match.group(2).strip()
            elif 'move' in text.lower():
                parameters['action'] = 'move'
                if match.lastindex >= 3:
                    parameters['source_path'] = match.group(2).strip()
                    parameters['dest_path'] = match.group(3).strip()

        elif intent == Intent.WINDOWS_SERVICES:
            # Extract service action and name
            if 'start' in text.lower():
                parameters['action'] = 'start'
            elif 'stop' in text.lower():
                parameters['action'] = 'stop'
            elif 'restart' in text.lower():
                parameters['action'] = 'restart'

            if match.lastindex >= 2:
                parameters['service_name'] = match.group(2).strip()

        elif intent == Intent.WINDOWS_REGISTRY:
            # Extract registry key path
            if match.lastindex >= 2:
                key_path = match.group(2).strip()
                # Try to extract key and value name
                if '\\' in key_path:
                    parts = key_path.split('\\')
                    if len(parts) >= 2:
                        parameters['key_path'] = '\\'.join(parts[:-1])
                        parameters['value_name'] = parts[-1]
                    else:
                        parameters['key_path'] = key_path
                        parameters['value_name'] = ''
                else:
                    parameters['key_path'] = key_path
                    parameters['value_name'] = ''

        elif intent == Intent.WINDOWS_EVENT_LOG:
            # Extract log type (default to System)
            if 'system' in text.lower():
                parameters['log_type'] = 'System'
            elif 'application' in text.lower():
                parameters['log_type'] = 'Application'
            elif 'security' in text.lower():
                parameters['log_type'] = 'Security'
            else:
                parameters['log_type'] = 'System'

        elif intent == Intent.TTS_CONTROL:
            # Extract TTS control parameters
            text_lower = text.lower()

            # Voice selection
            if 'change voice to' in text_lower or 'set voice to' in text_lower or 'use' in text_lower:
                if 'male' in text_lower or 'man' in text_lower:
                    parameters['action'] = 'set_voice'
                    parameters['gender'] = 'male'
                elif 'female' in text_lower or 'woman' in text_lower:
                    parameters['action'] = 'set_voice'
                    parameters['gender'] = 'female'

            # Speech rate control
            elif 'speak faster' in text_lower or 'increase speed' in text_lower:
                parameters['action'] = 'set_rate'
                parameters['rate'] = 'faster'  # Will be handled in execution
            elif 'speak slower' in text_lower or 'decrease speed' in text_lower:
                parameters['action'] = 'set_rate'
                parameters['rate'] = 'slower'  # Will be handled in execution
            elif 'set speech rate to' in text_lower:
                parameters['action'] = 'set_rate'
                # Extract the rate number from the regex match
                if match and len(match.groups()) >= 2:
                    try:
                        parameters['rate'] = int(match.group(2))
                    except ValueError:
                        parameters['rate'] = 180  # Default

            # Volume control
            elif 'set volume to' in text_lower:
                parameters['action'] = 'set_volume'
                if match and len(match.groups()) >= 2:
                    try:
                        parameters['volume'] = float(match.group(2))
                    except ValueError:
                        parameters['volume'] = 0.8  # Default
            elif 'increase volume' in text_lower:
                parameters['action'] = 'set_volume'
                parameters['volume'] = 'increase'
            elif 'decrease volume' in text_lower:
                parameters['action'] = 'set_volume'
                parameters['volume'] = 'decrease'

            # Save to file
            elif 'save' in text_lower and ('text' in text_lower or 'speech' in text_lower) and 'to' in text_lower:
                parameters['action'] = 'save_to_file'
                # Extract filename from the match
                if match and len(match.groups()) >= 2:
                    filename = match.group(2).strip()
                    if not filename.lower().endswith(('.mp3', '.wav')):
                        filename += '.mp3'
                    parameters['filename'] = filename

            # Test/preview voices
            elif 'test voices' in text_lower or 'preview voices' in text_lower or 'try voices' in text_lower:
                parameters['action'] = 'test_voices'
            elif 'show available voices' in text_lower or 'list voices' in text_lower or 'get voices' in text_lower:
                parameters['action'] = 'list_voices'
            elif 'preview voice' in text_lower or 'test voice' in text_lower:
                parameters['action'] = 'preview_voice'
                if match and len(match.groups()) >= 2:
                    parameters['voice_id'] = match.group(2).strip()
            elif 'speak with voice' in text_lower:
                parameters['action'] = 'speak_with_voice'
                if match and len(match.groups()) >= 2:
                    parameters['voice_id'] = match.group(2).strip()

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

            elif result.intent == Intent.WIKIPEDIA:
                return self._handle_wikipedia(result)

            elif result.intent == Intent.TODO_GENERATION:
                return self._handle_todo_generation(result)

            elif result.intent == Intent.TODO_MANAGEMENT:
                return self._handle_todo_management(result)

            elif result.intent == Intent.JOKES:
                return self._handle_jokes(result)

            elif result.intent == Intent.YOUTUBE:
                return self._handle_youtube(result)

            elif result.intent == Intent.LOCATION_SERVICES:
                return self._handle_location_services(result)

            elif result.intent == Intent.SYSTEM_MONITORING:
                return self._handle_system_monitoring(result)

            elif result.intent == Intent.PRICE_COMPARISON:
                return self._handle_price_comparison(result)

            elif result.intent == Intent.RECIPE_LOOKUP:
                return self._handle_recipe_lookup(result)

            elif result.intent == Intent.DICTIONARY:
                return self._handle_dictionary(result)

            elif result.intent == Intent.STOCK_PRICE:
                return self._handle_stock_price(result)

            elif result.intent == Intent.WEATHER:
                return self._handle_weather(result)

            elif result.intent == Intent.WINDOWS_SYSTEM_INFO:
                return self._handle_windows_system_info(result)

            elif result.intent == Intent.FILE_OPERATION:
                return self._handle_file_operation(result)

            elif result.intent == Intent.WINDOWS_SERVICES:
                return self._handle_windows_services(result)

            elif result.intent == Intent.WINDOWS_REGISTRY:
                return self._handle_windows_registry(result)

            elif result.intent == Intent.WINDOWS_EVENT_LOG:
                return self._handle_windows_event_log(result)

            elif result.intent == Intent.TTS_CONTROL:
                return self._handle_tts_control(result)

            else:
                responses = [
                    "Sorry, I didn't understand that command",
                    "I didn't catch that",
                    "Could you repeat that?",
                    "I'm not sure what you mean"
                ]
                import random
                self.tts.say(random.choice(responses))
                time.sleep(1)  # Prevent microphone from capturing TTS output
                return False

        except Exception as e:
            print(f"[ERROR] Command execution failed: {e}")
            self.tts.say("Sorry, there was an error executing that command")
            time.sleep(1)  # Prevent microphone from capturing TTS output
            return False

    def _handle_mode_switch(self, result: CommandResult) -> bool:
        """Handle mode switching commands."""
        if result.action == "start_dictation":
            self.mode = 'dictation'
            self.tts.say("Dictation started")
            time.sleep(1)  # Prevent microphone from capturing TTS output
            return True
        elif result.action == "stop_dictation":
            self.mode = 'command'
            self.tts.say("Dictation stopped")
            time.sleep(1)  # Prevent microphone from capturing TTS output
            return True
        return False

    def _handle_dictation(self, result: CommandResult) -> bool:
        """Handle dictation mode."""
        text = result.parameters.get('text', '')
        print(f"[DICTATION] Typing text: '{text}' (len: {len(text)}, repr: {repr(text)})")
        print(f"[DICTATION] Hex: {[hex(ord(c)) for c in text]}")
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
                time.sleep(1)  # Prevent microphone from capturing TTS output
                return True
            else:
                self.tts.say(f"Sorry, I couldn't open {best_match}")
                time.sleep(1)  # Prevent microphone from capturing TTS output
                return False
        else:
            self.tts.say(f"I don't know how to open {app_name}")
            time.sleep(1)  # Prevent microphone from capturing TTS output
            return False

    def _handle_close_window(self, result: CommandResult) -> bool:
        """Handle window closing commands."""
        self.actions.close_window()
        responses = ["Window closed", "Done", "Closed"]
        import random
        self.tts.say(random.choice(responses))
        time.sleep(1)  # Prevent microphone from capturing TTS output
        return True

    def _handle_volume_control(self, result: CommandResult) -> bool:
        """Handle volume control commands."""
        text = result.parameters.get('text', '').lower()

        if 'up' in text or 'higher' in text or 'increase' in text:
            self.actions.volume_up(steps=2)
            responses = ["Volume increased", "Louder", "Turning it up"]
            import random
            self.tts.say(random.choice(responses))
            time.sleep(1)  # Prevent microphone from capturing TTS output
            return True
        elif 'down' in text or 'lower' in text or 'decrease' in text:
            self.actions.volume_down(steps=2)
            responses = ["Volume decreased", "Quieter", "Turning it down"]
            import random
            self.tts.say(random.choice(responses))
            time.sleep(1)  # Prevent microphone from capturing TTS output
            return True
        elif 'mute' in text:
            self.actions.volume_down(steps=10)  # Effectively mute
            responses = ["Volume muted", "Muted", "Shh"]
            import random
            self.tts.say(random.choice(responses))
            time.sleep(1)  # Prevent microphone from capturing TTS output
            return True

        return False

    def _handle_screenshot(self, result: CommandResult) -> bool:
        """Handle screenshot commands."""
        filename = self.actions.take_screenshot()
        if filename:
            responses = ["Screenshot taken", "Captured", "Got it", "Screen captured"]
            import random
            self.tts.say(random.choice(responses))
            time.sleep(1)  # Prevent microphone from capturing TTS output
            return True
        else:
            self.tts.say("Sorry, couldn't take screenshot")
            time.sleep(1)  # Prevent microphone from capturing TTS output
            return False

    def _handle_text_operation(self, result: CommandResult) -> bool:
        """Handle text operation commands."""
        text = result.parameters.get('text', '').lower()

        if 'copy' in text and 'paste' not in text:
            self.actions.copy()
            responses = ["Copied", "Copied to clipboard", "Done"]
            import random
            self.tts.say(random.choice(responses))
            time.sleep(1)  # Prevent microphone from capturing TTS output
            return True
        elif 'paste' in text:
            self.actions.paste()
            responses = ["Pasted", "Pasted from clipboard", "Done"]
            import random
            self.tts.say(random.choice(responses))
            time.sleep(1)  # Prevent microphone from capturing TTS output
            return True
        elif 'save' in text:
            self.actions.save()
            responses = ["Saved", "File saved", "Done"]
            import random
            self.tts.say(random.choice(responses))
            time.sleep(1)  # Prevent microphone from capturing TTS output
            return True
        elif 'select all' in text:
            self.actions.select_all()
            responses = ["Selected all", "Everything selected", "Done"]
            import random
            self.tts.say(random.choice(responses))
            time.sleep(1)  # Prevent microphone from capturing TTS output
            return True

        return False

    def _handle_search(self, result: CommandResult) -> bool:
        """Handle search commands."""
        query = result.parameters.get('query', '')
        if query:
            summary = self.actions.perform_search(query)
            if summary:
                # Provide the summary
                self.tts.say(f"Here's what I found about {query}: {summary}")
                time.sleep(1)  # Prevent microphone from capturing TTS output
                return True
            else:
                # Fallback to opening browser if no summary available
                url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                success = self.actions.open_url(url)
                if success:
                    responses = [f"Searching for {query}", f"Let me search for {query}", f"Looking up {query}"]
                    import random
                    self.tts.say(random.choice(responses))
                    time.sleep(1)  # Prevent microphone from capturing TTS output
                    return True
                else:
                    self.tts.say(f"Sorry, couldn't search for {query}")
                    time.sleep(1)  # Prevent microphone from capturing TTS output
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
                time.sleep(1)  # Prevent microphone from capturing TTS output
                return True
            else:
                self.tts.say(f"Sorry, couldn't open {url}")
                time.sleep(1)  # Prevent microphone from capturing TTS output
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
                time.sleep(1)  # Prevent microphone from capturing TTS output
                return True
            else:
                self.tts.say("Sorry, I couldn't fetch the news right now")
                time.sleep(1)  # Prevent microphone from capturing TTS output
                return False
        except Exception as e:
            print(f"[ERROR] News reporting failed: {e}")
            self.tts.say("Sorry, there was an error getting the news")
            time.sleep(1)  # Prevent microphone from capturing TTS output
            return False

    def _handle_wikipedia(self, result: CommandResult) -> bool:
        """Handle Wikipedia search commands."""
        topic = result.parameters.get('topic', '')
        if topic:
            summary = self.actions.get_wikipedia_summary(topic)
            if summary:
                # Speak the summary
                self.tts.say(f"According to Wikipedia: {summary}")
                time.sleep(1)  # Prevent microphone from capturing TTS output
                return True
            else:
                self.tts.say(f"Sorry, I couldn't find information about {topic} on Wikipedia")
                time.sleep(1)  # Prevent microphone from capturing TTS output
                return False
        return False

    def _handle_todo_generation(self, result: CommandResult) -> bool:
        """Handle todo list generation commands."""
        list_name = result.parameters.get('list_name', 'default')
        tasks = result.parameters.get('tasks', [])

        success = self.actions.create_todo_list(list_name, tasks)
        if success:
            if tasks:
                self.tts.say(f"Created todo list '{list_name}' with {len(tasks)} tasks")
                time.sleep(1)  # Prevent microphone from capturing TTS output
            else:
                self.tts.say(f"Created empty todo list '{list_name}'")
                time.sleep(1)  # Prevent microphone from capturing TTS output
            return True
        else:
            self.tts.say("Sorry, couldn't create the todo list")
            time.sleep(1)  # Prevent microphone from capturing TTS output
            return False

    def _handle_todo_management(self, result: CommandResult) -> bool:
        """Handle todo list management commands."""
        action = result.parameters.get('action', '')

        if action == 'list':
            lists = self.actions.get_todo_lists()
            if lists:
                list_names = list(lists.keys())
                self.tts.say(f"You have {len(list_names)} todo lists: {', '.join(list_names)}")
                time.sleep(1)  # Prevent microphone from capturing TTS output
                return True
            else:
                self.tts.say("You don't have any todo lists yet")
                time.sleep(1)  # Prevent microphone from capturing TTS output
                return True

        elif action == 'add':
            task = result.parameters.get('task', '')
            if task:
                success = self.actions.add_todo_task('default', task)  # Default list
                if success:
                    self.tts.say(f"Added task: {task}")
                    time.sleep(1)  # Prevent microphone from capturing TTS output
                    return True
                else:
                    self.tts.say("Sorry, couldn't add the task")
                    time.sleep(1)  # Prevent microphone from capturing TTS output
                    return False
            else:
                self.tts.say("No task specified")
                time.sleep(1)  # Prevent microphone from capturing TTS output
                return False

        elif action == 'remove':
            task = result.parameters.get('task', '')
            if task:
                success = self.actions.remove_todo_task('default', task)
                if success:
                    self.tts.say(f"Removed task: {task}")
                    time.sleep(1)  # Prevent microphone from capturing TTS output
                    return True
                else:
                    self.tts.say("Sorry, couldn't remove the task")
                    time.sleep(1)  # Prevent microphone from capturing TTS output
                    return False
            else:
                self.tts.say("No task specified")
                time.sleep(1)  # Prevent microphone from capturing TTS output
                return False

        elif action == 'complete':
            task = result.parameters.get('task', '')
            if task:
                success = self.actions.complete_todo_task('default', task)
                if success:
                    self.tts.say(f"Marked as completed: {task}")
                    time.sleep(1)  # Prevent microphone from capturing TTS output
                    return True
                else:
                    self.tts.say("Sorry, couldn't mark the task as completed")
                    time.sleep(1)  # Prevent microphone from capturing TTS output
                    return False
            else:
                self.tts.say("No task specified")
                time.sleep(1)  # Prevent microphone from capturing TTS output
                return False

        return False

    def _handle_jokes(self, result: CommandResult) -> bool:
        """Handle joke telling commands."""
        joke_type = result.parameters.get('joke_type', 'random')

        try:
            if joke_type == 'random':
                joke = self.actions.get_random_joke()
            elif joke_type == 'programming':
                joke = self.actions.get_programming_joke()
            else:
                joke = self.actions.get_joke_by_category(joke_type)

            if joke:
                # Speak the joke
                self.tts.say(joke)
                time.sleep(1)  # Prevent microphone from capturing TTS output
                return True
            else:
                self.tts.say("Sorry, I couldn't fetch a joke right now")
                time.sleep(1)  # Prevent microphone from capturing TTS output
                return False
        except Exception as e:
            print(f"[ERROR] Joke handling failed: {e}")
            self.tts.say("Sorry, there was an error getting a joke")
            time.sleep(1)  # Prevent microphone from capturing TTS output
            return False

    def _handle_youtube(self, result: CommandResult) -> bool:
        """Handle YouTube commands."""
        action = result.parameters.get('action', '')
        query = result.parameters.get('query', '')

        try:
            if action == 'search':
                results = self.actions.search_youtube(query)
                if results:
                    self.tts.say(f"Found {len(results)} YouTube videos for '{query}': {', '.join([r['title'] for r in results[:3]])}")
                    time.sleep(1)
                    return True
                else:
                    self.tts.say(f"Sorry, couldn't find YouTube videos for '{query}'")
                    time.sleep(1)
                    return False

            elif action == 'info':
                info = self.actions.get_youtube_video_info(query)
                if info:
                    self.tts.say(f"Video info: {info['title']}, Duration: {info['duration']}, Views: {info['views']}")
                    time.sleep(1)
                    return True
                else:
                    self.tts.say(f"Sorry, couldn't get video info for '{query}'")
                    time.sleep(1)
                    return False

            elif action == 'download_audio':
                success = self.actions.download_youtube_audio(query)
                if success:
                    self.tts.say("Audio download completed successfully")
                    time.sleep(1)
                    return True
                else:
                    self.tts.say("Sorry, audio download failed")
                    time.sleep(1)
                    return False

            elif action == 'download_video':
                success = self.actions.download_youtube_video(query)
                if success:
                    self.tts.say("Video download completed successfully")
                    time.sleep(1)
                    return True
                else:
                    self.tts.say("Sorry, video download failed")
                    time.sleep(1)
                    return False

            else:
                self.tts.say("Unknown YouTube action")
                time.sleep(1)
                return False

        except Exception as e:
            print(f"[ERROR] YouTube handling failed: {e}")
            self.tts.say("Sorry, there was an error with the YouTube operation")
            time.sleep(1)
            return False

    def _handle_location_services(self, result: CommandResult) -> bool:
        """Handle location services commands."""
        action = result.parameters.get('action', '')

        try:
            if action == 'current_location':
                location_data = self.actions.get_current_location()
                if location_data:
                    city = location_data.get('city', 'Unknown')
                    state = location_data.get('state', 'Unknown')
                    country = location_data.get('country', 'Unknown')
                    address = location_data.get('address', 'Unknown location')

                    response = f"Your current location appears to be in {city}, {state}, {country}. Full address: {address}"
                    self.tts.say(response)
                    time.sleep(1)
                    return True
                else:
                    self.tts.say("Sorry, I couldn't determine your current location")
                    time.sleep(1)
                    return False

            elif action == 'geocode':
                address = result.parameters.get('address', '')
                if address:
                    location_data = self.actions.geocode_address(address)
                    if location_data:
                        lat = location_data.get('latitude', 'Unknown')
                        lng = location_data.get('longitude', 'Unknown')
                        response = f"The coordinates for {address} are latitude {lat}, longitude {lng}"
                        self.tts.say(response)
                        time.sleep(1)
                        return True
                    else:
                        self.tts.say(f"Sorry, I couldn't find coordinates for {address}")
                        time.sleep(1)
                        return False
                else:
                    self.tts.say("Please specify an address to geocode")
                    time.sleep(1)
                    return False

            elif action == 'reverse_geocode':
                lat = result.parameters.get('latitude')
                lng = result.parameters.get('longitude')

                if lat is not None and lng is not None:
                    location_data = self.actions.reverse_geocode(lat, lng)
                    if location_data:
                        address = location_data.get('address', 'Unknown address')
                        response = f"The address for coordinates {lat}, {lng} is {address}"
                        self.tts.say(response)
                        time.sleep(1)
                        return True
                    else:
                        self.tts.say(f"Sorry, I couldn't find an address for coordinates {lat}, {lng}")
                        time.sleep(1)
                        return False
                else:
                    self.tts.say("Please specify valid coordinates for reverse geocoding")
                    time.sleep(1)
                    return False

            elif action == 'calculate_distance':
                location1 = result.parameters.get('location1', '')
                location2 = result.parameters.get('location2', '')

                if location1 and location2:
                    # First geocode both locations
                    loc1_data = self.actions.geocode_address(location1)
                    loc2_data = self.actions.geocode_address(location2)

                    if loc1_data and loc2_data:
                        lat1 = loc1_data['latitude']
                        lng1 = loc1_data['longitude']
                        lat2 = loc2_data['latitude']
                        lng2 = loc2_data['longitude']

                        distance = self.actions.calculate_distance(lat1, lng1, lat2, lng2)
                        if distance is not None:
                            response = f"The distance between {location1} and {location2} is approximately {distance} kilometers"
                            self.tts.say(response)
                            time.sleep(1)
                            return True
                        else:
                            self.tts.say("Sorry, I couldn't calculate the distance")
                            time.sleep(1)
                            return False
                    else:
                        self.tts.say("Sorry, I couldn't find coordinates for one or both locations")
                        time.sleep(1)
                        return False
                else:
                    self.tts.say("Please specify two locations to calculate distance between")
                    time.sleep(1)
                    return False

            else:
                self.tts.say("Unknown location service action")
                time.sleep(1)
                return False

        except Exception as e:
            print(f"[ERROR] Location services handling failed: {e}")
            self.tts.say("Sorry, there was an error with the location service")
            time.sleep(1)
            return False

    def _handle_system_monitoring(self, result: CommandResult) -> bool:
        """Handle system monitoring commands."""
        action = result.parameters.get('action', '')

        try:
            if action == 'cpu_usage':
                info = self.actions.get_cpu_usage()
                self.tts.say(info)
                time.sleep(1)
                return True

            elif action == 'memory_usage':
                info = self.actions.get_memory_usage()
                self.tts.say(info)
                time.sleep(1)
                return True

            elif action == 'memory_free':
                # For free memory, we can use the same method as memory_usage
                info = self.actions.get_memory_usage()
                self.tts.say(info)
                time.sleep(1)
                return True

            elif action == 'disk_space':
                info = self.actions.get_disk_space()
                self.tts.say(info)
                time.sleep(1)
                return True

            elif action == 'battery_status':
                info = self.actions.get_battery_status()
                self.tts.say(info)
                time.sleep(1)
                return True

            elif action == 'running_processes':
                info = self.actions.get_running_processes()
                self.tts.say(info)
                time.sleep(1)
                return True

            elif action == 'network_info':
                info = self.actions.get_network_info()
                self.tts.say(info)
                time.sleep(1)
                return True

            else:
                self.tts.say("Unknown system monitoring action")
                time.sleep(1)
                return False

        except Exception as e:
            print(f"[ERROR] System monitoring failed: {e}")
            self.tts.say("Sorry, there was an error getting system information")
            time.sleep(1)
            return False

    def _handle_price_comparison(self, result: CommandResult) -> bool:
        """Handle price comparison commands."""
        product = result.parameters.get('product', '')
        if product:
            try:
                summary = self.actions.compare_prices(product)
                if summary:
                    self.tts.say(f"Price comparison for {product}: {summary}")
                    time.sleep(1)
                    return True
                else:
                    self.tts.say(f"Sorry, I couldn't find price information for {product}")
                    time.sleep(1)
                    return False
            except Exception as e:
                print(f"[ERROR] Price comparison failed: {e}")
                self.tts.say("Sorry, there was an error comparing prices")
                time.sleep(1)
                return False
        return False

    def _handle_recipe_lookup(self, result: CommandResult) -> bool:
        """Handle recipe lookup commands."""
        recipe = result.parameters.get('recipe', '')
        if recipe:
            try:
                summary = self.actions.lookup_recipe(recipe)
                if summary:
                    self.tts.say(f"Recipe for {recipe}: {summary}")
                    time.sleep(1)
                    return True
                else:
                    self.tts.say(f"Sorry, I couldn't find a recipe for {recipe}")
                    time.sleep(1)
                    return False
            except Exception as e:
                print(f"[ERROR] Recipe lookup failed: {e}")
                self.tts.say("Sorry, there was an error looking up the recipe")
                time.sleep(1)
                return False
        return False

    def _handle_dictionary(self, result: CommandResult) -> bool:
        """Handle dictionary lookup commands."""
        word = result.parameters.get('word', '')
        if word:
            try:
                definition = self.actions.define_word(word)
                if definition:
                    self.tts.say(f"Definition of {word}: {definition}")
                    time.sleep(1)
                    return True
                else:
                    self.tts.say(f"Sorry, I couldn't find a definition for {word}")
                    time.sleep(1)
                    return False
            except Exception as e:
                print(f"[ERROR] Dictionary lookup failed: {e}")
                self.tts.say("Sorry, there was an error looking up the word")
                time.sleep(1)
                return False
        return False

    def _handle_stock_price(self, result: CommandResult) -> bool:
        """Handle stock price lookup commands."""
        stock = result.parameters.get('stock', '')
        if stock:
            try:
                price_info = self.actions.get_stock_price(stock)
                if price_info:
                    self.tts.say(f"Stock price for {stock}: {price_info}")
                    time.sleep(1)
                    return True
                else:
                    self.tts.say(f"Sorry, I couldn't find stock price information for {stock}")
                    time.sleep(1)
                    return False
            except Exception as e:
                print(f"[ERROR] Stock price lookup failed: {e}")
                self.tts.say("Sorry, there was an error getting stock price information")
                time.sleep(1)
                return False
        return False

    def _handle_weather(self, result: CommandResult) -> bool:
        """Handle weather lookup commands."""
        location = result.parameters.get('location', '')
        if location:
            try:
                weather_info = self.actions.get_weather(location)
                if weather_info:
                    self.tts.say(f"Weather in {location}: {weather_info}")
                    time.sleep(1)
                    return True
                else:
                    self.tts.say(f"Sorry, I couldn't find weather information for {location}")
                    time.sleep(1)
                    return False
            except Exception as e:
                print(f"[ERROR] Weather lookup failed: {e}")
                self.tts.say("Sorry, there was an error getting weather information")
                time.sleep(1)
                return False
        return False

    def _handle_windows_system_info(self, result: CommandResult) -> bool:
        """Handle Windows system info commands."""
        try:
            info = self.actions.get_windows_system_info()
            if info:
                self.tts.say(f"Windows system information: {info}")
                time.sleep(1)
                return True
            else:
                self.tts.say("Sorry, I couldn't retrieve Windows system information")
                time.sleep(1)
                return False
        except Exception as e:
            print(f"[ERROR] Windows system info failed: {e}")
            self.tts.say("Sorry, there was an error getting Windows system information")
            time.sleep(1)
            return False

    def _handle_file_operation(self, result: CommandResult) -> bool:
        """Handle file operation commands."""
        action = result.parameters.get('action', '')

        try:
            if action == 'create':
                file_path = result.parameters.get('file_path', '')
                if file_path:
                    success = self.actions.create_file(file_path)
                    if success:
                        self.tts.say(f"File created: {file_path}")
                        time.sleep(1)
                        return True
                    else:
                        self.tts.say(f"Sorry, couldn't create file: {file_path}")
                        time.sleep(1)
                        return False
                else:
                    self.tts.say("Please specify a file path to create")
                    time.sleep(1)
                    return False

            elif action == 'delete':
                file_path = result.parameters.get('file_path', '')
                if file_path:
                    success = self.actions.delete_file(file_path)
                    if success:
                        self.tts.say(f"File deleted: {file_path}")
                        time.sleep(1)
                        return True
                    else:
                        self.tts.say(f"Sorry, couldn't delete file: {file_path}")
                        time.sleep(1)
                        return False
                else:
                    self.tts.say("Please specify a file path to delete")
                    time.sleep(1)
                    return False

            elif action == 'move':
                source_path = result.parameters.get('source_path', '')
                dest_path = result.parameters.get('dest_path', '')
                if source_path and dest_path:
                    success = self.actions.move_file(source_path, dest_path)
                    if success:
                        self.tts.say(f"File moved from {source_path} to {dest_path}")
                        time.sleep(1)
                        return True
                    else:
                        self.tts.say(f"Sorry, couldn't move file from {source_path} to {dest_path}")
                        time.sleep(1)
                        return False
                else:
                    self.tts.say("Please specify source and destination paths")
                    time.sleep(1)
                    return False

            else:
                self.tts.say("Unknown file operation")
                time.sleep(1)
                return False

        except Exception as e:
            print(f"[ERROR] File operation failed: {e}")
            self.tts.say("Sorry, there was an error with the file operation")
            time.sleep(1)
            return False

    def _handle_windows_services(self, result: CommandResult) -> bool:
        """Handle Windows services commands."""
        action = result.parameters.get('action', '')
        service_name = result.parameters.get('service_name', '')

        try:
            if action == 'start' and service_name:
                success = self.actions.start_windows_service(service_name)
                if success:
                    self.tts.say(f"Windows service started: {service_name}")
                    time.sleep(1)
                    return True
                else:
                    self.tts.say(f"Sorry, couldn't start service: {service_name}")
                    time.sleep(1)
                    return False

            elif action == 'stop' and service_name:
                success = self.actions.stop_windows_service(service_name)
                if success:
                    self.tts.say(f"Windows service stopped: {service_name}")
                    time.sleep(1)
                    return True
                else:
                    self.tts.say(f"Sorry, couldn't stop service: {service_name}")
                    time.sleep(1)
                    return False

            elif action == 'restart' and service_name:
                # Restart = stop then start
                stop_success = self.actions.stop_windows_service(service_name)
                if stop_success:
                    time.sleep(2)  # Wait a bit
                    start_success = self.actions.start_windows_service(service_name)
                    if start_success:
                        self.tts.say(f"Windows service restarted: {service_name}")
                        time.sleep(1)
                        return True
                    else:
                        self.tts.say(f"Service stopped but couldn't restart: {service_name}")
                        time.sleep(1)
                        return False
                else:
                    self.tts.say(f"Sorry, couldn't restart service: {service_name}")
                    time.sleep(1)
                    return False

            else:
                self.tts.say("Please specify a service name and action")
                time.sleep(1)
                return False

        except Exception as e:
            print(f"[ERROR] Windows services failed: {e}")
            self.tts.say("Sorry, there was an error with the Windows service operation")
            time.sleep(1)
            return False

    def _handle_windows_registry(self, result: CommandResult) -> bool:
        """Handle Windows registry commands."""
        key_path = result.parameters.get('key_path', '')
        value_name = result.parameters.get('value_name', '')

        try:
            if key_path:
                # Construct full registry path
                full_key_path = f"HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\{key_path}"
                if value_name:
                    full_key_path = f"{full_key_path}\\{value_name}"

                value = self.actions.read_registry_value(full_key_path, value_name or "(Default)")
                if value and not value.startswith("Error"):
                    self.tts.say(f"Registry value: {value}")
                    time.sleep(1)
                    return True
                else:
                    self.tts.say(f"Could not read registry value: {value}")
                    time.sleep(1)
                    return False
            else:
                self.tts.say("Please specify a registry key path")
                time.sleep(1)
                return False

        except Exception as e:
            print(f"[ERROR] Windows registry failed: {e}")
            self.tts.say("Sorry, there was an error reading the registry")
            time.sleep(1)
            return False

    def _handle_windows_event_log(self, result: CommandResult) -> bool:
        """Handle Windows event log commands."""
        log_type = result.parameters.get('log_type', 'System')

        try:
            logs = self.actions.get_windows_event_logs(log_type)
            if logs:
                self.tts.say(f"Windows event logs: {logs}")
                time.sleep(1)
                return True
            else:
                self.tts.say(f"Sorry, couldn't retrieve {log_type} event logs")
                time.sleep(1)
                return False

        except Exception as e:
            print(f"[ERROR] Windows event log failed: {e}")
            self.tts.say("Sorry, there was an error reading event logs")
            time.sleep(1)
            return False

    def _handle_tts_control(self, result: CommandResult) -> bool:
        """Handle TTS control commands."""
        action = result.parameters.get('action', '')

        try:
            if action == 'set_voice':
                gender = result.parameters.get('gender', '')
                if gender:
                    success = self.actions.set_voice(gender=gender)
                    if success:
                        self.tts.say(f"Voice set to {gender}")
                        time.sleep(1)
                        return True
                    else:
                        self.tts.say(f"Sorry, couldn't set voice to {gender}")
                        time.sleep(1)
                        return False

            elif action == 'set_rate':
                rate = result.parameters.get('rate', '')
                if rate == 'faster':
                    # Increase current rate by 50
                    success = self.actions.set_speech_rate(230)  # Default was 180
                    if success:
                        self.tts.say("Speech rate increased")
                        time.sleep(1)
                        return True
                elif rate == 'slower':
                    # Decrease current rate by 50
                    success = self.actions.set_speech_rate(130)  # Default was 180
                    if success:
                        self.tts.say("Speech rate decreased")
                        time.sleep(1)
                        return True
                elif isinstance(rate, int):
                    success = self.actions.set_speech_rate(rate)
                    if success:
                        self.tts.say(f"Speech rate set to {rate} words per minute")
                        time.sleep(1)
                        return True
                else:
                    self.tts.say("Invalid speech rate")
                    time.sleep(1)
                    return False

            elif action == 'set_volume':
                volume = result.parameters.get('volume', '')
                if volume == 'increase':
                    success = self.actions.set_volume(1.0)  # Max volume
                    if success:
                        self.tts.say("Volume increased")
                        time.sleep(1)
                        return True
                elif volume == 'decrease':
                    success = self.actions.set_volume(0.3)  # Low volume
                    if success:
                        self.tts.say("Volume decreased")
                        time.sleep(1)
                        return True
                elif isinstance(volume, (int, float)):
                    success = self.actions.set_volume(float(volume))
                    if success:
                        self.tts.say(f"Volume set to {volume}")
                        time.sleep(1)
                        return True
                else:
                    self.tts.say("Invalid volume level")
                    time.sleep(1)
                    return False

            elif action == 'save_to_file':
                filename = result.parameters.get('filename', '')
                if filename:
                    # For demo, save a test message
                    test_text = "This is a test of the text to speech audio file saving feature."
                    success = self.actions.save_text_to_audio_file(test_text, filename)
                    if success:
                        self.tts.say(f"Audio saved to file: {filename}")
                        time.sleep(1)
                        return True
                    else:
                        self.tts.say(f"Sorry, couldn't save audio to {filename}")
                        time.sleep(1)
                        return False
                else:
                    self.tts.say("Please specify a filename")
                    time.sleep(1)
                    return False

            elif action == 'test_voices':
                success = self.actions.test_voices()
                if success:
                    self.tts.say("Voice testing completed")
                    time.sleep(1)
                    return True
                else:
                    self.tts.say("Sorry, couldn't test voices")
                    time.sleep(1)
                    return False

            elif action == 'list_voices':
                voices = self.actions.get_available_voices()
                if voices:
                    voice_names = [v['name'] for v in voices[:5]]  # First 5 voices
                    self.tts.say(f"Available voices: {', '.join(voice_names)}")
                    time.sleep(1)
                    return True
                else:
                    self.tts.say("No voices available")
                    time.sleep(1)
                    return False

            elif action == 'preview_voice':
                voice_id = result.parameters.get('voice_id', '')
                if voice_id:
                    success = self.actions.preview_voice(voice_id)
                    if success:
                        self.tts.say("Voice preview completed")
                        time.sleep(1)
                        return True
                    else:
                        self.tts.say("Sorry, couldn't preview voice")
                        time.sleep(1)
                        return False
                else:
                    self.tts.say("Please specify a voice ID")
                    time.sleep(1)
                    return False

            elif action == 'speak_with_voice':
                voice_id = result.parameters.get('voice_id', '')
                if voice_id:
                    test_text = "This is a test of the selected voice."
                    success = self.actions.speak_text_pyttsx3(test_text, voice_id=voice_id)
                    if success:
                        self.tts.say("Voice test completed")
                        time.sleep(1)
                        return True
                    else:
                        self.tts.say("Sorry, couldn't test voice")
                        time.sleep(1)
                        return False
                else:
                    self.tts.say("Please specify a voice ID")
                    time.sleep(1)
                    return False

            else:
                self.tts.say("Unknown TTS action")
                time.sleep(1)
                return False

        except Exception as e:
            print(f"[ERROR] TTS control failed: {e}")
            self.tts.say("Sorry, there was an error with the TTS control")
            time.sleep(1)
            return False

    def get_stats(self) -> dict:
        """Get parser statistics."""
        return self.stats.copy()

    def get_learning_data(self) -> dict:
        """Get user learning data."""
        return self.learning_data.copy()


# Backward compatibility
CommandParser = EnhancedCommandParser