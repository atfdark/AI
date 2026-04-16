# ============================================================
# PATCH FILE — parser_enhanced.py  (file operations extension)
# ============================================================
# Four edits total. Search for the anchor text in each section.
# ============================================================


# ---- EDIT 1: Add new Intent values --------------------------------
# Anchor: find the Intent enum, after FILE_OPERATION add these:
#
#   FILE_OPERATION = "file_operation"       ← already exists
#
# ADD after it:
    FILE_READ     = "file_read"
    FILE_SEARCH   = "file_search"
    FILE_SUMMARISE = "file_summarise"
    FOLDER_LIST   = "folder_list"
    FOLDER_OPEN   = "folder_open"


# ---- EDIT 2: Add patterns in _initialize_patterns() ---------------
# Anchor: find the existing FILE_OPERATION block:
#
#   patterns.update({
#       Intent.FILE_OPERATION: [
#           ...
#       ],
#   })
#
# ADD this NEW block immediately AFTER it:

        patterns.update({
            Intent.FILE_READ: [
                (r'\bread\s+(?:me\s+)?(?:my\s+|the\s+)?(.+?)(?:\s+file)?\s*(?:please|now|aloud)?$', 0.9),
                (r'\bopen\s+(?:and\s+read\s+)?(?:my\s+|the\s+)?(.+?)\s+(?:file|document|pdf|doc)\s*$', 0.85),
                (r'\bread\s+(?:out\s+)?(?:the\s+)?(?:contents?\s+of\s+)?(?:my\s+)?(.+?)$', 0.8),
            ],
            Intent.FILE_SUMMARISE: [
                (r'\b(?:summarise|summarize|give\s+me\s+a\s+summary\s+of)\s+(?:my\s+|the\s+)?(.+?)(?:\s+file)?\s*$', 0.9),
                (r'\bwhat(?:\'s|\s+is)\s+in\s+(?:my\s+|the\s+)?(.+?)(?:\s+file)?\s*(?:\?)?$', 0.85),
                (r'\bgive\s+me\s+(?:a\s+)?(?:brief\s+)?overview\s+of\s+(?:my\s+|the\s+)?(.+?)$', 0.8),
            ],
            Intent.FILE_SEARCH: [
                (r'\bsearch\s+(?:my\s+)?files?\s+for\s+(.+?)(?:\s+in\s+(.+?))?\s*$', 0.9),
                (r'\bfind\s+(?:(?:a|the)\s+)?file\s+(?:called\s+|named\s+)?(.+?)(?:\s+in\s+(.+?))?\s*$', 0.9),
                (r'\blook\s+for\s+(.+?)\s+(?:file|document)\s*(?:in\s+(.+?))?\s*$', 0.85),
                (r'\bdo\s+I\s+have\s+(?:a\s+)?(?:file\s+(?:called|named)\s+)?(.+?)\s*(?:\?)?\s*$', 0.8),
            ],
            Intent.FOLDER_LIST: [
                (r'\bwhat\s+(?:files?\s+)?(?:are|is)\s+in\s+(?:my\s+)?(.+?)\s+folder\s*(?:\?)?$', 0.9),
                (r'\bshow\s+(?:me\s+)?(?:the\s+)?(?:contents?\s+of\s+)?(?:my\s+)?(.+?)\s+folder\s*$', 0.85),
                (r'\blist\s+(?:(?:my|the)\s+)?(.+?)\s+folder\s*$', 0.85),
                (r'\bwhat\'s\s+in\s+(?:my\s+)?(.+?)\s+folder\s*(?:\?)?$', 0.9),
            ],
            Intent.FOLDER_OPEN: [
                (r'\bopen\s+(?:my\s+)?(.+?)\s+folder\s*$', 0.9),
                (r'\bgo\s+to\s+(?:my\s+)?(.+?)\s+folder\s*$', 0.85),
                (r'\bshow\s+(?:my\s+)?(.+?)\s+folder\s+in\s+explorer\s*$', 0.9),
                (r'\bbrowse\s+(?:my\s+)?(.+?)\s+folder\s*$', 0.8),
            ],
        })


# ---- EDIT 3: Wire up parameter extraction in _extract_parameters() ----
# Anchor: find the existing elif block for Intent.FILE_OPERATION:
#
#   elif intent == Intent.FILE_OPERATION:
#       if 'create' in text.lower():
#           ...
#
# ADD these new elif blocks immediately AFTER the FILE_OPERATION block:

        elif intent == Intent.FILE_READ:
            if match and match.lastindex >= 1:
                parameters['file_name'] = match.group(1).strip()
            parameters['action'] = 'read'

        elif intent == Intent.FILE_SUMMARISE:
            if match and match.lastindex >= 1:
                parameters['file_name'] = match.group(1).strip()
            parameters['action'] = 'summarise'

        elif intent == Intent.FILE_SEARCH:
            if match and match.lastindex >= 1:
                parameters['query'] = match.group(1).strip()
            if match and match.lastindex >= 2 and match.group(2):
                parameters['folder'] = match.group(2).strip()
            parameters['action'] = 'search'

        elif intent == Intent.FOLDER_LIST:
            if match and match.lastindex >= 1:
                parameters['folder'] = match.group(1).strip()
            parameters['action'] = 'list'

        elif intent == Intent.FOLDER_OPEN:
            if match and match.lastindex >= 1:
                parameters['folder'] = match.group(1).strip()
            parameters['action'] = 'open'


# ---- EDIT 4: Add handler dispatch in execute_command() ------------
# Anchor: find the existing elif block:
#
#   elif result.intent == Intent.FILE_OPERATION:
#       return self._handle_file_operation(result)
#
# ADD these new elif branches immediately AFTER it:

            elif result.intent == Intent.FILE_READ:
                return self._handle_file_read(result)

            elif result.intent == Intent.FILE_SUMMARISE:
                return self._handle_file_summarise(result)

            elif result.intent == Intent.FILE_SEARCH:
                return self._handle_file_search(result)

            elif result.intent == Intent.FOLDER_LIST:
                return self._handle_folder_list(result)

            elif result.intent == Intent.FOLDER_OPEN:
                return self._handle_folder_open(result)


# ---- NEW HANDLER METHODS -------------------------------------------
# Add these five methods to the EnhancedCommandParser class,
# alongside the other _handle_* methods (e.g. after _handle_file_operation).

    def _get_file_ops(self):
        """Lazy-load FileOps to avoid import at module level."""
        if not hasattr(self, '_file_ops_instance'):
            try:
                from .file_ops import FileOps
                self._file_ops_instance = FileOps(config_path=self.config_path)
            except ImportError:
                self._file_ops_instance = None
        return self._file_ops_instance

    def _handle_file_read(self, result: CommandResult) -> bool:
        file_ops = self._get_file_ops()
        if not file_ops:
            self.tts.say("File reading module is not available.")
            time.sleep(1)
            return False

        file_name = result.parameters.get('file_name', '')
        if not file_name:
            self.tts.say("Which file would you like me to read?")
            time.sleep(1)
            return False

        self.tts.say(f"Let me find and read {file_name}.")
        success, spoken = file_ops.read_file(file_name)
        self.tts.say(spoken, sync=True)
        time.sleep(1)
        return success

    def _handle_file_summarise(self, result: CommandResult) -> bool:
        file_ops = self._get_file_ops()
        if not file_ops:
            self.tts.say("File reading module is not available.")
            time.sleep(1)
            return False

        file_name = result.parameters.get('file_name', '')
        if not file_name:
            self.tts.say("Which file would you like me to summarise?")
            time.sleep(1)
            return False

        self.tts.say(f"Summarising {file_name}.")
        success, spoken = file_ops.summarise_file(file_name)
        self.tts.say(spoken, sync=True)
        time.sleep(1)
        return success

    def _handle_file_search(self, result: CommandResult) -> bool:
        file_ops = self._get_file_ops()
        if not file_ops:
            self.tts.say("File search is not available.")
            time.sleep(1)
            return False

        query  = result.parameters.get('query', '')
        folder = result.parameters.get('folder', '')
        if not query:
            self.tts.say("What should I search for?")
            time.sleep(1)
            return False

        self.tts.say(f"Searching your files for {query}.")
        success, spoken = file_ops.search_files(query, folder)
        self.tts.say(spoken, sync=True)
        time.sleep(1)
        return success

    def _handle_folder_list(self, result: CommandResult) -> bool:
        file_ops = self._get_file_ops()
        if not file_ops:
            self.tts.say("Folder listing is not available.")
            time.sleep(1)
            return False

        folder = result.parameters.get('folder', '')
        if not folder:
            self.tts.say("Which folder would you like me to list?")
            time.sleep(1)
            return False

        success, spoken = file_ops.list_folder(folder)
        self.tts.say(spoken, sync=True)
        time.sleep(1)
        return success

    def _handle_folder_open(self, result: CommandResult) -> bool:
        file_ops = self._get_file_ops()
        if not file_ops:
            self.tts.say("Folder opening is not available.")
            time.sleep(1)
            return False

        folder = result.parameters.get('folder', '')
        if not folder:
            self.tts.say("Which folder would you like me to open?")
            time.sleep(1)
            return False

        success, spoken = file_ops.open_folder(folder)
        self.tts.say(spoken)
        time.sleep(1)
        return success
