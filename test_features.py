#!/usr/bin/env python3
"""
Comprehensive test script for the newly implemented features:
- Website opening
- News reporting
- Todo list generation and management

Tests include error handling and TTS integration verification.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import shutil

# Add assistant module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'assistant'))

from assistant.actions import Actions
from assistant.tts import TTS
from assistant.parser_enhanced import EnhancedCommandParser, Intent, CommandResult


class MockTTS:
    """Mock TTS class to capture what would be spoken."""
    def __init__(self):
        self.spoken_texts = []

    def say(self, text, sync=False):
        """Capture spoken text instead of actually speaking."""
        self.spoken_texts.append(text)
        print(f"[MOCK TTS] {text}")

    def get_last_spoken(self):
        """Get the last spoken text."""
        return self.spoken_texts[-1] if self.spoken_texts else None

    def get_all_spoken(self):
        """Get all spoken texts."""
        return self.spoken_texts.copy()

    def clear(self):
        """Clear spoken texts."""
        self.spoken_texts.clear()


class TestWebsiteOpening(unittest.TestCase):
    """Test cases for website opening functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tts = MockTTS()
        self.actions = Actions()
        self.parser = EnhancedCommandParser(actions=self.actions, tts=self.mock_tts)

    def test_open_valid_url_with_https(self):
        """Test opening a valid URL with https protocol."""
        with patch('webbrowser.open') as mock_open:
            mock_open.return_value = True

            result = self.parser.parse_intent("go to https://www.google.com")
            success = self.parser.execute_command(result)

            self.assertTrue(success)
            mock_open.assert_called_once_with("https://www.google.com")
            self.assertIn("https://www.google.com", self.mock_tts.get_last_spoken())

    def test_open_url_without_protocol(self):
        """Test opening a URL without protocol (should add https)."""
        with patch('webbrowser.open') as mock_open:
            mock_open.return_value = True

            result = self.parser.parse_intent("go to www.example.com")
            success = self.parser.execute_command(result)

            self.assertTrue(success)
            mock_open.assert_called_once_with("https://www.example.com")
            self.assertIn("www.example.com", self.mock_tts.get_last_spoken())

    def test_open_invalid_url(self):
        """Test opening an invalid URL."""
        # Note: webbrowser.open doesn't validate URLs, so this will succeed
        with patch('webbrowser.open') as mock_open:
            mock_open.return_value = True

            result = self.parser.parse_intent("go to invalid-url")
            success = self.parser.execute_command(result)

            self.assertTrue(success)
            self.assertIn("invalid-url", self.mock_tts.get_last_spoken())

    def test_open_empty_url(self):
        """Test opening an empty URL."""
        result = self.parser.parse_intent("go to")
        success = self.parser.execute_command(result)

        # Empty URL doesn't match web browsing pattern, goes to unknown
        self.assertFalse(success)
        self.assertTrue(any(phrase in self.mock_tts.get_last_spoken() for phrase in ["didn't understand", "not sure", "repeat that"]))

    def test_browser_open_failure(self):
        """Test when browser.open() raises an exception."""
        with patch('webbrowser.open', side_effect=Exception("Browser error")):
            result = self.parser.parse_intent("go to https://www.test.com")
            success = self.parser.execute_command(result)

            self.assertFalse(success)
            self.assertIn("Sorry, couldn't open", self.mock_tts.get_last_spoken())


class TestNewsReporting(unittest.TestCase):
    """Test cases for news reporting functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tts = MockTTS()
        # Create temporary config with API key
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'config.json')
        with open(self.config_path, 'w') as f:
            json.dump({
                "newsapi": {"api_key": "test_api_key"}
            }, f)

        self.actions = Actions(config_path=self.config_path)
        self.parser = EnhancedCommandParser(actions=self.actions, tts=self.mock_tts, config_path=self.config_path)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_successful_news_fetch(self):
        """Test successful news fetching."""
        mock_newsapi = Mock()
        mock_newsapi.get_top_headlines.return_value = {
            'status': 'ok',
            'articles': [
                {'title': 'Test Headline 1', 'source': {'name': 'Test Source 1'}},
                {'title': 'Test Headline 2', 'source': {'name': 'Test Source 2'}},
                {'title': 'Test Headline 3', 'source': {'name': 'Test Source 3'}}
            ]
        }

        with patch('assistant.actions.NewsApiClient', return_value=mock_newsapi):
            result = self.parser.parse_intent("what's the news")
            success = self.parser.execute_command(result)

            self.assertTrue(success)
            spoken = self.mock_tts.get_last_spoken()
            self.assertIn("Here are the latest news headlines", spoken)
            self.assertIn("Test Headline 1", spoken)
            self.assertIn("Test Headline 2", spoken)
            self.assertIn("Test Headline 3", spoken)

    def test_news_fetch_no_api_key(self):
        """Test news fetching without API key."""
        # Config without API key
        with open(self.config_path, 'w') as f:
            json.dump({}, f)

        result = self.parser.parse_intent("get news")
        success = self.parser.execute_command(result)

        self.assertFalse(success)
        self.assertIn("Sorry, I couldn't fetch the news", self.mock_tts.get_last_spoken())

    def test_news_fetch_api_error(self):
        """Test news fetching with API error."""
        mock_newsapi = Mock()
        mock_newsapi.get_top_headlines.return_value = {'status': 'error'}

        with patch('assistant.actions.NewsApiClient', return_value=mock_newsapi):
            result = self.parser.parse_intent("tell me the news")
            success = self.parser.execute_command(result)

            self.assertFalse(success)
            self.assertIn("Sorry, I couldn't fetch the news", self.mock_tts.get_last_spoken())

    def test_news_fetch_no_articles(self):
        """Test news fetching with no articles returned."""
        mock_newsapi = Mock()
        mock_newsapi.get_top_headlines.return_value = {
            'status': 'ok',
            'articles': []
        }

        with patch('assistant.actions.NewsApiClient', return_value=mock_newsapi):
            result = self.parser.parse_intent("latest news")
            success = self.parser.execute_command(result)

            self.assertTrue(success)
            self.assertIn("No news articles found", self.mock_tts.get_last_spoken())

    def test_news_fetch_exception(self):
        """Test news fetching with exception."""
        with patch('assistant.actions.NewsApiClient', side_effect=Exception("API Error")):
            result = self.parser.parse_intent("news update")
            success = self.parser.execute_command(result)

            self.assertFalse(success)
            self.assertIn("couldn't fetch the news right now", self.mock_tts.get_last_spoken())

    def test_newsapi_not_installed(self):
        """Test when NewsAPI library is not installed."""
        with patch('assistant.actions.NewsApiClient', None):
            result = self.parser.parse_intent("what's the news")
            success = self.parser.execute_command(result)

            self.assertFalse(success)
            self.assertIn("Sorry, I couldn't fetch the news", self.mock_tts.get_last_spoken())


class TestTodoListFeatures(unittest.TestCase):
    """Test cases for todo list generation and management."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tts = MockTTS()
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'config.json')
        with open(self.config_path, 'w') as f:
            json.dump({}, f)

        self.actions = Actions(config_path=self.config_path)
        self.parser = EnhancedCommandParser(actions=self.actions, tts=self.mock_tts, config_path=self.config_path)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_create_todo_list_success(self):
        """Test creating a new todo list successfully."""
        result = self.parser.parse_intent("create todo list for shopping")
        success = self.parser.execute_command(result)

        self.assertTrue(success)
        self.assertIn("Created empty todo list 'shopping'", self.mock_tts.get_last_spoken())

        # Verify list was created
        lists = self.actions.get_todo_lists()
        self.assertIn('shopping', lists)
        self.assertEqual(len(lists['shopping']['tasks']), 0)

    def test_create_todo_list_already_exists(self):
        """Test creating a todo list that already exists."""
        # Create list first
        self.actions.create_todo_list('work', [])

        result = self.parser.parse_intent("create todo list for work")
        success = self.parser.execute_command(result)

        self.assertFalse(success)
        self.assertIn("Sorry, couldn't create the todo list", self.mock_tts.get_last_spoken())

    def test_add_task_to_list(self):
        """Test adding a task to an existing list."""
        self.actions.create_todo_list('test', [])

        result = self.parser.parse_intent("add task buy milk to todo")
        success = self.parser.execute_command(result)

        self.assertTrue(success)
        self.assertIn("Added task: buy milk", self.mock_tts.get_last_spoken())

        # Verify task was added
        tasks = self.actions.get_todo_list('default')
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]['description'], 'buy milk')
        self.assertFalse(tasks[0]['completed'])

    def test_add_task_to_nonexistent_list(self):
        """Test adding a task to a list that doesn't exist (should create it)."""
        result = self.parser.parse_intent("add task test task")
        success = self.parser.execute_command(result)

        self.assertTrue(success)
        self.assertIn("Added task: test task", self.mock_tts.get_last_spoken())

        # Verify list and task were created
        lists = self.actions.get_todo_lists()
        self.assertIn('default', lists)
        tasks = lists['default']['tasks']
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]['description'], 'test task')

    def test_remove_existing_task(self):
        """Test removing an existing task."""
        self.actions.create_todo_list('default', [])
        self.actions.add_todo_task('default', 'test task')

        result = self.parser.parse_intent("remove test task from todo")
        success = self.parser.execute_command(result)

        self.assertTrue(success)
        self.assertIn("Removed task: test task", self.mock_tts.get_last_spoken())

        # Verify task was removed
        tasks = self.actions.get_todo_list('default')
        self.assertEqual(len(tasks), 0)

    def test_remove_nonexistent_task(self):
        """Test removing a task that doesn't exist."""
        self.actions.create_todo_list('default', [])

        result = self.parser.parse_intent("remove nonexistent task from todo")
        success = self.parser.execute_command(result)

        self.assertFalse(success)
        self.assertIn("Sorry, couldn't remove the task", self.mock_tts.get_last_spoken())

    def test_complete_existing_task(self):
        """Test marking an existing task as completed."""
        self.actions.create_todo_list('default', [])
        self.actions.add_todo_task('default', 'test task')

        result = self.parser.parse_intent("mark test task as done")
        success = self.parser.execute_command(result)

        self.assertTrue(success)
        self.assertIn("Marked as completed: test task", self.mock_tts.get_last_spoken())

        # Verify task was completed
        tasks = self.actions.get_todo_list('default')
        self.assertEqual(len(tasks), 1)
        self.assertTrue(tasks[0]['completed'])

    def test_complete_nonexistent_task(self):
        """Test marking a nonexistent task as completed."""
        self.actions.create_todo_list('default', [])

        result = self.parser.parse_intent("complete nonexistent task")
        success = self.parser.execute_command(result)

        self.assertFalse(success)
        self.assertIn("Sorry, couldn't mark the task as completed", self.mock_tts.get_last_spoken())

    def test_show_empty_todo_lists(self):
        """Test showing todo lists when none exist."""
        result = self.parser.parse_intent("show my todo lists")
        success = self.parser.execute_command(result)

        self.assertTrue(success)
        self.assertIn("You don't have any todo lists yet", self.mock_tts.get_last_spoken())

    def test_show_existing_todo_lists(self):
        """Test showing todo lists when some exist."""
        self.actions.create_todo_list('work', [])
        self.actions.create_todo_list('personal', [])

        result = self.parser.parse_intent("list my todo lists")
        success = self.parser.execute_command(result)

        self.assertTrue(success)
        spoken = self.mock_tts.get_last_spoken()
        self.assertIn("2 todo lists", spoken)
        self.assertIn("work", spoken)
        self.assertIn("personal", spoken)

    def test_no_task_specified_add(self):
        """Test adding a task without specifying the task."""
        result = self.parser.parse_intent("add task")
        success = self.parser.execute_command(result)

        self.assertFalse(success)
        self.assertTrue(any(phrase in self.mock_tts.get_last_spoken() for phrase in ["didn't understand", "didn't catch", "not sure", "repeat that", "didn't catch that"]))

    def test_no_task_specified_remove(self):
        """Test removing a task without specifying the task."""
        result = self.parser.parse_intent("remove from todo")
        success = self.parser.execute_command(result)

        self.assertFalse(success)
        self.assertTrue(any(phrase in self.mock_tts.get_last_spoken() for phrase in ["didn't understand", "didn't catch", "not sure", "repeat that"]))

    def test_no_task_specified_complete(self):
        """Test completing a task without specifying the task."""
        result = self.parser.parse_intent("mark as done")
        success = self.parser.execute_command(result)

        self.assertFalse(success)
        self.assertTrue(any(phrase in self.mock_tts.get_last_spoken() for phrase in ["didn't understand", "didn't catch", "not sure", "repeat that"]))


class TestSearchFunctionality(unittest.TestCase):
    """Test cases for search functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tts = MockTTS()
        self.actions = Actions()
        self.parser = EnhancedCommandParser(actions=self.actions, tts=self.mock_tts)

    def test_search_with_summary(self):
        """Test search that returns a summary."""
        mock_response = {
            'AbstractText': 'Python is a programming language.'
        }

        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = mock_response

            result = self.parser.parse_intent("search for python")
            success = self.parser.execute_command(result)

            self.assertTrue(success)
            spoken = self.mock_tts.get_last_spoken()
            self.assertIn("Here's what I found about python", spoken)
            self.assertIn("Python is a programming language", spoken)

    def test_search_fallback_to_browser(self):
        """Test search that falls back to opening browser when no summary."""
        with patch('assistant.actions.Actions.perform_search', return_value=None):
            with patch('webbrowser.open') as mock_open:
                mock_open.return_value = True

                result = self.parser.parse_intent("search for unknown topic")
                success = self.parser.execute_command(result)

                self.assertTrue(success)
                mock_open.assert_called_once()
                spoken = self.mock_tts.get_last_spoken()
                self.assertTrue(any(phrase in spoken for phrase in ["Searching for unknown topic", "Let me search for unknown topic", "Looking up unknown topic"]))

    def test_search_api_failure(self):
        """Test search when API fails."""
        with patch('assistant.actions.Actions.perform_search', return_value=None):
            with patch('webbrowser.open', side_effect=Exception("Browser error")):

                result = self.parser.parse_intent("search for test")
                success = self.parser.execute_command(result)

                self.assertFalse(success)
                self.assertIn("Sorry, couldn't search for test", self.mock_tts.get_last_spoken())


class TestTTSIntegration(unittest.TestCase):
    """Test TTS integration across all features."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tts = MockTTS()
        self.actions = Actions()
        self.parser = EnhancedCommandParser(actions=self.actions, tts=self.mock_tts)

    def test_website_opening_tts_responses(self):
        """Test that website opening provides appropriate TTS responses."""
        with patch('webbrowser.open') as mock_open:
            mock_open.return_value = True

            # Test successful opening
            result = self.parser.parse_intent("go to google.com")
            self.parser.execute_command(result)
            self.assertTrue(any("google.com" in text for text in self.mock_tts.get_all_spoken()))

            self.mock_tts.clear()

            # Test failure
            with patch('webbrowser.open', side_effect=Exception("Error")):
                result = self.parser.parse_intent("go to invalid site")
                self.parser.execute_command(result)
                self.assertTrue(any("couldn't open" in text for text in self.mock_tts.get_all_spoken()))

    def test_news_reporting_tts_responses(self):
        """Test that news reporting provides appropriate TTS responses."""
        # Test successful news fetch
        mock_newsapi = Mock()
        mock_newsapi.get_top_headlines.return_value = {
            'status': 'ok',
            'articles': [{'title': 'Test News', 'source': {'name': 'Test Source'}}]
        }

        with patch('assistant.actions.NewsApiClient', return_value=mock_newsapi):
            result = self.parser.parse_intent("what's the news")
            self.parser.execute_command(result)
            self.assertTrue(any("latest news headlines" in text for text in self.mock_tts.get_all_spoken()))

    def test_todo_operations_tts_responses(self):
        """Test that todo operations provide appropriate TTS responses."""
        import time
        unique_id = str(int(time.time()))

        # Test successful operations with unique names
        result = self.parser.parse_intent(f"create todo list for test_{unique_id}")
        self.parser.execute_command(result)
        self.assertTrue(any("Created" in text and "todo list" in text for text in self.mock_tts.get_all_spoken()))

        self.mock_tts.clear()

        result = self.parser.parse_intent(f"add task test {unique_id} task")
        self.parser.execute_command(result)
        self.assertTrue(any("Added task:" in text for text in self.mock_tts.get_all_spoken()))

        self.mock_tts.clear()

        result = self.parser.parse_intent(f"mark test {unique_id} task as done")
        self.parser.execute_command(result)
        self.assertTrue(any("Marked as completed:" in text for text in self.mock_tts.get_all_spoken()))


def run_tests():
    """Run all tests and provide a summary."""
    print("=" * 60)
    print("RUNNING FEATURE TESTS")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestWebsiteOpening))
    suite.addTests(loader.loadTestsFromTestCase(TestNewsReporting))
    suite.addTests(loader.loadTestsFromTestCase(TestTodoListFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestSearchFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestTTSIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")

    if result.wasSuccessful():
        print("\n[SUCCESS] ALL TESTS PASSED!")
        print("Features are working correctly with proper error handling and TTS integration.")
    else:
        print("\n[FAILED] SOME TESTS FAILED!")
        print("Please review the failures and fix the issues.")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)