#!/usr/bin/env python3
"""Test script for YouTube functionality."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'assistant'))

from actions import Actions

def test_youtube_search():
    """Test YouTube search functionality."""
    actions = Actions()
    print("Testing YouTube search...")

    results = actions.search_youtube("python tutorial", 3)
    if results:
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            try:
                print(f"{i}. {result['title']} - {result['author']}")
            except UnicodeEncodeError:
                print(f"{i}. [Unicode title] - {result['author']}")
    else:
        print("Search failed")

def test_youtube_info():
    """Test YouTube video info functionality."""
    actions = Actions()
    print("\nTesting YouTube video info...")

    # Test with a known video URL
    info = actions.get_youtube_video_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ")  # Rick Roll
    if info:
        try:
            print(f"Title: {info['title']}")
            print(f"Author: {info['author']}")
        except UnicodeEncodeError:
            print("Title: [Unicode title]")
            print(f"Author: {info['author']}")
        print(f"Duration: {info['length']} seconds")
        print(f"Views: {info['views']}")
    else:
        print("Failed to get video info")

if __name__ == "__main__":
    test_youtube_search()
    test_youtube_info()
    print("\nTest completed.")