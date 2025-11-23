import pyautogui
import time

def test_unicode_typing():
    """Test if pyautogui can type Unicode characters."""

    # Test texts
    test_cases = [
        "Hello World",  # English
        "Main Hoon",    # Romanized Hindi
        "मैं हूँ",       # Devanagari Hindi
        "कैसे हैं आप",   # More Hindi
    ]

    print("Testing pyautogui.write with different texts...")
    print("Make sure Notepad is open and focused.")

    time.sleep(3)  # Give time to focus notepad

    for i, text in enumerate(test_cases):
        print(f"\nTest {i+1}: Typing '{text}' (repr: {repr(text)})")

        try:
            pyautogui.write(text, interval=0.01)
            pyautogui.press('enter')  # New line
            print(f"Success: Typed '{text}'")
        except Exception as e:
            print(f"Error typing '{text}': {e}")

        time.sleep(1)  # Pause between tests

if __name__ == "__main__":
    test_unicode_typing()