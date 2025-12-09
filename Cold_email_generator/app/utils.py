import re


def clean_text(text: str) -> str:
    """
    Cleans raw HTML/text scraped from websites.

    Steps:
    - Remove HTML tags
    - Remove URLs
    - Remove special characters (keep letters, numbers, and spaces)
    - Replace multiple spaces with a single space
    - Trim leading/trailing whitespace
    """
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]*?>', '', text)

    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)

    # Remove special characters except letters, numbers, and spaces
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

    # Replace multiple spaces/newlines/tabs with a single space
    text = re.sub(r'\s+', ' ', text)

    # Trim leading and trailing whitespace
    text = text.strip()

    return text
