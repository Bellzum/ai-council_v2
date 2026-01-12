#!/usr/bin/env python3
"""
Migrate scripts to new console infrastructure.

Removes emojis, ANSI codes, and updates print statements to use console utilities.
"""

import re
from pathlib import Path

# Emoji patterns to remove
EMOJI_PATTERN = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]')

# ANSI escape codes pattern
ANSI_PATTERN = re.compile(r'\033\[[0-9;]+m')

def remove_emojis_and_ansi(content: str) -> str:
    """Remove all emojis and ANSI escape codes from content."""
    # Remove ANSI codes
    content = ANSI_PATTERN.sub('', content)
    # Remove emojis
    content = EMOJI_PATTERN.sub('', content)
    # Clean up double spaces left by emoji removal
    content = re.sub(r'  +', ' ', content)
    # Remove trailing spaces on print lines
    content = re.sub(r'print\(f?"([^"]*?)  "', r'print(f"\1"', content)
    content = re.sub(r'print\(f?\'([^\']*?)  \'', r"print(f'\1'", content)
    return content

def migrate_daily_standup():
    """Migrate daily_standup.py to new console infrastructure."""
    file_path = Path("scripts/daily_standup.py")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove emojis and ANSI codes
    content = remove_emojis_and_ansi(content)

    # Save the migrated file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Migrated: {file_path}")

def migrate_sprint_retrospective():
    """Migrate sprint_retrospective.py to new console infrastructure."""
    file_path = Path("scripts/sprint_retrospective.py")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove emojis and ANSI codes
    content = remove_emojis_and_ansi(content)

    # Replace custom approval gate with standardized function
    old_pattern = r'print\(f"\n✅ User APPROVED'
    new_pattern = r'print(f"\nUser APPROVED'
    content = re.sub(old_pattern, new_pattern, content)

    # Save the migrated file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Migrated: {file_path}")

if __name__ == "__main__":
    migrate_daily_standup()
    migrate_sprint_retrospective()
    print("\nMigration complete!")
