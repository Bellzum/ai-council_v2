#!/usr/bin/env python3
"""
Migrate backlog_grooming.py and sprint_execution.py to standardized console output.

Implements Task #1314: Console output migration for high-complexity workflow scripts
- Removes decorative emojis (keeps allowed symbols: ✔✓✗✘⚠ℹ▶)
- Removes ANSI escape codes
- Removes first-person language ("I", "we", "our")
- Keeps simple print() statements (professional tone without emojis)
"""

import re
from pathlib import Path
from typing import Tuple


# Emoji patterns - remove decorative emojis, keep simple symbols
DECORATIVE_EMOJI_PATTERN = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]')

# ANSI escape codes pattern
ANSI_PATTERN = re.compile(r'\\033\[[0-9;]+m')

# Allowed symbols (not emojis, just unicode characters for UI)
ALLOWED_SYMBOLS = ['✔', '✓', '✗', '✘', '⚠', 'ℹ', '▶', '└', '─', '│', '├', '┌', '┐', '┘']


def is_allowed_symbol(char: str) -> bool:
    """Check if character is an allowed UI symbol."""
    return char in ALLOWED_SYMBOLS


def remove_emojis(content: str) -> str:
    """
    Remove decorative emojis while preserving allowed UI symbols.

    This function removes all emoji characters except for simple text
    symbols used for UI formatting (checkmarks, warning signs, etc).
    """
    result = []
    for char in content:
        # Keep if not an emoji or if it's an allowed symbol
        if not DECORATIVE_EMOJI_PATTERN.match(char) or is_allowed_symbol(char):
            result.append(char)
        else:
            # Replace emoji with space to avoid text collision
            result.append(' ')

    return ''.join(result)


def remove_ansi_codes(content: str) -> str:
    """Remove ANSI escape codes from content."""
    return ANSI_PATTERN.sub('', content)


def remove_first_person_language(content: str) -> str:
    """
    Remove first-person language from print statements.

    Replacements:
    - "I will" -> "Will"
    - "I'm" -> "Now" or context-appropriate replacement
    - "We will" -> "Will"
    - "We're" -> "Now"
    - "Our" -> "The"
    """
    # Pattern: print statements with first-person pronouns
    patterns = [
        (r'print\([^)]*\bI will\b[^)]*\)', lambda m: m.group(0).replace('I will', 'Will')),
        (r'print\([^)]*\bI\'m\b[^)]*\)', lambda m: m.group(0).replace("I'm", 'Now')),
        (r'print\([^)]*\bWe will\b[^)]*\)', lambda m: m.group(0).replace('We will', 'Will')),
        (r'print\([^)]*\bWe\'re\b[^)]*\)', lambda m: m.group(0).replace("We're", 'Now')),
        (r'print\([^)]*\bOur\b[^)]*\)', lambda m: m.group(0).replace('Our', 'The')),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    return content


def clean_up_spacing(content: str) -> str:
    """Clean up extra spaces left by emoji removal."""
    # Remove multiple spaces (but preserve indentation)
    lines = []
    for line in content.split('\n'):
        # Preserve leading whitespace
        stripped = line.lstrip()
        leading_ws = line[:len(line) - len(stripped)]
        # Collapse multiple spaces in content
        cleaned = re.sub(r' {2,}', ' ', stripped)
        lines.append(leading_ws + cleaned)

    return '\n'.join(lines)


def migrate_file(file_path: Path) -> Tuple[int, int, int]:
    """
    Migrate a workflow script file.

    Returns:
        Tuple of (emoji_count, ansi_count, first_person_count) removed
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        original = f.read()

    # Count before migration
    emoji_count = len([c for c in original if DECORATIVE_EMOJI_PATTERN.match(c) and not is_allowed_symbol(c)])
    ansi_count = len(ANSI_PATTERN.findall(original))

    # Count first-person occurrences in print statements
    first_person_count = len(re.findall(r'print\([^)]*\b(I will|I\'m|We will|We\'re|Our)\b[^)]*\)', original))

    # Apply migrations
    content = original
    content = remove_emojis(content)
    content = remove_ansi_codes(content)
    content = remove_first_person_language(content)
    content = clean_up_spacing(content)

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return emoji_count, ansi_count, first_person_count


def main():
    """Migrate backlog_grooming.py and sprint_execution.py."""
    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / "scripts"

    files_to_migrate = [
        "backlog_grooming.py",
        "sprint_execution.py"
    ]

    print("Migrating high-complexity workflow scripts...")
    print("=" * 70)

    total_emojis = 0
    total_ansi = 0
    total_first_person = 0

    for filename in files_to_migrate:
        file_path = scripts_dir / filename
        if not file_path.exists():
            print(f"SKIP: {filename} (not found)")
            continue

        print(f"\nProcessing: {filename}")
        emoji_count, ansi_count, first_person_count = migrate_file(file_path)
        print(f"  - Removed {emoji_count} decorative emojis")
        print(f"  - Removed {ansi_count} ANSI codes")
        print(f"  - Fixed {first_person_count} first-person statements")

        total_emojis += emoji_count
        total_ansi += ansi_count
        total_first_person += first_person_count

    print("\n" + "=" * 70)
    print("Migration complete!")
    print(f"Total: {total_emojis} emojis, {total_ansi} ANSI codes, {total_first_person} first-person fixes")


if __name__ == "__main__":
    main()
