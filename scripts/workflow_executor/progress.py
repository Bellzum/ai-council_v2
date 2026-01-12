"""
Progress Indicators for External Enforcement Workflows.

Provides:
- Retry with progress indicators
- Animated spinners for long operations
- Progress bars for batch operations
- User feedback during AI calls

Usage:
    from scripts.workflow_executor.progress import (
        retry_with_progress,
        Spinner,
        ProgressBar
    )

    @retry_with_progress(max_attempts=3, delay=2.0)
    def create_task(adapter, title):
        return adapter.create_work_item(title=title, ...)
"""
import time
import sys
import threading
from typing import Callable, TypeVar, Any, Optional, List, Dict
from functools import wraps
from dataclasses import dataclass
from pathlib import Path

# Import console functions from cli.console
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cli.console import (
    print_success as _console_print_success,
    print_error as _console_print_error,
    print_warning as _console_print_warning,
    print_info as _console_print_info,
)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 2.0
    backoff_factor: float = 2.0
    max_delay: float = 30.0
    retryable_exceptions: tuple = (Exception,)


def retry_with_progress(
    max_attempts: int = 3,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
    operation_name: str = ""
):
    """
    Decorator for retrying operations with progress feedback.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        backoff_factor: Multiply delay by this factor each retry
        retryable_exceptions: Tuple of exception types to retry
        operation_name: Name for progress messages

    Usage:
        @retry_with_progress(max_attempts=3, operation_name="Creating task")
        def create_task(adapter, title):
            return adapter.create_work_item(title=title, ...)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Validate max_attempts parameter
            if max_attempts <= 0:
                raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

            name = operation_name or func.__name__
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        print_success(f"Succeeded on attempt {attempt}")
                    return result

                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        print_warning(f"Attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {e}")
                        print_status(f"⏳ Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, 30.0)  # Max 30s delay
                    else:
                        print_error(f"All {max_attempts} attempts failed")
                        raise

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Retry logic error in {name}")

        return wrapper
    return decorator


class Spinner:
    """
    Animated spinner for long-running operations.

    Usage:
        with Spinner("Analyzing with AI"):
            result = claude_api.call(prompt)
    """

    FRAMES = ['\u280b', '\u2819', '\u2839', '\u2838', '\u283c', '\u2834', '\u2826', '\u2827', '\u2807', '\u280f']

    def __init__(self, message: str = "Processing"):
        """
        Initialize spinner.

        Args:
            message: Message to display while spinning
        """
        self.message = message
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        """Start spinner on context enter."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop spinner on context exit."""
        self.stop(success=exc_type is None)
        return False  # Don't suppress exceptions

    def start(self):
        """Start the spinner animation."""
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self, success: bool = True):
        """
        Stop the spinner.

        Args:
            success: If True, show success icon. If False, show failure icon.
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)

        # Clear spinner line and show result
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()

        # Use console functions for consistent theming
        if success:
            print_success(self.message)
        else:
            print_error(self.message)

    def _spin(self):
        """Animation loop."""
        frame_idx = 0
        while self._running:
            frame = self.FRAMES[frame_idx % len(self.FRAMES)]
            sys.stdout.write(f'\r{frame} {self.message}')
            sys.stdout.flush()
            frame_idx += 1
            time.sleep(0.1)


class ProgressBar:
    """
    Progress bar for batch operations.

    Usage:
        with ProgressBar(total=10, label="Creating tasks") as bar:
            for item in items:
                create_task(item)
                bar.update()
    """

    def __init__(
        self,
        total: int,
        label: str = "Progress",
        width: int = 40,
        show_count: bool = True
    ):
        """
        Initialize progress bar.

        Args:
            total: Total number of items
            label: Label to display
            width: Width of progress bar in characters
            show_count: If True, show count (e.g., "5/10")
        """
        self.total = total
        self.label = label
        self.width = width
        self.show_count = show_count
        self.current = 0

    def __enter__(self):
        """Start progress bar on context enter."""
        self._display()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete progress bar on context exit."""
        if exc_type is None:
            # Success - fill to 100%
            self.current = self.total
            self._display()
            print()  # Newline after completion
        else:
            print()  # Newline before error
        return False

    def update(self, amount: int = 1):
        """
        Update progress.

        Args:
            amount: Amount to increment (default 1)
        """
        self.current = min(self.current + amount, self.total)
        self._display()

    def _display(self):
        """Render the progress bar."""
        if self.total == 0:
            percent = 100.0
        else:
            percent = (self.current / self.total) * 100

        filled = int(self.width * self.current / max(self.total, 1))
        bar = '\u2588' * filled + '\u2591' * (self.width - filled)

        if self.show_count:
            count = f" {self.current}/{self.total}"
        else:
            count = ""

        sys.stdout.write(f'\r{self.label}: [{bar}] {percent:5.1f}%{count}')
        sys.stdout.flush()


def print_step_header(step_number: int, step_name: str, total_steps: int = 0):
    """
    Print a formatted step header.

    Bridges to console functions for consistent theming.
    Maintains backward compatibility with existing API.

    Args:
        step_number: Current step number
        step_name: Name of the step
        total_steps: Total number of steps (0 to hide)
    """
    from cli.console import console

    if total_steps > 0:
        header = f"Step {step_number}/{total_steps}: {step_name}"
    else:
        header = f"Step {step_number}: {step_name}"

    console.print(f"\n{'=' * 60}", style="dim")
    console.print(f"▶ {header}", style="bold_accent2")
    console.print(f"{'=' * 60}", style="dim")


def print_success(message: str):
    """
    Print success message with icon.

    Bridges to cli.console.print_success() for consistent theming.
    Maintains backward compatibility with existing API.
    """
    _console_print_success(f"✔ {message}")


def print_warning(message: str):
    """
    Print warning message with icon.

    Bridges to cli.console.print_warning() for consistent theming.
    Maintains backward compatibility with existing API.
    """
    _console_print_warning(f"⚠ {message}")


def print_error(message: str):
    """
    Print error message with icon.

    Bridges to cli.console.print_error() for consistent theming.
    Maintains backward compatibility with existing API.
    """
    _console_print_error(f"✖ {message}")


def print_info(message: str):
    """
    Print info message with icon.

    Bridges to cli.console.print_info() for consistent theming.
    Maintains backward compatibility with existing API.
    """
    _console_print_info(f"ℹ {message}")


def confirm_action(prompt: str, default: bool = False) -> bool:
    """
    Prompt user for confirmation.

    Args:
        prompt: Question to ask
        default: Default response if user presses Enter

    Returns:
        True if confirmed, False otherwise
    """
    default_hint = "[Y/n]" if default else "[y/N]"
    full_prompt = f"{prompt} {default_hint}: "

    try:
        response = input(full_prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    if not response:
        return default

    return response in ('y', 'yes', 'true', '1')


def print_status(message: str):
    """
    Print status message (neutral, for ongoing operations).

    Bridges to cli.console for consistent theming.
    Maintains backward compatibility with existing API.
    """
    from cli.console import console
    console.print(f"  {message}", style="primary")


def print_work_items_table(items: List[Dict[str, Any]], title: str = "Work Items"):
    """
    Print work items in a structured table format.

    Bridges to console functions for consistent theming.
    Maintains backward compatibility with existing API.

    Args:
        items: List of work item dictionaries
        title: Table title
    """
    from cli.console import console

    if not items:
        console.print(f"\n{title}: None", style="dim")
        return

    console.print(f"\n{title}:", style="bold_primary")
    console.print("-" * 80, style="dim")
    for item in items:
        item_id = item.get('id', 'N/A')
        item_type = item.get('type') or item.get('fields', {}).get('System.WorkItemType', 'Item')
        item_title = item.get('title') or item.get('fields', {}).get('System.Title', 'Untitled')
        item_state = item.get('state') or item.get('fields', {}).get('System.State', 'Unknown')

        console.print(f"  {item_type} #{item_id}: {item_title}", style="primary")
        console.print(f"    State: {item_state}", style="tertiary")
    console.print("-" * 80, style="dim")
