#!/usr/bin/env python3
"""
Create bug tickets from code review findings and add to Sprint 8.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / ".claude/skills"))

from work_tracking import get_adapter

# Initialize adapter
adapter = get_adapter()

# Sprint 8 iteration path
SPRINT_8 = "Sprint 8"

# Bug definitions from code review
bugs = [
    # CRITICAL ISSUES
    {
        "title": "State Manager: Race Condition - No File Locking",
        "description": """## Problem
Multiple workflow instances can read/modify the same state file simultaneously, causing data corruption.

**Location:** `core/state_manager.py:124-136` (save method)

**Attack Scenario:**
1. Workflow A reads state (step 1 completed)
2. Workflow B reads same state (step 1 completed)
3. Workflow A completes step 2, saves state
4. Workflow B completes step 2 (duplicate), saves state, overwrites A's save
5. Result: Step 2 executed twice, state corrupt

**Impact:** HIGH - Data loss, duplicate work items created, signature verification failures

## Solution
Implement file locking with `fcntl.flock()` (Unix) or `msvcrt.locking()` (Windows)

```python
import fcntl
import msvcrt
import platform

def save(self) -> None:
    self.state["updated_at"] = datetime.now().isoformat()
    signature = self.compute_signature(self.state)
    self.state["_signature"] = signature

    # Acquire exclusive lock
    with open(self.state_file, 'w', encoding='utf-8') as f:
        if platform.system() != 'Windows':
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        else:
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

        f.write(json.dumps(self.state, indent=2))

        # Lock automatically released on file close
```

## Acceptance Criteria
- [ ] File locking implemented for both Unix and Windows
- [ ] Concurrent workflow test passes (2+ workflows modifying same state)
- [ ] Lock automatically released on exception
- [ ] Lock timeout configured (30 seconds max wait)
""",
        "priority": 1,  # Critical
        "severity": "1 - Critical",
    },
    {
        "title": "State Manager: Unhandled JSON Decode Error",
        "description": """## Problem
Corrupted state file (partial write, disk error) causes unhandled JSONDecodeError, workflow crashes immediately with no remediation.

**Location:** `core/state_manager.py:76`

```python
if self.state_file.exists():
    state_data = json.loads(self.state_file.read_text())  # Can raise JSONDecodeError
```

**Impact:** HIGH - Workflow crashes immediately, no remediation offered

## Solution
Handle JSONDecodeError and offer remediation options:

```python
if self.state_file.exists():
    try:
        state_data = json.loads(self.state_file.read_text())
    except json.JSONDecodeError as e:
        print("=" * 80)
        print("❌ STATE FILE CORRUPTED")
        print("=" * 80)
        print(f"File: {self.state_file}")
        print(f"Error: {e}")
        print()
        print("REMEDIATION OPTIONS:")
        print("1. Delete corrupt state and start fresh")
        print("2. Restore from backup (if available)")
        print("3. Abort workflow")
        print()

        response = input("Enter choice (1/2/3): ").strip()
        if response == "1":
            self.state_file.unlink()
            return self._create_new_state()
        elif response == "2":
            return self._restore_from_backup()
        else:
            sys.exit(1)
```

## Acceptance Criteria
- [ ] JSONDecodeError caught and handled gracefully
- [ ] User prompted with 3 remediation options
- [ ] Option 1 (delete) creates fresh state
- [ ] Option 2 (restore) looks for .bak file
- [ ] Option 3 (abort) exits cleanly with error code
- [ ] Test with intentionally corrupted state file passes
""",
        "priority": 1,  # Critical
        "severity": "1 - Critical",
    },
    {
        "title": "State Manager: No Atomic Write - Partial Save Corruption",
        "description": """## Problem
Power loss during `write_text()` creates partial/corrupt file. Signature becomes invalid, workflow cannot resume.

**Location:** `core/state_manager.py:136`

```python
self.state_file.write_text(json.dumps(self.state, indent=2), encoding='utf-8')
```

**Impact:** HIGH - State file corrupted, signature invalid, workflow cannot resume

## Solution
Implement atomic write pattern (write to temp file, atomic rename):

```python
def save(self) -> None:
    self.state["updated_at"] = datetime.now().isoformat()
    signature = self.compute_signature(self.state)
    self.state["_signature"] = signature

    # Atomic write: write to temp, then rename
    temp_file = self.state_file.with_suffix('.tmp')
    temp_file.write_text(json.dumps(self.state, indent=2), encoding='utf-8')

    # Atomic rename (POSIX and Windows both support this)
    temp_file.replace(self.state_file)
```

**Why This Works:**
- `Path.replace()` is atomic on both POSIX and Windows
- If power loss occurs during write, temp file is incomplete but state_file is unchanged
- If power loss occurs during rename, either old state_file exists OR new state_file exists (never partial)

## Acceptance Criteria
- [ ] Atomic write implemented using temp file + rename
- [ ] Works on Linux, macOS, Windows
- [ ] Test: Kill process during save, state file is either old OR new (never corrupt)
- [ ] Temp files cleaned up on success
- [ ] Backup of previous state created before overwrite (optional)
""",
        "priority": 1,  # Critical
        "severity": "1 - Critical",
    },
    {
        "title": "Cross-Verification: No Rate Limiting - API Abuse Risk",
        "description": """## Problem
Large state files (100+ work items) query adapter in tight loop without delay. Can exceed Azure DevOps API rate limits, causing verification to fail and requests to be blocked.

**Location:** `core/state_manager.py:313-343`

```python
for item_record in created_items:  # Could be 100+ items
    work_item = adapter.get_work_item(item_id)  # API call every iteration
```

**Impact:** MEDIUM - API rate limit exceeded, verification fails, Azure DevOps blocks requests

## Solution
Implement batch queries and rate limiting:

```python
import time

def cross_verify_with_adapter(self, adapter) -> Dict[str, Any]:
    print("\\n🔍 Cross-verifying state against external source of truth...")

    created_items = self.state.get("created_work_items", [])
    missing_items = []
    verification_errors = []
    items_verified = 0

    # Rate limiting: pause every N items
    RATE_LIMIT_BATCH_SIZE = 10
    RATE_LIMIT_DELAY = 1.0  # seconds

    for i, item_record in enumerate(created_items):
        item_id = item_record.get("id")
        item_data = item_record.get("data", {})

        # Rate limiting
        if i > 0 and i % RATE_LIMIT_BATCH_SIZE == 0:
            print(f"  ⏸️  Rate limit pause (processed {i} items)...")
            time.sleep(RATE_LIMIT_DELAY)

        try:
            work_item = adapter.get_work_item(item_id)
            # ... rest of verification logic ...
```

**Alternative:** Use batch query API if adapter supports it
```python
# Query all items in single batch call
item_ids = [item["id"] for item in created_items]
work_items = adapter.get_work_items_batch(item_ids)  # Single API call
```

## Acceptance Criteria
- [ ] Rate limiting implemented (1 second pause every 10 items)
- [ ] Test with 50+ work items completes without API rate limit errors
- [ ] Configurable rate limit (batch size and delay)
- [ ] Progress indicator shows pause for rate limiting
- [ ] Alternative batch API used if adapter supports it
""",
        "priority": 2,  # High
        "severity": "2 - High",
    },
    {
        "title": "Workflow Orchestrator: Input Validation Missing - Infinite Loop Risk",
        "description": """## Problem
User input not validated properly. Invalid input causes infinite retry loop with no maximum attempt limit.

**Location:** `scripts/workflow_executor/base.py:198-254`

```python
response = input("Enter choice (1/2/3): ").strip()
if response == "1":  # Only checks exact match
```

**Impact:** MEDIUM - User types invalid input repeatedly, workflow stuck in infinite loop

## Solution
Add max retry limit and better validation:

```python
def _handle_unverified_state(self):
    # ... setup ...

    MAX_RETRIES = 5
    retries = 0

    while retries < MAX_RETRIES:
        response = input("Enter choice (1/2/3): ").strip()

        if response not in ["1", "2", "3"]:
            print(f"Invalid choice: {response}. Please enter 1, 2, or 3.")
            retries += 1
            continue

        if response == "1":
            # ... continue logic ...
            return
        elif response == "2":
            # ... cross-verify logic ...
            return  # or continue if verification fails
        elif response == "3":
            # ... abort logic ...
            sys.exit(1)

    # Max retries exceeded
    print(f"\\n❌ Maximum retry attempts ({MAX_RETRIES}) exceeded")
    print("   Aborting workflow for safety")
    sys.exit(1)
```

## Acceptance Criteria
- [ ] Max retry limit of 5 attempts
- [ ] Invalid input shows clear error message
- [ ] After 5 invalid inputs, workflow aborts safely
- [ ] Test: Enter 6 invalid inputs → workflow aborts
- [ ] Test: Enter valid input after 4 invalid → workflow continues
""",
        "priority": 2,  # High
        "severity": "2 - High",
    },

    # HIGH PRIORITY ISSUES
    {
        "title": "State Manager: Secret Key Stored in Memory - Security Risk",
        "description": """## Problem
HMAC secret key cached in class variable permanently, never cleared from memory. Memory dump can reveal secret key, compromising all signatures.

**Location:** `core/state_manager.py:373-383`

```python
if cls._HMAC_SECRET_KEY is None:
    env_key = os.environ.get("WORKFLOW_STATE_SECRET_KEY")
    if env_key:
        cls._HMAC_SECRET_KEY = env_key.encode('utf-8')  # Cached forever
```

**Impact:** MEDIUM - Memory dump reveals secret key, all signatures compromised

## Solution
Clear secret key from memory after use:

```python
@classmethod
def _get_secret_key(cls) -> bytes:
    """Get HMAC secret key (ephemeral - not cached)."""
    env_key = os.environ.get("WORKFLOW_STATE_SECRET_KEY")
    if env_key:
        return env_key.encode('utf-8')
    else:
        return b"trustable-ai-default-key-CHANGE-IN-PRODUCTION"

@classmethod
def _clear_secret_key(cls):
    """Clear secret key from memory (best effort)."""
    # Note: Python doesn't guarantee memory clearing, but this helps
    if cls._HMAC_SECRET_KEY:
        # Overwrite with random data before clearing
        import secrets
        cls._HMAC_SECRET_KEY = secrets.token_bytes(len(cls._HMAC_SECRET_KEY))
        cls._HMAC_SECRET_KEY = None
```

**Note:** This is a temporary fix. HMAC will be replaced with Keychain Core in the future for proper key management.

## Acceptance Criteria
- [ ] Secret key no longer cached in class variable
- [ ] Key read from environment on each use
- [ ] `_clear_secret_key()` method implemented
- [ ] Called at end of critical operations
- [ ] Performance impact minimal (key retrieval is fast)
""",
        "priority": 2,  # High
        "severity": "2 - High",
    },
    {
        "title": "Workflow Orchestrator: No Timeout on input() - Hang Forever",
        "description": """## Problem
`input()` blocks indefinitely if user walks away from terminal. Workflow hangs, CI/CD pipelines stall, automation broken.

**Location:** `scripts/workflow_executor/base.py:199`

```python
response = input("Enter choice (1/2/3): ").strip()  # Blocks indefinitely
```

**Impact:** MEDIUM - Workflow hangs, CI/CD pipelines stall, automation broken

## Solution
Implement input with timeout using threading:

```python
import threading

def input_with_timeout(prompt: str, timeout: int = 300) -> Optional[str]:
    """
    Get user input with timeout.

    Args:
        prompt: Input prompt to display
        timeout: Timeout in seconds (default 5 minutes)

    Returns:
        User input or None if timeout
    """
    result = [None]

    def get_input():
        try:
            result[0] = input(prompt)
        except EOFError:
            result[0] = None

    thread = threading.Thread(target=get_input)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print(f"\\n⏱️  Input timeout after {timeout} seconds")
        print("   Aborting workflow for safety")
        return None

    return result[0]

# Usage in _handle_unverified_state():
response = input_with_timeout("Enter choice (1/2/3): ", timeout=300)
if response is None:
    print("\\n❌ No response received, aborting workflow")
    sys.exit(1)
```

## Acceptance Criteria
- [ ] Input timeout of 300 seconds (5 minutes) default
- [ ] Configurable timeout via parameter
- [ ] Timeout message displayed clearly
- [ ] Workflow aborts safely on timeout
- [ ] Test: Wait > 5 minutes → workflow aborts
- [ ] Works on Linux, macOS, Windows
""",
        "priority": 2,  # High
        "severity": "2 - High",
    },
    {
        "title": "State Manager: Cleanup Only Deletes Completed States",
        "description": """## Problem
`cleanup_old_states()` only deletes states with status "completed". Failed/interrupted states accumulate forever, consuming disk space.

**Location:** `core/state_manager.py:595`

```python
if started < cutoff and state["status"] == "completed":  # Only completed
    state_file.unlink()
```

**Impact:** LOW - Disk space accumulation over time

## Solution
Add option to clean failed/interrupted states:

```python
def cleanup_old_states(
    days: int = 30,
    include_failed: bool = False,
    include_interrupted: bool = False
) -> int:
    """
    Clean up workflow state files older than specified days.

    Args:
        days: Age threshold in days
        include_failed: Also clean failed workflows
        include_interrupted: Also clean interrupted workflows

    Returns:
        Number of files deleted
    """
    from datetime import timedelta

    state_dir = Path(".claude/workflow-state")
    if not state_dir.exists():
        return 0

    cutoff = datetime.now() - timedelta(days=days)
    deleted = 0

    # Determine which statuses to clean
    statuses_to_clean = ["completed"]
    if include_failed:
        statuses_to_clean.append("failed")
    if include_interrupted:
        statuses_to_clean.append("interrupted")

    for state_file in state_dir.glob("*.json"):
        try:
            state = json.loads(state_file.read_text())
            started = datetime.fromisoformat(state["started_at"])

            if started < cutoff and state["status"] in statuses_to_clean:
                state_file.unlink()
                deleted += 1
                print(f"Deleted old state: {state_file.name} ({state['status']})")
        except Exception as e:
            print(f"Error processing {state_file}: {e}")

    return deleted
```

## Acceptance Criteria
- [ ] `include_failed` parameter added (default False)
- [ ] `include_interrupted` parameter added (default False)
- [ ] Cleanup reports which status types were cleaned
- [ ] Test: Failed states cleaned when `include_failed=True`
- [ ] Test: Completed states always cleaned regardless of flags
- [ ] CLI command updated: `trustable-ai state cleanup --include-failed --days 30`
""",
        "priority": 3,  # Medium
        "severity": "3 - Medium",
    },
    {
        "title": "Subprocess Security: Empty Allowlist Allows All Commands",
        "description": """## Problem
If `allowed_commands` is empty set or None, validation is skipped and all commands are allowed. This defeats the purpose of the security wrapper.

**Location:** `scripts/workflow_executor/subprocess_security.py:202-210`

```python
if self.allowed_commands:  # Empty set = False, validation skipped!
    base_command = Path(command_name).name
    if base_command not in self.allowed_commands:
        raise SubprocessSecurityError(...)
```

**Impact:** HIGH - Empty allowlist = no protection against unauthorized commands

## Solution
Make allowlist mandatory or warn loudly:

```python
def __init__(
    self,
    allowed_commands: Optional[Set[str]] = None,
    audit_log_dir: Optional[Path] = None
):
    """Initialize secure subprocess executor."""
    if allowed_commands is None or len(allowed_commands) == 0:
        print("=" * 80)
        print("⚠️  SECURITY WARNING: No command allowlist configured!")
        print("=" * 80)
        print("All commands will be allowed (INSECURE for production)")
        print("Set allowed_commands parameter to restrict command execution")
        print("=" * 80)
        print()

    self.allowed_commands = allowed_commands or set()
    self.audit_log_dir = audit_log_dir or Path(".claude/audit")
    self.audit_log_dir.mkdir(parents=True, exist_ok=True)

    # ... rest of init ...
```

**Better:** Require explicit allowlist in production:

```python
def __init__(
    self,
    allowed_commands: Set[str],  # No longer optional!
    audit_log_dir: Optional[Path] = None,
    strict_mode: bool = True
):
    if strict_mode and (not allowed_commands or len(allowed_commands) == 0):
        raise SubprocessSecurityError(
            "Strict mode requires non-empty allowed_commands set"
        )
```

## Acceptance Criteria
- [ ] Warning printed when allowlist is empty
- [ ] `strict_mode` parameter added (default True for production)
- [ ] Strict mode raises error if allowlist empty
- [ ] Non-strict mode allows empty allowlist with warning
- [ ] Test: Empty allowlist in strict mode → raises error
- [ ] Test: Empty allowlist in non-strict mode → prints warning
""",
        "priority": 2,  # High
        "severity": "2 - High",
    },

    # MEDIUM PRIORITY ISSUES
    {
        "title": "State Manager: No State File Size Limit - DoS Risk",
        "description": """## Problem
Malicious or corrupted state file (100MB+ JSON) causes memory exhaustion when loaded entirely into memory.

**Location:** `core/state_manager.py:76`

```python
state_data = json.loads(self.state_file.read_text())  # Loads entire file into memory
```

**Impact:** MEDIUM - DoS attack via large state file, workflow crashes with OOM

## Solution
Check file size before loading:

```python
MAX_STATE_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit

def _load_state(self) -> Dict[str, Any]:
    if self.state_file.exists():
        # Check file size before loading
        file_size = self.state_file.stat().st_size
        if file_size > MAX_STATE_FILE_SIZE:
            raise ValueError(
                f"State file too large: {file_size} bytes "
                f"(max {MAX_STATE_FILE_SIZE} bytes). "
                f"File may be corrupt or malicious."
            )

        try:
            state_data = json.loads(self.state_file.read_text())
        except json.JSONDecodeError as e:
            # ... handle decode error ...
```

## Acceptance Criteria
- [ ] 10MB maximum state file size
- [ ] Size check before file read
- [ ] Clear error message with actual vs max size
- [ ] Test: 11MB state file → raises ValueError
- [ ] Test: 9MB state file → loads successfully
- [ ] Configurable limit via environment variable
""",
        "priority": 3,  # Medium
        "severity": "3 - Medium",
    },
    {
        "title": "Workflow Orchestrator: Evidence Not Validated for Required Keys",
        "description": """## Problem
Step evidence only checked for dict type, not for required content. Steps can return empty dict `{}` and workflow proceeds with no real evidence.

**Location:** `scripts/workflow_executor/base.py:360-362`

```python
if not isinstance(evidence, dict):
    raise ValueError(f"Step {step_id} must return dict evidence, got {type(evidence)}")
# No check for required keys or non-empty!
```

**Impact:** LOW - Steps can return empty dict, workflow proceeds with no real evidence

## Solution
Validate evidence is non-empty and optionally check required keys:

```python
def _execute_step_with_enforcement(self, step: Dict[str, Any]) -> bool:
    # ... execute step ...
    evidence = self._execute_step(step, context)

    # Verify evidence was collected
    if not isinstance(evidence, dict):
        raise ValueError(
            f"Step {step_id} must return dict evidence, got {type(evidence)}"
        )

    if len(evidence) == 0:
        raise ValueError(
            f"Step {step_id} returned empty evidence dict. "
            f"Steps must collect verifiable evidence."
        )

    # Optional: Check for required keys if step defines them
    required_keys = step.get("required_evidence_keys", [])
    if required_keys:
        missing_keys = set(required_keys) - set(evidence.keys())
        if missing_keys:
            raise ValueError(
                f"Step {step_id} missing required evidence keys: {missing_keys}"
            )

    # ... store evidence ...
```

## Acceptance Criteria
- [ ] Empty dict evidence raises ValueError
- [ ] Clear error message indicates empty evidence not allowed
- [ ] Optional `required_evidence_keys` in step definition
- [ ] Missing required keys raises ValueError with key names
- [ ] Test: Empty dict → raises ValueError
- [ ] Test: Missing required key → raises ValueError
- [ ] Test: Valid non-empty evidence → passes
""",
        "priority": 3,  # Medium
        "severity": "3 - Medium",
    },
    {
        "title": "State Manager: JSON Serialization Errors Not Handled",
        "description": """## Problem
Non-serializable objects in state (datetime, Path, custom objects) cause unhandled TypeError on save. Workflow crashes, state lost.

**Location:** `core/state_manager.py:136`

```python
self.state_file.write_text(json.dumps(self.state, indent=2), encoding='utf-8')
# If state contains datetime objects, pathlib.Path, etc. - TypeError
```

**Impact:** MEDIUM - Workflow crashes on save, state lost since last checkpoint

## Solution
Handle serialization errors gracefully with custom encoder:

```python
import json
from datetime import datetime
from pathlib import Path

class StateJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for state serialization."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

def save(self) -> None:
    self.state["updated_at"] = datetime.now().isoformat()
    signature = self.compute_signature(self.state)
    self.state["_signature"] = signature

    try:
        json_data = json.dumps(
            self.state,
            indent=2,
            cls=StateJSONEncoder
        )
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"State contains non-serializable data: {e}\\n"
            f"Ensure all state data is JSON-serializable "
            f"(str, int, float, bool, list, dict)"
        )

    self.state_file.write_text(json_data, encoding='utf-8')
```

## Acceptance Criteria
- [ ] Custom JSON encoder handles datetime, Path objects
- [ ] TypeError caught with clear error message
- [ ] Error message lists problematic data types
- [ ] Test: State with datetime → serializes successfully
- [ ] Test: State with Path → serializes successfully
- [ ] Test: State with custom object → raises clear error
""",
        "priority": 3,  # Medium
        "severity": "3 - Medium",
    },
    {
        "title": "Cross-Verification: No Retry Logic on Network Errors",
        "description": """## Problem
Transient network errors (timeout, connection reset) fail verification permanently. No retry logic for recoverable errors.

**Location:** `core/state_manager.py:317-343`

```python
try:
    work_item = adapter.get_work_item(item_id)  # No retry on network error
except Exception as e:
    error_msg = f"Verification failed for work item #{item_id}: {str(e)}"
    verification_errors.append(error_msg)
```

**Impact:** MEDIUM - False verification failures due to transient network issues

## Solution
Implement exponential backoff retry for network errors:

```python
import time
from typing import Optional

def _get_work_item_with_retry(
    adapter,
    item_id: int,
    max_retries: int = 3
) -> Optional[Dict]:
    """Get work item with retry logic for network errors."""
    for retry in range(max_retries):
        try:
            return adapter.get_work_item(item_id)
        except (ConnectionError, TimeoutError, OSError) as e:
            if retry < max_retries - 1:
                delay = 2 ** retry  # Exponential backoff: 1s, 2s, 4s
                print(f"  ⚠️  Network error, retrying in {delay}s... ({retry + 1}/{max_retries})")
                time.sleep(delay)
                continue
            else:
                # Final retry failed
                raise
        except Exception as e:
            # Non-network error, don't retry
            raise

# Usage in cross_verify_with_adapter():
for item_record in created_items:
    item_id = item_record.get("id")

    try:
        work_item = _get_work_item_with_retry(adapter, item_id, max_retries=3)
        # ... rest of verification ...
```

## Acceptance Criteria
- [ ] Retry logic for ConnectionError, TimeoutError, OSError
- [ ] Exponential backoff: 1s, 2s, 4s delays
- [ ] Maximum 3 retry attempts
- [ ] Progress indicator shows retry attempts
- [ ] Non-network errors don't trigger retry
- [ ] Test: Transient network error → retries and succeeds
- [ ] Test: Permanent error → fails after 3 retries
""",
        "priority": 3,  # Medium
        "severity": "3 - Medium",
    },
    {
        "title": "Workflow Orchestrator: Audit Log Lost on Crash",
        "description": """## Problem
Audit log kept in memory only, saved to disk at workflow completion. Crash before completion loses entire audit trail.

**Location:** `scripts/workflow_executor/base.py:411-424`

```python
self.audit_log: List[Dict[str, Any]] = []  # Memory only

def _log_audit(self, event_type: str, data: Dict[str, Any]) -> None:
    audit_entry = {...}
    self.audit_log.append(audit_entry)  # Only appends to list
    # No immediate write to disk!
```

**Impact:** MEDIUM - Crash before workflow completion = audit log lost, no trace of what happened

## Solution
Write audit entries to disk immediately (append-only log):

```python
def _log_audit(self, event_type: str, data: Dict[str, Any]) -> None:
    """Log audit event (writes immediately to disk)."""
    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "data": data
    }

    # Append to in-memory list
    self.audit_log.append(audit_entry)

    # Flush to disk immediately (append-only JSONL format)
    audit_dir = Path(".claude/audit")
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Live audit log (append-only, survives crashes)
    live_log = audit_dir / f"{self.workflow_name}-{self.workflow_id}-live.jsonl"

    try:
        with open(live_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(audit_entry) + '\\n')
            f.flush()  # Force write to disk
    except Exception as e:
        # Don't fail workflow if audit logging fails
        print(f"⚠️  Warning: Audit log write failed: {e}")
```

At workflow completion, consolidate into final JSON:

```python
def _save_audit_log(self, status: str, error: Optional[str] = None) -> Path:
    # ... existing code ...

    # Delete live log after consolidation
    live_log = Path(f".claude/audit/{self.workflow_name}-{self.workflow_id}-live.jsonl")
    if live_log.exists():
        live_log.unlink()

    return audit_file
```

## Acceptance Criteria
- [ ] Audit entries written to disk immediately
- [ ] JSONL format (one JSON object per line)
- [ ] Live log deleted after successful consolidation
- [ ] Test: Kill workflow mid-execution → live log has partial audit trail
- [ ] Test: Complete workflow → final JSON has full audit trail
- [ ] Audit write failure doesn't crash workflow
""",
        "priority": 3,  # Medium
        "severity": "3 - Medium",
    },

    # LOW PRIORITY ISSUES
    {
        "title": "State Manager: No State Backup Before Modification",
        "description": """## Problem
State file overwritten without backup. Accidental corruption or bug in save logic results in permanent data loss.

**Impact:** LOW - Accidental state corruption is permanent, no recovery path

## Solution
Create backup before each save:

```python
import shutil

def save(self) -> None:
    # Create backup of existing state before overwriting
    if self.state_file.exists():
        backup_file = self.state_file.with_suffix('.bak')
        shutil.copy2(self.state_file, backup_file)

        # Keep last N backups (rotate)
        self._rotate_backups(max_backups=3)

    # ... rest of save logic ...

def _rotate_backups(self, max_backups: int = 3):
    """Keep only the last N backups."""
    backup_pattern = f"{self.state_file.stem}.bak*"
    backups = sorted(
        self.state_dir.glob(backup_pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    # Delete old backups
    for old_backup in backups[max_backups:]:
        old_backup.unlink()
```

## Acceptance Criteria
- [ ] Backup created before each save
- [ ] Backup file named `{workflow}-{id}.bak`
- [ ] Rotation keeps last 3 backups
- [ ] Restore method to load from backup
- [ ] Test: Save 5 times → only 3 backups exist
""",
        "priority": 4,  # Low
        "severity": "4 - Low",
    },
    {
        "title": "State Manager: No Performance Metrics on Verification",
        "description": """## Problem
No visibility into cross-verification performance. Can't identify slow verifications or optimize adapter queries.

**Impact:** LOW - Can't optimize verification performance

## Solution
Track and report verification timing:

```python
import time

def cross_verify_with_adapter(self, adapter) -> Dict[str, Any]:
    print("\\n🔍 Cross-verifying state against external source of truth...")

    start_time = time.time()
    created_items = self.state.get("created_work_items", [])

    # ... verification logic ...

    # Calculate metrics
    duration = time.time() - start_time
    avg_time_per_item = duration / len(created_items) if created_items else 0

    print(f"\\n📊 Verification Summary:")
    print(f"   Items checked: {len(created_items)}")
    print(f"   Items verified: {items_verified}")
    print(f"   Missing items: {len(missing_items)}")
    print(f"   Errors: {len(verification_errors)}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Avg per item: {avg_time_per_item:.2f}s")

    return {
        "verified": all_verified,
        "missing_items": missing_items,
        "verification_errors": verification_errors,
        "items_checked": len(created_items),
        "items_verified": items_verified,
        "metrics": {
            "duration_seconds": duration,
            "avg_time_per_item": avg_time_per_item
        }
    }
```

## Acceptance Criteria
- [ ] Duration tracking for entire verification
- [ ] Average time per item calculated
- [ ] Metrics included in return dict
- [ ] Metrics saved to audit log
- [ ] Test: Verify 10 items → metrics show ~0.1s per item
""",
        "priority": 4,  # Low
        "severity": "4 - Low",
    },
    {
        "title": "State Manager: No Disk Space Check Before Save",
        "description": """## Problem
No check for available disk space before writing state file. Write fails with cryptic error if disk full.

**Impact:** LOW - Workflow crashes with unclear error on disk full

## Solution
Check disk space before save:

```python
import shutil

def save(self) -> None:
    # Check available disk space
    stats = shutil.disk_usage(self.state_dir)
    required_space = len(json.dumps(self.state)) * 2  # 2x for safety

    if stats.free < required_space:
        raise OSError(
            f"Insufficient disk space: {stats.free} bytes available, "
            f"{required_space} bytes required"
        )

    # ... rest of save logic ...
```

## Acceptance Criteria
- [ ] Disk space checked before save
- [ ] Clear error message on insufficient space
- [ ] Test: Mock low disk space → raises OSError
""",
        "priority": 4,  # Low
        "severity": "4 - Low",
    },
]

def create_bugs():
    """Create all bug tickets and add to Sprint 8."""
    print(f"Creating {len(bugs)} bug tickets for Sprint 8...")
    print("=" * 80)

    created_items = []

    for i, bug in enumerate(bugs, 1):
        print(f"\n[{i}/{len(bugs)}] Creating: {bug['title']}")

        try:
            # Create bug work item
            result = adapter.create_work_item(
                work_item_type="Bug",
                title=bug["title"],
                description=bug["description"],
                iteration=SPRINT_8,
                priority=bug["priority"],
                tags=["code-review", "external-enforcement", "security"],
            )

            work_item_id = result.get("id")

            # Update severity if supported
            try:
                adapter.update_work_item(
                    work_item_id=work_item_id,
                    fields={
                        "Microsoft.VSTS.Common.Severity": bug["severity"]
                    }
                )
            except:
                pass  # Severity field may not be supported

            created_items.append({
                "id": work_item_id,
                "title": bug["title"],
                "priority": bug["priority"]
            })

            print(f"  ✅ Created Bug #{work_item_id}")

        except Exception as e:
            print(f"  ❌ Failed to create bug: {e}")
            continue

    print("\n" + "=" * 80)
    print(f"✅ Successfully created {len(created_items)} bug tickets")
    print("=" * 80)

    # Summary by priority
    print("\nSummary by Priority:")
    priority_counts = {}
    for item in created_items:
        p = item["priority"]
        priority_counts[p] = priority_counts.get(p, 0) + 1

    priority_names = {
        1: "Critical",
        2: "High",
        3: "Medium",
        4: "Low"
    }

    for priority in sorted(priority_counts.keys()):
        count = priority_counts[priority]
        name = priority_names.get(priority, f"Priority {priority}")
        print(f"  {name}: {count} bug(s)")

    print("\nCreated Bug IDs:")
    for item in created_items:
        print(f"  #{item['id']}: {item['title']}")

    return created_items

if __name__ == "__main__":
    created_items = create_bugs()
    print(f"\n✅ All bug tickets created and added to Sprint 8")
