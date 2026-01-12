#!/usr/bin/env python3
"""Update Medium and Low priority bug tickets with detailed solutions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / ".claude/skills"))
from work_tracking import get_adapter

adapter = get_adapter()

# Medium Priority Bugs

bug_1187_desc = """**Problem Summary**
cleanup_old_states() only deletes completed status. Failed/interrupted states accumulate forever.

**Impact**
- Disk space waste: Failed states never cleaned
- Directory clutter: Hard to find active states
- Data leak: Old secrets in abandoned states

**Location**
File: core/state_manager.py
Line: 595 (cleanup_old_states() method)

**Solution Implementation**

Add include_failed and include_interrupted parameters to cleanup method.

Implementation:
- Add parameters: include_failed, include_interrupted (both default False)
- Query states by status: completed, failed, interrupted
- Delete based on age threshold (e.g., 30 days)
- Provide dry-run mode to preview deletions
- Log deleted state files

**Testing Plan**
- Test cleanup with completed states only (current behavior)
- Test cleanup with failed states included
- Test cleanup with interrupted states included
- Test age threshold filtering
- Test dry-run mode

**Acceptance Criteria**
- [ ] Configurable cleanup of failed/interrupted states
- [ ] Age threshold parameter (default: 30 days)
- [ ] Dry-run mode for preview
- [ ] Logging of deleted files
- [ ] Unit tests: 100% coverage
"""

bug_1189_desc = """**Problem Summary**
No file size check before loading state. Malicious 100MB+ state file causes memory exhaustion.

**Impact**
- DoS risk: Large file exhausts memory
- Crash: Out of memory error
- Resource abuse: Slows system

**Location**
File: core/state_manager.py
Line: 76 (_load_state() method)

**Solution Implementation**

Add file size check before loading (max: 10MB).

Implementation:
- Check state file size before reading
- Max size: 10MB (configurable via env var)
- Raise clear error if file too large
- Suggest remediation (manual inspection, corruption)

**Testing Plan**
- Test loading small files (< 10MB)
- Test loading 10MB file (boundary)
- Test loading 11MB file (rejected)
- Test configurable max size
- Test error message clarity

**Acceptance Criteria**
- [ ] Max file size check (default: 10MB)
- [ ] Configurable via environment variable
- [ ] Clear error message with remediation
- [ ] Unit tests: 100% coverage
"""

bug_1190_desc = """**Problem Summary**
Steps can return empty dict as evidence. No validation for required keys.

**Impact**
- Workflow continues with no evidence
- Verification impossible later
- Audit trail incomplete

**Location**
File: scripts/workflow_executor/base.py
Lines: 360-362 (evidence collection)

**Solution Implementation**

Add evidence validation with required keys check.

Implementation:
- Validate evidence is non-empty dict
- Add optional required_evidence_keys parameter to step definition
- Validate required keys present in returned evidence
- Raise clear error if validation fails

**Testing Plan**
- Test step returns valid evidence (passes)
- Test step returns empty dict (fails)
- Test step missing required keys (fails)
- Test step with optional keys (passes)

**Acceptance Criteria**
- [ ] Non-empty evidence validation
- [ ] Required keys validation (if specified)
- [ ] Clear error messages
- [ ] Unit tests: 100% coverage
"""

bug_1191_desc = """**Problem Summary**
Non-serializable objects (datetime, Path) cause TypeError on save. No error handling.

**Impact**
- Workflow crash during save
- State data lost
- No recovery path

**Location**
File: core/state_manager.py
Line: 136 (save() method using json.dump)

**Solution Implementation**

Add custom JSON encoder for common non-serializable types.

Implementation:
- Create StateJSONEncoder extending json.JSONEncoder
- Handle datetime (convert to ISO format)
- Handle Path (convert to string)
- Handle set (convert to list)
- Handle other common types
- Add try/catch with clear error

**Testing Plan**
- Test saving state with datetime objects
- Test saving state with Path objects
- Test saving state with sets
- Test saving state with custom objects (should fail with clear error)

**Acceptance Criteria**
- [ ] Custom JSON encoder handles datetime, Path, set
- [ ] Clear error for truly non-serializable types
- [ ] Unit tests: 100% coverage
"""

bug_1192_desc = """**Problem Summary**
Transient network errors fail verification permanently. No retry logic.

**Impact**
- Flaky workflows: Random failures
- Poor reliability in cloud environments
- User frustration

**Location**
File: core/state_manager.py
Lines: 317-343 (cross_verify_with_adapter() method)

**Solution Implementation**

Add exponential backoff retry for network errors.

Implementation:
- Retry on network errors: ConnectionError, Timeout
- Max retries: 3 attempts
- Backoff delays: 1s, 2s, 4s (exponential)
- Log retry attempts
- Give up after max retries with clear error

**Testing Plan**
- Test successful verification (no retries)
- Test transient error + success on retry
- Test 3 failures (max retries exceeded)
- Test backoff timing
- Mock network errors

**Acceptance Criteria**
- [ ] Exponential backoff retry (3 attempts)
- [ ] Delays: 1s, 2s, 4s
- [ ] Retry on network errors only
- [ ] Clear error after max retries
- [ ] Unit tests: 100% coverage
"""

bug_1193_desc = """**Problem Summary**
Audit log kept in memory, lost on crash. No durability.

**Impact**
- Data loss on crash
- Incomplete audit trail
- Compliance violation

**Location**
File: scripts/workflow_executor/base.py
Lines: 411-424 (audit log collection)

**Solution Implementation**

Write audit log to disk immediately (JSONL append-only format).

Implementation:
- Open audit log file in append mode
- Write each log entry as single JSON line immediately
- Flush after each write (durability)
- Use JSONL format (one JSON object per line)
- Store in .claude/audit/workflow-name/

**Testing Plan**
- Test log entries written immediately
- Test log survives crash (kill process mid-workflow)
- Test JSONL format valid
- Test log rotation (size limit)

**Acceptance Criteria**
- [ ] Immediate write to disk (JSONL format)
- [ ] Flush after each write
- [ ] Audit log survives crash
- [ ] Unit tests: 100% coverage
"""

# Low Priority Bugs

bug_1194_desc = """**Problem Summary**
State overwritten without backup. No rollback capability.

**Impact**
- Data loss if new state corrupted
- No rollback to previous version
- Manual recovery difficult

**Location**
File: core/state_manager.py
Line: 136 (save() method)

**Solution Implementation**

Create backup before save with rotation (keep last 3).

Implementation:
- Before saving, copy current state to backup
- Backup naming: state-file.backup.1, .backup.2, .backup.3
- Rotate backups (delete .backup.3, rename .backup.2 to .backup.3, etc.)
- Keep last 3 backups
- Provide restore_from_backup() method

**Testing Plan**
- Test backup created before save
- Test backup rotation (3 versions)
- Test restore from backup
- Test backup deletion when old

**Acceptance Criteria**
- [ ] Backup created before each save
- [ ] Keep last 3 backups
- [ ] restore_from_backup() method
- [ ] Unit tests: 100% coverage
"""

bug_1195_desc = """**Problem Summary**
No visibility into verification performance. No metrics.

**Impact**
- Cannot optimize slow verification
- No performance regression detection
- Unknown bottlenecks

**Location**
File: core/state_manager.py
Lines: 313-343 (cross_verify_with_adapter() method)

**Solution Implementation**

Add duration tracking and metrics.

Implementation:
- Track start/end time for verification
- Calculate average time per work item
- Log metrics to console and file
- Include in verification result dict
- Add to profiling reports

**Testing Plan**
- Test metrics collected during verification
- Test average calculation
- Test metrics logging
- Test metrics in result dict

**Acceptance Criteria**
- [ ] Duration tracking
- [ ] Average time per item metric
- [ ] Metrics logged and returned
- [ ] Unit tests: 100% coverage
"""

bug_1196_desc = """**Problem Summary**
No disk space check before save. Fails with unclear error.

**Impact**
- Cryptic error messages
- Partial writes if disk full
- Poor debugging experience

**Location**
File: core/state_manager.py
Line: 136 (save() method)

**Solution Implementation**

Check disk space before save with clear error.

Implementation:
- Use shutil.disk_usage() to check available space
- Require 2x state file size free space (safety margin)
- Raise clear error if insufficient space
- Suggest cleanup actions in error message

**Testing Plan**
- Test save with sufficient disk space
- Test save with insufficient space (mock)
- Test error message clarity
- Test safety margin calculation

**Acceptance Criteria**
- [ ] Disk space check before save
- [ ] 2x file size safety margin
- [ ] Clear error message with remediation
- [ ] Unit tests: 100% coverage
"""

# Update all bugs
medium_bugs = [
    (1187, bug_1187_desc),
    (1189, bug_1189_desc),
    (1190, bug_1190_desc),
    (1191, bug_1191_desc),
    (1192, bug_1192_desc),
    (1193, bug_1193_desc),
]

low_bugs = [
    (1194, bug_1194_desc),
    (1195, bug_1195_desc),
    (1196, bug_1196_desc),
]

print("Updating Medium priority bugs...")
for bug_id, description in medium_bugs:
    adapter.update_work_item(bug_id, fields={'System.Description': description})
    print(f"  ✅ Updated Bug #{bug_id}")

print("\nUpdating Low priority bugs...")
for bug_id, description in low_bugs:
    adapter.update_work_item(bug_id, fields={'System.Description': description})
    print(f"  ✅ Updated Bug #{bug_id}")

print("\n✅ All Medium and Low priority bugs updated with detailed solutions")
