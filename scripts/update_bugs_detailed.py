#!/usr/bin/env python3
"""Update bug tickets with detailed solutions - avoiding heredoc issues."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / ".claude/skills"))
from work_tracking import get_adapter

adapter = get_adapter()

# Bug descriptions stored as separate variables to avoid heredoc nesting

bug_1184_desc = """**Problem Summary**
Invalid user input causes infinite retry loop in _handle_unverified_state(). The method prompts user for choice (1/2/3) but doesn't enforce max retries.

**Impact**
- Workflow hang: User stuck in infinite prompt loop
- Poor UX: No escape from bad input
- Resource waste: Workflow session cannot timeout

**Location**
File: scripts/workflow_executor/base.py
Lines: 198-254 (_handle_unverified_state() method)

**Solution Implementation**

Add max retry limit (default: 5 attempts) with input validation. Abort workflow if max retries exceeded.

Implementation approach:
1. Add max_retries parameter to _handle_unverified_state()
2. Track retry count in while loop
3. Validate input against allowed choices
4. Display remaining attempts to user
5. Raise clear error when max retries exceeded

**Code Changes Required**
- File: scripts/workflow_executor/base.py
- Method: _handle_unverified_state() (lines 198-254)
- Add parameter: max_retries (default: 5)
- Add retry counter and validation
- Add helper: _get_validated_input() for reusable input validation

**Testing Plan**

1. Unit Tests
   - Test valid input on first attempt
   - Test invalid input once, then valid (retry works)
   - Test invalid input 5 times (max retries exceeded)
   - Test empty input and whitespace-only input

2. Integration Tests
   - Test user chooses option after invalid attempts
   - Test max retries exceeded (workflow aborts)

3. Edge Cases
   - Test very long input
   - Test special characters
   - Test numeric values outside range

**Acceptance Criteria**
- [ ] Max retry limit enforced (default: 5)
- [ ] Input validation: Only accept 1, 2, 3
- [ ] Clear error after max retries
- [ ] Retry counter displayed to user
- [ ] Unit tests: 100% coverage
- [ ] Integration test: Workflow aborts after 5 bad inputs
"""

bug_1185_desc = """**Problem Summary**
HMAC secret key cached in WorkflowState class variable forever, exposing it to memory dumps and debuggers.

**Impact**
- Security risk: Secret key exposed in memory dumps
- Credential leakage: Debuggers can inspect cached key
- Compliance violation: PCI/HIPAA require clearing secrets

**Location**
File: core/state_manager.py
Lines: 373-383 (_get_secret_key() method)

**Solution Implementation**

IMPORTANT: This is a TEMPORARY fix. HMAC will be replaced with Keychain Core (DID-based verifiable credentials) in future. Do NOT implement complex KDF - just clear the secret from memory after use.

Simple approach:
1. Remove class-level secret key cache
2. Get secret fresh on each use (from environment)
3. Clear secret from memory immediately after signing/verifying (del secret_key in finally block)
4. No caching, no complex key derivation needed

Why not KDF: HMAC signatures are temporary until Keychain Core migration provides DID-based identities and verifiable credentials with proper provenance.

**Code Changes Required**
- File: core/state_manager.py
- Method: _get_secret_key() (lines 373-383)
- Remove: Class-level _secret_key cache
- Add: del secret_key in finally blocks
- Methods to update: _sign_state(), verify_signature()

**Testing Plan**

1. Unit Tests
   - Test secret not cached (call twice, different objects)
   - Test signing works after clearing key
   - Test verification works after clearing key
   - Test key cleared on exception

2. Security Tests
   - Memory inspection: Key not in heap after signing
   - Exception safety: Key cleared even if signing fails

3. Integration Tests
   - Test workflow state save/load with non-cached key
   - Test signature verification after key cleared

**Acceptance Criteria**
- [ ] Secret key NOT cached in class variable
- [ ] Secret cleared from memory after use (del in finally)
- [ ] Signing and verification still work
- [ ] Unit tests: 100% coverage
- [ ] Memory test: Key not in heap after signing
- [ ] No performance regression

**Future Work (Keychain Core Integration)**
- Replace HMAC with DID-based signatures (Sprint 9+)
- Implement verifiable credentials for state provenance
- Add chain of custody tracking
"""

bug_1186_desc = """**Problem Summary**
input() blocks indefinitely if user walks away, causing workflows to hang forever. No timeout on approval gates.

**Impact**
- Workflow hang: Sessions left open indefinitely
- Resource waste: Server resources consumed
- Poor UX: Cannot abort remotely

**Location**
File: scripts/workflow_executor/base.py
Line: 199 (and other input() calls in approval gates)

**Solution Implementation**

Implement threading-based input timeout (works cross-platform unlike signal-based approach).

Implementation approach:
1. Create _get_user_input_with_timeout() method
2. Use threading.Thread with queue.Queue for timeout
3. Default timeout: 300 seconds (5 minutes)
4. Raise TimeoutError if user doesn't respond
5. Replace all input() calls with timeout version

Why threading: Signal-based timeout (signal.alarm) doesn't work on Windows. Threading approach is cross-platform compatible.

**Code Changes Required**
- File: scripts/workflow_executor/base.py
- Add method: _get_user_input_with_timeout(prompt, timeout_seconds)
- Replace all input() calls with timeout version
- Add imports: threading, queue
- Add config: input_timeout_seconds (default: 300)

**Testing Plan**

1. Unit Tests
   - Test normal input (user responds immediately)
   - Test timeout (mock queue.get() timeout)
   - Test thread cleanup after timeout
   - Test configurable timeout values

2. Integration Tests
   - Test approval gate with timeout
   - Test user responds before timeout (continues)
   - Test user doesn't respond (timeout, abort)

3. Edge Cases
   - Test timeout = 0 (immediate)
   - Test very long timeout
   - Test exception during input

4. Platform Tests
   - Test on Windows (threading required)
   - Test on Linux/macOS (threading also works)

**Acceptance Criteria**
- [ ] All input() calls replaced with timeout version
- [ ] Default timeout: 300 seconds
- [ ] Configurable timeout via config/env
- [ ] TimeoutError with clear message
- [ ] Unit tests: 100% coverage
- [ ] Integration test: Workflow aborts after timeout
- [ ] Works on Windows, Linux, macOS
"""

bug_1188_desc = """**Problem Summary**
Empty allowed_commands set skips validation entirely, allowing all subprocess commands. Logic flaw in validate_command() method.

**Impact**
- Security bypass: Empty allowlist = no restrictions
- Command injection: Arbitrary commands can run
- Silent failure: No warning when allowlist empty

**Location**
File: scripts/workflow_executor/subprocess_security.py
Lines: 202-210 (validate_command() method)

**Solution Implementation**

Fix empty allowlist validation with strict mode enforcement.

Implementation approach:
1. Add strict_mode parameter (default: True)
2. Raise SecurityError if allowlist empty in strict mode
3. Print warning if allowlist empty in non-strict mode
4. Provide clear error explaining security risk
5. Show allowed commands list when blocking

**Code Changes Required**
- File: scripts/workflow_executor/subprocess_security.py
- Method: validate_command() (lines 202-210)
- Add parameter: strict_mode (default: True)
- Add validation: Raise error if allowlist empty in strict mode
- Add warning: Print warning if empty in non-strict mode

**Testing Plan**

1. Unit Tests
   - Test empty allowlist + strict_mode=True (raises SecurityError)
   - Test empty allowlist + strict_mode=False (warning, allows)
   - Test non-empty allowlist (normal validation)
   - Test command in allowlist (passes)
   - Test command not in allowlist (blocked)

2. Integration Tests
   - Test workflow with empty allowlist in strict mode (fails)
   - Test workflow with populated allowlist (works)
   - Test workflow with strict_mode=False (warning)

3. Security Tests
   - Test command injection with empty allowlist blocked
   - Test command injection with populated allowlist blocked
   - Test bypass attempts

4. Edge Cases
   - Test allowlist with 1 command
   - Test allowlist with 100 commands
   - Test case-sensitive matching

**Acceptance Criteria**
- [ ] Empty allowlist raises SecurityError in strict mode
- [ ] Empty allowlist prints warning in non-strict mode
- [ ] Strict mode enabled by default
- [ ] Clear error message explains security risk
- [ ] Unit tests: 100% coverage
- [ ] Security test: Empty allowlist blocked in production
- [ ] Documentation: When to use strict_mode=False
"""

# Update bugs
updates = [
    (1184, bug_1184_desc),
    (1185, bug_1185_desc),
    (1186, bug_1186_desc),
    (1188, bug_1188_desc),
]

for bug_id, description in updates:
    adapter.update_work_item(bug_id, fields={'System.Description': description})
    print(f"✅ Updated Bug #{bug_id}")

print("\n✅ All remaining High priority bugs updated with detailed solutions")
