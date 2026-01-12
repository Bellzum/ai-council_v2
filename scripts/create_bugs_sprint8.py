#!/usr/bin/env python3
"""Create bug tickets from code review findings for Sprint 8."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / ".claude/skills"))

from work_tracking import get_adapter

adapter = get_adapter()
SPRINT_8 = "Trusted AI Development Workbench\\Sprint 8"

# Bug definitions - simplified for reliability
bugs = [
    # CRITICAL
    {
        "title": "State Manager: Race Condition - No File Locking",
        "description": "Multiple workflow instances can modify same state file simultaneously, causing corruption. Need fcntl.flock() (Unix) or msvcrt.locking() (Windows). Location: core/state_manager.py:124-136",
        "priority": 1,
        "severity": "1 - Critical",
    },
    {
        "title": "State Manager: Unhandled JSON Decode Error",
        "description": "Corrupted state file causes unhandled JSONDecodeError, workflow crashes with no remediation. Need try/catch with user prompt. Location: core/state_manager.py:76",
        "priority": 1,
        "severity": "1 - Critical",
    },
    {
        "title": "State Manager: No Atomic Write - Partial Save Corruption",
        "description": "Power loss during write_text() creates corrupt file. Need atomic write pattern (temp file + rename). Location: core/state_manager.py:136",
        "priority": 1,
        "severity": "1 - Critical",
    },
    {
        "title": "Cross-Verification: No Rate Limiting - API Abuse Risk",
        "description": "Large state files (100+ items) query adapter in tight loop without delay, can exceed API rate limits. Need 1 second pause every 10 items. Location: core/state_manager.py:313-343",
        "priority": 2,
        "severity": "2 - High",
    },
    {
        "title": "Workflow Orchestrator: Input Validation Missing - Infinite Loop Risk",
        "description": "Invalid user input causes infinite retry loop. Need max retry limit of 5 attempts. Location: scripts/workflow_executor/base.py:198-254",
        "priority": 2,
        "severity": "2 - High",
    },
    # HIGH PRIORITY
    {
        "title": "State Manager: Secret Key Stored in Memory - Security Risk",
        "description": "HMAC secret key cached in class variable forever. Need to clear key from memory after use (not KDF - temporary fix until Keychain Core). Location: core/state_manager.py:373-383",
        "priority": 2,
        "severity": "2 - High",
    },
    {
        "title": "Workflow Orchestrator: No Timeout on input() - Hang Forever",
        "description": "input() blocks indefinitely if user walks away. Need threading-based timeout (300 seconds). Location: scripts/workflow_executor/base.py:199",
        "priority": 2,
        "severity": "2 - High",
    },
    {
        "title": "State Manager: Cleanup Only Deletes Completed States",
        "description": "cleanup_old_states() only deletes 'completed' status. Failed/interrupted states accumulate. Need include_failed and include_interrupted parameters. Location: core/state_manager.py:595",
        "priority": 3,
        "severity": "3 - Medium",
    },
    {
        "title": "Subprocess Security: Empty Allowlist Allows All Commands",
        "description": "Empty allowed_commands set skips validation, all commands allowed. Need warning or strict_mode to require non-empty allowlist. Location: scripts/workflow_executor/subprocess_security.py:202-210",
        "priority": 2,
        "severity": "2 - High",
    },
    # MEDIUM PRIORITY
    {
        "title": "State Manager: No State File Size Limit - DoS Risk",
        "description": "Malicious 100MB+ state file causes memory exhaustion. Need 10MB maximum file size check before loading. Location: core/state_manager.py:76",
        "priority": 3,
        "severity": "3 - Medium",
    },
    {
        "title": "Workflow Orchestrator: Evidence Not Validated for Required Keys",
        "description": "Steps can return empty dict as evidence. Need non-empty validation and optional required_evidence_keys. Location: scripts/workflow_executor/base.py:360-362",
        "priority": 3,
        "severity": "3 - Medium",
    },
    {
        "title": "State Manager: JSON Serialization Errors Not Handled",
        "description": "Non-serializable objects (datetime, Path) cause TypeError on save. Need custom JSON encoder or error handling. Location: core/state_manager.py:136",
        "priority": 3,
        "severity": "3 - Medium",
    },
    {
        "title": "Cross-Verification: No Retry Logic on Network Errors",
        "description": "Transient network errors fail verification permanently. Need exponential backoff retry (3 attempts: 1s, 2s, 4s). Location: core/state_manager.py:317-343",
        "priority": 3,
        "severity": "3 - Medium",
    },
    {
        "title": "Workflow Orchestrator: Audit Log Lost on Crash",
        "description": "Audit log kept in memory, lost on crash. Need immediate write to disk (JSONL append-only format). Location: scripts/workflow_executor/base.py:411-424",
        "priority": 3,
        "severity": "3 - Medium",
    },
    # LOW PRIORITY
    {
        "title": "State Manager: No State Backup Before Modification",
        "description": "State overwritten without backup. Need backup creation with rotation (keep last 3). Impact: LOW",
        "priority": 4,
        "severity": "4 - Low",
    },
    {
        "title": "State Manager: No Performance Metrics on Verification",
        "description": "No visibility into verification performance. Need duration tracking and avg time per item metrics. Impact: LOW",
        "priority": 4,
        "severity": "4 - Low",
    },
    {
        "title": "State Manager: No Disk Space Check Before Save",
        "description": "No check for disk space before save. Need shutil.disk_usage() check with clear error message. Impact: LOW",
        "priority": 4,
        "severity": "4 - Low",
    },
]

print(f"Creating {len(bugs)} bug tickets for Sprint 8...")
print("=" * 80)

created = []
for i, bug in enumerate(bugs, 1):
    print(f"\n[{i}/{len(bugs)}] {bug['title']}")
    try:
        result = adapter.create_work_item(
            work_item_type="Bug",
            title=bug["title"],
            description=bug["description"],
            iteration=SPRINT_8,
        )

        work_item_id = result.get("id")

        # Set priority, severity, and tags via update
        try:
            adapter.update_work_item(
                work_item_id=work_item_id,
                fields={
                    "Microsoft.VSTS.Common.Priority": bug["priority"],
                    "Microsoft.VSTS.Common.Severity": bug["severity"],
                    "System.Tags": "code-review; external-enforcement; security"
                }
            )
        except Exception as e:
            print(f"  ⚠️  Could not set priority/severity/tags: {e}")

        created.append({"id": work_item_id, "title": bug["title"], "priority": bug["priority"]})
        print(f"  ✅ Created Bug #{work_item_id}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

print("\n" + "=" * 80)
print(f"✅ Created {len(created)} bug tickets in Sprint 8")
print("=" * 80)

# Summary
priority_names = {1: "Critical", 2: "High", 3: "Medium", 4: "Low"}
priority_counts = {}
for item in created:
    p = item["priority"]
    priority_counts[p] = priority_counts.get(p, 0) + 1

print("\nSummary by Priority:")
for priority in sorted(priority_counts.keys()):
    print(f"  {priority_names[priority]}: {priority_counts[priority]} bug(s)")

print("\nCreated Bug IDs:")
for item in created:
    print(f"  #{item['id']}: {item['title']}")
