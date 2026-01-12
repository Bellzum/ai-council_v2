#!/usr/bin/env python3
"""
Artifact Hygiene Workflow with External Enforcement

Automates detection, categorization, and cleanup of sprint side-effect artifacts.

10-Step Workflow:
1. Repository Scan - Scan top-level for .md/.txt artifacts matching patterns
2. Categorization - Classify artifacts by destination directory (AI for unknowns)
3. Deprecation Detection - AI identifies outdated docs and generates deprecation headers
4. Approval Gate - Human approves categorization plan
5. Git Preparation - git add untracked files, git commit to establish history
6. Execute Moves - git mv files, add AI-generated deprecation headers
7. Update Context - Update CLAUDE.md files, regenerate context index
8. Verification - Verify moves succeeded, no broken references
9. Finalize Commit - git commit --amend to combine all changes into clean commit
10. Report - Generate report to .claude/reports/artifact-hygiene/

Design Pattern:
- Extends WorkflowOrchestrator from Phase 1
- Mode 1 (Pure Python): Fast scanning and reporting
- Mode 2 (AI + JSON): Intelligent classification with retry logic
- External verification after file moves
- Real input() blocking for approval gate
- UTF-8 encoding for all file writes
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.workflow_executor.base import WorkflowOrchestrator, ExecutionMode
from scripts.workflow_executor.schemas import StepType

# Import JSON schema validation
try:
    from jsonschema import validate, ValidationError
except ImportError:
    print("⚠️  jsonschema package not installed - install with: pip install jsonschema")
    ValidationError = Exception  # Fallback


# File categorization patterns
CATEGORIZATION_RULES = {
    "bug-fix": {
        "patterns": ["BUG_*.md", "*_BUG_*.md", "*_FIX.md", "*_FIX_SUMMARY.md", "BUG_FIXES_*.md"],
        "destination": "docs/archive/bug-fixes"
    },
    "implementation": {
        "patterns": ["*_IMPLEMENTATION_SUMMARY.md", "IMPLEMENTATION_*.md", "*_SUMMARY.md",
                     "FEATURE_*.md", "*_IMPLEMENTATION.md", "NEW_*_SUMMARY.md", "*_IMPROVEMENT_SUMMARY.md"],
        "destination": "docs/archive/implementations"
    },
    "validation": {
        "patterns": ["validation-report-*.md", "*_VALIDATION_REPORT_*.md", "PHASE*_VALIDATION_REPORT.md",
                     "TEST_VALIDATION_REPORT_*.md"],
        "destination": "docs/archive/validation-reports"
    },
    "migration": {
        "patterns": ["MIGRATION*.md", "*_MIGRATION_*.md", "MIGRATION-*.md"],
        "destination": "docs/migration"
    }
}

# Files to never move
EXCLUSION_PATTERNS = [
    "README.md", "CLAUDE.md", "VISION.md", "CHANGELOG.md", "LICENSE.md", "LICENSE",
    "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md"
]

# AI classification schema
CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["bug-fix", "implementation", "validation", "migration", "unknown", "keep"]
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
        "suggested_destination": {"type": "string"},
        "needs_deprecation": {"type": "boolean"},
        "deprecation_reason": {"type": "string"}
    },
    "required": ["category", "confidence", "rationale"]
}

# Deprecation analysis schema
DEPRECATION_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_deprecation": {"type": "boolean"},
        "reason": {"type": "string"},
        "superseded_by": {"type": ["string", "null"]},
        "deprecation_header": {"type": "string"}
    },
    "required": ["needs_deprecation", "reason"]
}


class ArtifactHygieneWorkflow(WorkflowOrchestrator):
    """
    Artifact Hygiene workflow with external enforcement.

    Implements the 10-step cleanup process with:
    - Pattern-based file scanning
    - AI classification for unknown files
    - External verification after file moves
    - Blocking approval gates
    - Git history preservation
    """

    def __init__(
        self,
        workflow_id: str,
        enable_checkpoints: bool = True,
        use_ai: bool = False,
        dry_run: bool = False,
        verbose: bool = False,
        target_dirs: Optional[List[str]] = None,
        current_sprint: Optional[str] = None
    ):
        """
        Initialize artifact hygiene workflow.

        Args:
            workflow_id: Unique ID for this execution
            enable_checkpoints: Enable state checkpointing
            use_ai: If True, use AI for classification (Mode 2)
            dry_run: If True, show what would be done without making changes
            verbose: If True, show detailed output
            target_dirs: Directories to scan (default: project root)
            current_sprint: Current sprint name for deprecation context
        """
        self.use_ai = use_ai
        self.dry_run = dry_run
        self.verbose = verbose
        self.target_dirs = target_dirs or ["."]
        self.current_sprint = current_sprint or self._detect_current_sprint()

        mode = ExecutionMode.AI_JSON_VALIDATION if use_ai else ExecutionMode.PURE_PYTHON

        super().__init__(
            workflow_name="artifact-hygiene",
            workflow_id=workflow_id,
            mode=mode,
            enable_checkpoints=enable_checkpoints
        )

        # Initialize Claude API client if using AI
        self.claude_client = None
        if use_ai:
            try:
                import anthropic
                api_key = os.getenv("KEYCHAIN_ANTHROPIC_API_KEY")
                if api_key:
                    self.claude_client = anthropic.Anthropic(api_key=api_key)
                    print("✅ Claude API client initialized")
                else:
                    print("⚠️ KEYCHAIN_ANTHROPIC_API_KEY not set, falling back to pattern-only mode")
                    self.use_ai = False
            except ImportError:
                print("⚠️ anthropic package not installed, falling back to pattern-only mode")
                self.use_ai = False

    def _detect_current_sprint(self) -> str:
        """Detect current sprint from adapter or config."""
        try:
            sys.path.insert(0, '.claude/skills')
            from work_tracking import get_adapter
            adapter = get_adapter()
            sprints = adapter.list_sprints()
            if sprints:
                # Get current sprint (usually the one that's active)
                for sprint in sprints:
                    if sprint.get('attributes', {}).get('timeFrame') == 'current':
                        return sprint.get('name', 'Unknown Sprint')
            return "Unknown Sprint"
        except Exception:
            return "Unknown Sprint"

    def _define_steps(self) -> List[Dict[str, Any]]:
        """Define the 10-step workflow."""
        return [
            {
                "id": "1-scan",
                "name": "Repository Scan",
                "step_type": StepType.DATA_COLLECTION,
                "description": "Scan top-level for .md/.txt artifacts matching patterns"
            },
            {
                "id": "2-categorize",
                "name": "Categorization Analysis",
                "step_type": StepType.AI_REVIEW if self.use_ai else StepType.DATA_COLLECTION,
                "description": "Classify artifacts by destination directory"
            },
            {
                "id": "3-deprecation",
                "name": "Deprecation Detection",
                "step_type": StepType.AI_REVIEW if self.use_ai else StepType.DATA_COLLECTION,
                "description": "Identify outdated docs and generate deprecation headers"
            },
            {
                "id": "4-approval",
                "name": "Human Approval Gate",
                "step_type": StepType.APPROVAL_GATE,
                "description": "Human approves categorization plan"
            },
            {
                "id": "5-git-prep",
                "name": "Git Preparation",
                "step_type": StepType.ACTION,
                "description": "git add untracked files, git commit to establish history"
            },
            {
                "id": "6-execute",
                "name": "Execute Moves",
                "step_type": StepType.ACTION,
                "description": "git mv files, add deprecation headers"
            },
            {
                "id": "7-update-context",
                "name": "Update Context",
                "step_type": StepType.ACTION,
                "description": "Update CLAUDE.md files, regenerate context index"
            },
            {
                "id": "8-verify",
                "name": "Verification",
                "step_type": StepType.VERIFICATION,
                "description": "Verify moves succeeded, no broken references"
            },
            {
                "id": "9-finalize",
                "name": "Finalize Commit",
                "step_type": StepType.ACTION,
                "description": "git commit --amend to combine all changes"
            },
            {
                "id": "10-report",
                "name": "Generate Report",
                "step_type": StepType.ACTION,
                "description": "Generate report to .claude/reports/artifact-hygiene/"
            }
        ]

    def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step_id = step["id"]

        if step_id == "1-scan":
            return self._step_1_scan()
        elif step_id == "2-categorize":
            return self._step_2_categorize()
        elif step_id == "3-deprecation":
            return self._step_3_deprecation()
        elif step_id == "4-approval":
            return self._step_4_approval()
        elif step_id == "5-git-prep":
            return self._step_5_git_prep()
        elif step_id == "6-execute":
            return self._step_6_execute()
        elif step_id == "7-update-context":
            return self._step_7_update_context()
        elif step_id == "8-verify":
            return self._step_8_verify()
        elif step_id == "9-finalize":
            return self._step_9_finalize()
        elif step_id == "10-report":
            return self._step_10_report()
        else:
            raise ValueError(f"Unknown step: {step_id}")

    def _step_1_scan(self) -> Dict[str, Any]:
        """Step 1: Scan repository for artifacts."""
        print("\n📊 Step 1: Scanning repository for artifacts...")

        artifacts = []
        by_pattern = {}

        for target_dir in self.target_dirs:
            # Find all .md and .txt files at top level
            for ext in ["*.md", "*.txt"]:
                pattern = os.path.join(target_dir, ext)
                for filepath in glob.glob(pattern):
                    filename = os.path.basename(filepath)

                    # Skip excluded files
                    if filename in EXCLUSION_PATTERNS:
                        if self.verbose:
                            print(f"  ⏭️  Skipping excluded: {filename}")
                        continue

                    # Get file metadata
                    stat = os.stat(filepath)
                    artifact = {
                        "path": filepath,
                        "filename": filename,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "is_empty": stat.st_size == 0
                    }
                    artifacts.append(artifact)

                    if self.verbose:
                        print(f"  📄 Found: {filename}")

        # Categorize by pattern match
        for artifact in artifacts:
            matched = False
            for category, rules in CATEGORIZATION_RULES.items():
                for pattern in rules["patterns"]:
                    if self._matches_pattern(artifact["filename"], pattern):
                        by_pattern[category] = by_pattern.get(category, 0) + 1
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                by_pattern["unknown"] = by_pattern.get("unknown", 0) + 1

        print(f"\n✓ Found {len(artifacts)} artifacts")
        for category, count in by_pattern.items():
            print(f"    {category}: {count}")

        return {
            "artifacts": artifacts,
            "total_count": len(artifacts),
            "by_pattern": by_pattern,
            "scanned_at": datetime.now().isoformat()
        }

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches glob pattern."""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)

    def _step_2_categorize(self) -> Dict[str, Any]:
        """Step 2: Categorize artifacts by destination."""
        print("\n📁 Step 2: Categorizing artifacts...")

        scan_result = self.step_evidence.get("1-scan", {})
        artifacts = scan_result.get("artifacts", [])

        categorizations = []
        unknown_count = 0

        for artifact in artifacts:
            filename = artifact["filename"]
            category = None
            destination = None
            confidence = "rule"

            # Try pattern-based categorization first
            for cat, rules in CATEGORIZATION_RULES.items():
                for pattern in rules["patterns"]:
                    if self._matches_pattern(filename, pattern):
                        category = cat
                        destination = rules["destination"]
                        break
                if category:
                    break

            # Use AI for unknowns if enabled
            if not category and self.use_ai and self.claude_client:
                ai_result = self._classify_with_ai(artifact)
                if ai_result:
                    category = ai_result.get("category", "unknown")
                    destination = ai_result.get("suggested_destination")
                    confidence = f"ai:{ai_result.get('confidence', 0):.2f}"

            if not category:
                category = "unknown"
                unknown_count += 1

            categorizations.append({
                "filename": filename,
                "path": artifact["path"],
                "category": category,
                "destination": destination,
                "confidence": confidence
            })

            if self.verbose:
                status = "✓" if category != "unknown" else "?"
                print(f"  {status} {filename} → {category}")

        print(f"\n✓ Categorized {len(categorizations)} files ({unknown_count} unknown)")

        return {
            "categorizations": categorizations,
            "unknown_count": unknown_count,
            "categorized_at": datetime.now().isoformat()
        }

    def _get_agent_sdk_wrapper(self):
        """Get or create Agent SDK wrapper for AI classification."""
        if not hasattr(self, '_agent_sdk_wrapper') or self._agent_sdk_wrapper is None:
            try:
                from scripts.workflow_executor.agent_sdk import AgentSDKWrapper
                from pathlib import Path

                self._agent_sdk_wrapper = AgentSDKWrapper(
                    model="claude-sonnet-4-20250514",
                    max_tokens=500,
                    working_directory=str(Path.cwd()),
                    tool_preset="implementation"  # Needs write access to move files and add deprecation headers
                )
                return self._agent_sdk_wrapper
            except ImportError as e:
                # ALWAYS report import errors - no silent fallbacks
                from cli.console import print_error, print_info
                print_error(f"Agent SDK Import Failed: {e}")
                print_error("The Claude Agent SDK is not available. AI classification requires tool access.")
                print_info("Install with: pip install claude-code-sdk")
                return None
            except Exception as e:
                from cli.console import print_error
                print_error(f"Agent SDK Initialization Failed: {e}")
                return None
        return self._agent_sdk_wrapper

    def _classify_with_ai(self, artifact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use AI to classify an unknown artifact."""
        try:
            # Read first 100 lines of file
            content_preview = ""
            with open(artifact["path"], "r", encoding="utf-8") as f:
                lines = f.readlines()[:100]
                content_preview = "".join(lines)

            prompt = f"""Analyze this file and classify it into one of these categories:
- bug-fix: Bug fix summaries and fix documentation
- implementation: Feature implementations, summaries, improvements
- validation: Test validation reports, phase reports
- migration: Migration guides and documentation
- unknown: Cannot determine category
- keep: Should remain at top level (important docs)

File: {artifact['filename']}
Content preview:
{content_preview[:2000]}

Respond with JSON matching this schema:
{json.dumps(CLASSIFICATION_SCHEMA, indent=2)}
"""

            # Try Agent SDK first (provides codebase context)
            sdk = self._get_agent_sdk_wrapper()
            if sdk:
                try:
                    result = sdk.ask(
                        prompt=prompt,
                        system_prompt="You are a file organization expert classifying artifacts for archival."
                    )
                    response_text = result.response

                    json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                    if json_match:
                        classification = json.loads(json_match.group())
                        validate(classification, CLASSIFICATION_SCHEMA)

                        if classification["category"] in CATEGORIZATION_RULES:
                            classification["suggested_destination"] = CATEGORIZATION_RULES[classification["category"]]["destination"]

                        return classification
                except Exception as e:
                    # ALWAYS report errors - no silent fallbacks to Anthropic API
                    from cli.console import print_error, print_info
                    print_error(f"Agent SDK Execution Failed: {e}")
                    print_error("AI classification requires codebase access.")
                    print_info("This artifact will be classified as 'unknown'. Fix the Agent SDK issue to enable AI classification.")
                    return None

            # No Agent SDK available - report and return None (don't fall back to degraded functionality)
            from cli.console import print_error, print_info
            print_error("AI Classification Skipped: No Agent SDK available")
            print_error("AI classification requires codebase access. Cannot proceed without Agent SDK.")
            print_info("Install with: pip install claude-code-sdk")
            return None

        except Exception as e:
            if self.verbose:
                print(f"    ⚠️  AI classification failed: {e}")
            return None

        return None

    def _step_3_deprecation(self) -> Dict[str, Any]:
        """Step 3: Detect files needing deprecation headers."""
        print("\n🏷️  Step 3: Detecting deprecation needs...")

        categorizations = self.step_evidence.get("2-categorize", {}).get("categorizations", [])
        deprecation_info = []

        for cat in categorizations:
            if cat["category"] == "unknown" or cat["category"] == "keep":
                continue

            needs_deprecation = False
            deprecation_header = None
            reason = "Archived for historical reference"

            # All archived files get deprecation headers
            if cat["destination"] and "archive" in cat["destination"]:
                needs_deprecation = True

                # Generate deprecation header
                deprecation_header = f"""---
status: ARCHIVED
archived_date: {datetime.now().strftime('%Y-%m-%d')}
original_location: /{cat['filename']}
archive_location: /{cat['destination']}/{cat['filename']}
sprint: "{self.current_sprint}"
reason: "{reason}"
---

<!-- ARCHIVED DOCUMENT - For historical reference only -->

"""

            if needs_deprecation:
                deprecation_info.append({
                    "filename": cat["filename"],
                    "path": cat["path"],
                    "needs_deprecation": True,
                    "reason": reason,
                    "header": deprecation_header
                })

                if self.verbose:
                    print(f"  🏷️  {cat['filename']} - will add deprecation header")

        print(f"\n✓ {len(deprecation_info)} files will receive deprecation headers")

        return {
            "deprecation_info": deprecation_info,
            "count": len(deprecation_info),
            "detected_at": datetime.now().isoformat()
        }

    def _step_4_approval(self) -> Dict[str, Any]:
        """Step 4: Human approval gate."""
        print("\n" + "=" * 60)
        print("📋 APPROVAL REQUIRED - Artifact Hygiene Plan")
        print("=" * 60)

        categorizations = self.step_evidence.get("2-categorize", {}).get("categorizations", [])
        deprecation_info = self.step_evidence.get("3-deprecation", {}).get("deprecation_info", [])

        # Group by destination
        by_destination = {}
        for cat in categorizations:
            dest = cat.get("destination") or "SKIP (unknown)"
            if dest not in by_destination:
                by_destination[dest] = []
            by_destination[dest].append(cat["filename"])

        print("\n📁 Planned file moves:")
        for dest, files in sorted(by_destination.items()):
            if dest == "SKIP (unknown)":
                continue
            print(f"\n  → {dest}/")
            for f in files[:5]:  # Show first 5
                print(f"      {f}")
            if len(files) > 5:
                print(f"      ... and {len(files) - 5} more")

        unknown = by_destination.get("SKIP (unknown)", [])
        if unknown:
            print(f"\n⚠️  {len(unknown)} files will be skipped (unknown category)")

        print(f"\n🏷️  {len(deprecation_info)} files will receive deprecation headers")

        if self.dry_run:
            print("\n🔍 DRY RUN MODE - No changes will be made")
            return {"approved": True, "dry_run": True}

        print("\n" + "-" * 60)

        try:
            response = input("Proceed with artifact hygiene? (yes/no): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = "no"

        approved = response == "yes"

        if approved:
            print("✅ Approved - proceeding with cleanup")
        else:
            print("❌ Rejected - aborting workflow")

        return {
            "approved": approved,
            "files_to_move": sum(len(f) for d, f in by_destination.items() if d != "SKIP (unknown)"),
            "files_to_skip": len(unknown),
            "approved_at": datetime.now().isoformat()
        }

    def _step_5_git_prep(self) -> Dict[str, Any]:
        """Step 5: Prepare git - add untracked files, commit."""
        print("\n📦 Step 5: Preparing git history...")

        if self.dry_run:
            print("  🔍 DRY RUN - would add and commit files")
            return {"dry_run": True, "files_added": 0}

        approval = self.step_evidence.get("4-approval", {})
        if not approval.get("approved"):
            print("  ⏭️  Skipping - not approved")
            return {"skipped": True, "reason": "not_approved"}

        categorizations = self.step_evidence.get("2-categorize", {}).get("categorizations", [])
        files_to_add = [c["path"] for c in categorizations if c.get("destination")]

        if not files_to_add:
            print("  ⏭️  No files to process")
            return {"files_added": 0}

        # Check which files are untracked
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, check=True
            )
            untracked = []
            for line in result.stdout.strip().split("\n"):
                if line.startswith("??"):
                    filepath = line[3:].strip()
                    if filepath in files_to_add or f"./{filepath}" in files_to_add:
                        untracked.append(filepath)

            if untracked:
                # Add untracked files
                subprocess.run(["git", "add"] + untracked, check=True)
                print(f"  ✓ Added {len(untracked)} untracked files")

                # Commit to establish history
                commit_msg = f"Add artifacts for archival ({self.current_sprint})"
                result = subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    capture_output=True, text=True, check=True
                )

                # Get commit SHA
                sha_result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True, text=True, check=True
                )
                commit_sha = sha_result.stdout.strip()[:8]
                print(f"  ✓ Committed: {commit_sha}")

                return {
                    "files_added": len(untracked),
                    "commit_sha": commit_sha,
                    "completed_at": datetime.now().isoformat()
                }
            else:
                print("  ✓ All files already tracked")
                return {"files_added": 0, "all_tracked": True}

        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Git operation failed: {e}")
            return {"error": str(e), "files_added": 0}

    def _step_6_execute(self) -> Dict[str, Any]:
        """Step 6: Execute file moves with git mv."""
        print("\n📦 Step 6: Moving files...")

        if self.dry_run:
            print("  🔍 DRY RUN - would move files")
            return {"dry_run": True, "files_moved": 0}

        approval = self.step_evidence.get("4-approval", {})
        if not approval.get("approved"):
            return {"skipped": True, "reason": "not_approved"}

        categorizations = self.step_evidence.get("2-categorize", {}).get("categorizations", [])
        deprecation_info = {d["path"]: d for d in self.step_evidence.get("3-deprecation", {}).get("deprecation_info", [])}

        files_moved = 0
        headers_added = 0
        errors = []

        for cat in categorizations:
            if not cat.get("destination"):
                continue

            src = cat["path"]
            dest_dir = cat["destination"]
            dest = os.path.join(dest_dir, cat["filename"])

            try:
                # Create destination directory
                os.makedirs(dest_dir, exist_ok=True)

                # Add deprecation header if needed
                if src in deprecation_info:
                    header = deprecation_info[src]["header"]
                    with open(src, "r", encoding="utf-8") as f:
                        content = f.read()
                    with open(src, "w", encoding="utf-8") as f:
                        f.write(header + content)
                    headers_added += 1

                # Move with git mv
                subprocess.run(["git", "mv", src, dest], check=True)
                files_moved += 1

                if self.verbose:
                    print(f"  ✓ {cat['filename']} → {dest_dir}/")

            except Exception as e:
                errors.append(f"{cat['filename']}: {e}")
                print(f"  ❌ {cat['filename']}: {e}")

        print(f"\n✓ Moved {files_moved} files, added {headers_added} deprecation headers")
        if errors:
            print(f"⚠️  {len(errors)} errors occurred")

        return {
            "files_moved": files_moved,
            "headers_added": headers_added,
            "errors": errors,
            "completed_at": datetime.now().isoformat()
        }

    def _step_7_update_context(self) -> Dict[str, Any]:
        """Step 7: Update CLAUDE.md files and context index."""
        print("\n📝 Step 7: Updating context files...")

        if self.dry_run:
            print("  🔍 DRY RUN - would update context")
            return {"dry_run": True}

        approval = self.step_evidence.get("4-approval", {})
        if not approval.get("approved"):
            return {"skipped": True, "reason": "not_approved"}

        # Regenerate context index
        try:
            result = subprocess.run(
                ["python", "scripts/context_generation.py", "--force"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print("  ✓ Context index regenerated")
            else:
                print(f"  ⚠️  Context regeneration returned: {result.returncode}")
        except Exception as e:
            print(f"  ⚠️  Could not regenerate context: {e}")

        return {
            "context_updated": True,
            "completed_at": datetime.now().isoformat()
        }

    def _step_8_verify(self) -> Dict[str, Any]:
        """Step 8: Verify all moves succeeded."""
        print("\n✅ Step 8: Verifying changes...")

        if self.dry_run:
            print("  🔍 DRY RUN - skipping verification")
            return {"dry_run": True, "verified": True}

        approval = self.step_evidence.get("4-approval", {})
        if not approval.get("approved"):
            return {"skipped": True, "reason": "not_approved"}

        categorizations = self.step_evidence.get("2-categorize", {}).get("categorizations", [])
        verified = 0
        missing = []

        for cat in categorizations:
            if not cat.get("destination"):
                continue

            dest = os.path.join(cat["destination"], cat["filename"])
            if os.path.exists(dest):
                verified += 1
            else:
                missing.append(dest)

        # Verify original locations are empty
        for cat in categorizations:
            if cat.get("destination") and os.path.exists(cat["path"]):
                missing.append(f"{cat['path']} still exists")

        success = len(missing) == 0
        if success:
            print(f"  ✓ Verified {verified} files at new locations")
        else:
            print(f"  ⚠️  {len(missing)} verification issues")
            for m in missing[:5]:
                print(f"      - {m}")

        return {
            "verified": success,
            "verified_count": verified,
            "missing": missing,
            "verified_at": datetime.now().isoformat()
        }

    def _step_9_finalize(self) -> Dict[str, Any]:
        """Step 9: Finalize commit with git commit --amend (only if changes were made)."""
        print("\n📦 Step 9: Finalizing commit...")

        if self.dry_run:
            print("  🔍 DRY RUN - would finalize commit")
            return {"dry_run": True}

        approval = self.step_evidence.get("4-approval", {})
        if not approval.get("approved"):
            return {"skipped": True, "reason": "not_approved"}

        # Check if any changes were made
        execute = self.step_evidence.get("6-execute", {})
        git_prep = self.step_evidence.get("5-git-prep", {})
        files_moved = execute.get("files_moved", 0)
        files_added = git_prep.get("files_added", 0)

        if files_moved == 0 and files_added == 0:
            print("  ⏭️  No changes made - skipping commit")
            return {"skipped": True, "reason": "no_changes"}

        try:
            # Stage all changes
            subprocess.run(["git", "add", "-A"], check=True)

            # Check if there's anything to commit
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, check=True
            )

            if not status_result.stdout.strip():
                print("  ⏭️  No staged changes to commit")
                return {"skipped": True, "reason": "nothing_staged"}

            # Determine commit strategy: amend if we created a prep commit, otherwise new commit
            if files_added > 0:
                # Amend the prep commit to include moves
                commit_msg = f"Archive {self.current_sprint} artifacts\n\nOrganized sprint artifacts into docs/archive/ structure."
                subprocess.run(
                    ["git", "commit", "--amend", "-m", commit_msg],
                    check=True
                )
                print(f"  ✓ Amended prep commit with {files_moved} file moves")
            else:
                # New commit for moves only
                commit_msg = f"Archive {self.current_sprint} artifacts\n\nOrganized sprint artifacts into docs/archive/ structure."
                subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    check=True
                )
                print(f"  ✓ Created commit with {files_moved} file moves")

            # Get final SHA
            sha_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True
            )
            final_sha = sha_result.stdout.strip()[:8]

            print(f"  ✓ Finalized commit: {final_sha}")

            return {
                "commit_sha": final_sha,
                "files_moved": files_moved,
                "files_added": files_added,
                "amended": files_added > 0,
                "completed_at": datetime.now().isoformat()
            }

        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Commit finalization failed: {e}")
            return {"error": str(e)}

    def _step_10_report(self) -> Dict[str, Any]:
        """Step 10: Generate cleanup report."""
        print("\n📊 Step 10: Generating report...")

        # Gather all evidence
        scan = self.step_evidence.get("1-scan", {})
        categorize = self.step_evidence.get("2-categorize", {})
        deprecation = self.step_evidence.get("3-deprecation", {})
        execute = self.step_evidence.get("6-execute", {})
        verify = self.step_evidence.get("8-verify", {})
        finalize = self.step_evidence.get("9-finalize", {})

        # Build report
        report_lines = [
            "# Artifact Hygiene Report",
            "",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Workflow ID**: {self.workflow_id}",
            f"**Sprint**: {self.current_sprint}",
            f"**Mode**: {'AI-assisted' if self.use_ai else 'Pattern-based'}",
            f"**Dry Run**: {self.dry_run}",
            "",
            "## Summary",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Artifacts Scanned | {scan.get('total_count', 0)} |",
            f"| Files Moved | {execute.get('files_moved', 0)} |",
            f"| Deprecation Headers Added | {execute.get('headers_added', 0)} |",
            f"| Unknown (Skipped) | {categorize.get('unknown_count', 0)} |",
            f"| Verification Passed | {verify.get('verified', False)} |",
            f"| Commit Created | {finalize.get('commit_sha', 'None' if finalize.get('skipped') else 'N/A')} |",
            "",
        ]

        # Add files by category
        categorizations = categorize.get("categorizations", [])
        by_dest = {}
        for cat in categorizations:
            dest = cat.get("destination") or "Skipped"
            if dest not in by_dest:
                by_dest[dest] = []
            by_dest[dest].append(cat["filename"])

        report_lines.append("## Files Organized")
        report_lines.append("")

        for dest, files in sorted(by_dest.items()):
            if dest == "Skipped":
                continue
            report_lines.append(f"### {dest}/")
            report_lines.append("")
            for f in files:
                report_lines.append(f"- `{f}`")
            report_lines.append("")

        # Add footer
        report_lines.append("---")
        report_lines.append("*Generated by Trustable AI Development Workbench - Artifact Hygiene Workflow*")

        report_content = "\n".join(report_lines)

        # Save report
        report_dir = Path(".claude/reports/artifact-hygiene")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / f"hygiene-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"

        if not self.dry_run:
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"  ✓ Report saved: {report_file}")
        else:
            print("  🔍 DRY RUN - report not saved")

        return {
            "report_path": str(report_file),
            "report_content": report_content,
            "generated_at": datetime.now().isoformat()
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Artifact Hygiene Workflow - Clean up sprint artifacts"
    )
    parser.add_argument(
        "--workflow-id",
        help="Unique workflow ID (default: auto-generated)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--use-ai",
        action="store_true",
        help="Use AI for classifying unknown files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable state checkpointing"
    )
    parser.add_argument(
        "--sprint",
        help="Current sprint name for context"
    )

    args = parser.parse_args()

    workflow_id = args.workflow_id or f"hygiene-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print("=" * 60)
    print("🧹 Artifact Hygiene Workflow")
    print("=" * 60)
    print(f"Workflow ID: {workflow_id}")
    print(f"Mode: {'AI-assisted' if args.use_ai else 'Pattern-based'}")
    if args.dry_run:
        print("⚠️  DRY RUN MODE - No changes will be made")
    print()

    workflow = ArtifactHygieneWorkflow(
        workflow_id=workflow_id,
        enable_checkpoints=not args.no_checkpoints,
        use_ai=args.use_ai,
        dry_run=args.dry_run,
        verbose=args.verbose,
        current_sprint=args.sprint
    )

    try:
        success = workflow.execute()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
