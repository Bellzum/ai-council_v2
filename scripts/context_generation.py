#!/usr/bin/env python3
"""
AI-Driven Context Generation Workflow

Generates hierarchical CLAUDE.md and README.md documentation using AI to:
1. Identify which directories warrant documentation (not rule-based)
2. Generate contextually appropriate README.md for humans
3. Generate agent-focused CLAUDE.md with smart merge for existing files

This is NOT a template-based generator. The AI analyzes each directory's
actual code to produce relevant, specific documentation.

Requires: KEYCHAIN_ANTHROPIC_API_KEY environment variable

Usage:
    # Generate documentation for all significant directories
    python3 scripts/context_generation.py

    # Limit to specific directories
    python3 scripts/context_generation.py --target-dirs src tests docs

    # Skip README generation (existing files are never overwritten anyway)
    python3 scripts/context_generation.py --skip-readme

    # Only regenerate index (skip all content generation)
    python3 scripts/context_generation.py --skip-readme --skip-claude-md

    # Limit directory depth
    python3 scripts/context_generation.py --max-depth 2
"""

import argparse
import os
import re
import subprocess
import sys
import yaml
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import workflow executor base
from scripts.workflow_executor.base import WorkflowOrchestrator, ExecutionMode


# Skip directories (don't generate docs for these)
DEFAULT_SKIP_DIRS = {
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    "dist",
    "build",
    "out",
    "target",
    "bin",
    "obj",
    ".idea",
    ".vscode",
    ".vs",
    "coverage",
    ".coverage",
    "htmlcov",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

# Code file extensions
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".cs",
    ".sh",
    ".bash",
    ".ps1",
}


@dataclass
class ClaudeMdSection:
    """Represents a section in a CLAUDE.md file."""

    header: str  # e.g., "## Purpose"
    level: int  # 1 for #, 2 for ##, etc.
    content: str  # Content after header, before next header
    is_managed: bool = False  # True if this section should be regenerated


class ClaudeMdMerger:
    """
    Smart merge strategy for CLAUDE.md files.

    Section Categories:
    1. REGENERATED (always updated from code analysis):
       - Front matter (keywords, children, dependencies)
       - ## Related (auto-generated from dependencies)

    2. STRUCTURE-AWARE (detect file additions/removals):
       - ## Key * sections (Key Components, Key Agents, etc.)
       - Added files get placeholder entries
       - Removed files get entries removed
       - Unchanged files preserve existing descriptions

    3. PRESERVED (keep user/agent edits):
       - ## Purpose, ## Architecture, ## Usage, ## Important Notes
       - ## Real Failure Scenarios Prevented
       - Any custom sections not in known list
    """

    # Sections that are always regenerated from structure
    REGENERATED_SECTIONS = {"## Related"}

    # Sections where we merge based on file structure
    STRUCTURE_AWARE_PREFIXES = ("## Key ",)  # Key Components, Key Agents, etc.

    # Known preserved sections (user edits respected)
    PRESERVED_SECTIONS = {
        "## Purpose",
        "## Architecture",
        "## Usage",
        "## Important Notes",
        "## Real Failure Scenarios Prevented",
        "## Verification Pattern",
        "## Constraints for AI Agents",
        "## Quick Reference",
        "## Execution Pattern",
    }

    def __init__(self, directory: Path, skip_dirs: Set[str] = None):
        self.directory = directory
        self.skip_dirs = skip_dirs or DEFAULT_SKIP_DIRS

    def parse_sections(self, content: str) -> Tuple[str, str, List[ClaudeMdSection]]:
        """
        Parse CLAUDE.md into front matter, title, and sections.

        Only ## headers are treated as section boundaries.
        ### and lower are included in section content.

        Returns:
            (front_matter, title_line, sections)
        """
        # Extract front matter
        front_matter = ""
        body = content
        fm_match = re.match(r"^(---\n.*?\n---\n)", content, re.DOTALL)
        if fm_match:
            front_matter = fm_match.group(1)
            body = content[fm_match.end() :]

        # Extract H1 title
        title_line = ""
        title_match = re.match(r"^(# .+?\n)", body)
        if title_match:
            title_line = title_match.group(1)
            body = body[title_match.end() :]

        # Parse sections by ## headers ONLY (not ### or lower)
        # This keeps subsections as part of their parent section
        sections = []
        section_pattern = r"^(##)\s+(.+?)$"  # Only match exactly ##

        lines = body.split("\n")
        current_header = None
        current_level = 2  # ## level
        current_content = []

        for line in lines:
            header_match = re.match(section_pattern, line)
            if header_match:
                # Save previous section
                if current_header:
                    sections.append(
                        ClaudeMdSection(
                            header=current_header,
                            level=current_level,
                            content="\n".join(current_content).strip(),
                            is_managed=self._is_managed_section(current_header),
                        )
                    )

                # Start new section
                current_header = f"## {header_match.group(2)}"
                current_content = []
            else:
                current_content.append(line)

        # Don't forget the last section
        if current_header:
            sections.append(
                ClaudeMdSection(
                    header=current_header,
                    level=current_level,
                    content="\n".join(current_content).strip(),
                    is_managed=self._is_managed_section(current_header),
                )
            )

        return front_matter, title_line, sections

    def _is_managed_section(self, header: str) -> bool:
        """Check if a section is managed (regenerated) or preserved."""
        if header in self.REGENERATED_SECTIONS:
            return True
        for prefix in self.STRUCTURE_AWARE_PREFIXES:
            if header.startswith(prefix):
                return True
        return False

    def _is_structure_aware_section(self, header: str) -> bool:
        """Check if section needs structure-aware merging."""
        for prefix in self.STRUCTURE_AWARE_PREFIXES:
            if header.startswith(prefix):
                return True
        return False

    def get_current_files(self) -> Set[str]:
        """Get current files in the directory."""
        files = set()
        if not self.directory.exists():
            return files

        for item in self.directory.iterdir():
            if item.is_file() and item.suffix in CODE_EXTENSIONS:
                files.add(item.name)
            elif item.is_dir() and item.name not in self.skip_dirs:
                # Include subdirectories as potential components
                files.add(item.name)

        return files

    def get_git_status(self) -> Dict[str, str]:
        """
        Get git status for files in this directory.

        Returns dict: {filename: status} where status is 'A', 'M', 'D', '?', etc.
        """
        status = {}
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", str(self.directory)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        st = line[:2].strip()
                        filepath = line[3:].strip()
                        filename = Path(filepath).name
                        status[filename] = st
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return status

    def merge_key_section(
        self,
        existing_section: Optional[ClaudeMdSection],
        new_files: Set[str],
        removed_files: Set[str],
    ) -> str:
        """
        Merge a ## Key * section with structure changes.

        - Preserve descriptions for unchanged files
        - Add placeholder for new files
        - Remove entries for deleted files
        """
        if not existing_section:
            # Generate new section from scratch
            return self._generate_key_section_content(new_files)

        content = existing_section.content
        lines = content.split("\n")
        result_lines = []
        seen_files = set()

        # Pattern to match component entries like "### filename" or "- **filename**:"
        component_pattern = r"^###\s+(\S+)|^-\s+\*\*(\S+)\*\*"

        i = 0
        while i < len(lines):
            line = lines[i]
            match = re.match(component_pattern, line)

            if match:
                filename = match.group(1) or match.group(2)
                seen_files.add(filename)

                if filename in removed_files:
                    # Skip this entry and its content until next component
                    i += 1
                    while i < len(lines) and not re.match(component_pattern, lines[i]):
                        if lines[i].startswith("## ") or lines[i].startswith("# "):
                            break
                        i += 1
                    continue
                else:
                    # Keep the entry
                    result_lines.append(line)
            else:
                result_lines.append(line)
            i += 1

        # Add new files that weren't in the existing content
        added_files = new_files - seen_files
        for filename in sorted(added_files):
            result_lines.append(f"\n### {filename}")
            result_lines.append(f"*New component - description pending*\n")

        return "\n".join(result_lines)

    def _generate_key_section_content(self, files: Set[str]) -> str:
        """Generate content for a new Key section."""
        lines = []
        for filename in sorted(files):
            lines.append(f"### {filename}")
            lines.append("*Description pending*\n")
        return "\n".join(lines)

    def merge(
        self,
        existing_content: str,
        new_front_matter: str,
        new_related_section: Optional[str] = None,
    ) -> str:
        """
        Perform smart merge of CLAUDE.md content.

        Args:
            existing_content: Current CLAUDE.md content
            new_front_matter: Regenerated front matter from code analysis
            new_related_section: Regenerated ## Related section (optional)

        Returns:
            Merged CLAUDE.md content
        """
        _, title_line, existing_sections = self.parse_sections(existing_content)

        # Get current files for structure-aware merge
        current_files = self.get_current_files()

        # Build result
        result_parts = [new_front_matter]

        if title_line:
            result_parts.append(title_line)

        # Track which sections we've processed
        processed_headers = set()

        for section in existing_sections:
            header = section.header

            if header in self.REGENERATED_SECTIONS:
                # Use new generated content
                if header == "## Related" and new_related_section:
                    result_parts.append(f"{header}\n\n{new_related_section}\n")
                    processed_headers.add(header)
                continue

            if self._is_structure_aware_section(header):
                # Smart merge based on file structure
                # For now, detect files that exist vs don't exist
                existing_files = self._extract_component_names(section.content)
                removed = existing_files - current_files
                added = current_files - existing_files

                if removed or added:
                    merged_content = self.merge_key_section(section, added, removed)
                    result_parts.append(f"{header}\n\n{merged_content}\n")
                else:
                    # No structural changes, preserve as-is
                    result_parts.append(f"{header}\n\n{section.content}\n")
                processed_headers.add(header)
                continue

            # Preserved section - keep as-is
            result_parts.append(f"{header}\n\n{section.content}\n")
            processed_headers.add(header)

        # Add ## Related if it wasn't in existing content but we have new content
        if "## Related" not in processed_headers and new_related_section:
            result_parts.append(f"## Related\n\n{new_related_section}\n")

        return "\n".join(result_parts)

    def _extract_component_names(self, content: str) -> Set[str]:
        """Extract component/file names from a Key section."""
        names = set()
        # Match "### filename" or "- **filename**:" patterns
        for match in re.finditer(
            r"^###\s+(\S+)|^-\s+\*\*(\S+)\*\*", content, re.MULTILINE
        ):
            name = match.group(1) or match.group(2)
            if name:
                names.add(name)
        return names


class ContextGenerationWorkflow(WorkflowOrchestrator):
    """
    Context Generation workflow with external enforcement.

    Generates hierarchical documentation:
    - README.md files (human-readable)
    - CLAUDE.md files (AI context loading with YAML front matter)
    """

    def __init__(
        self,
        target_dirs: Optional[List[str]] = None,
        skip_readme: bool = False,
        skip_claude_md: bool = False,
        max_depth: int = 3,
        max_fix_retries: int = 2,
        args: Optional[argparse.Namespace] = None,
    ):
        """
        Initialize context generation workflow.

        AI is integral to this workflow - it decides which directories
        warrant documentation and generates contextually appropriate content.

        Args:
            target_dirs: List of target directories (default: scan all)
            skip_readme: Skip README.md generation for dirs that already have it
            skip_claude_md: Skip CLAUDE.md generation (only regenerate index)
            max_depth: Maximum directory depth to scan
            max_fix_retries: Max attempts for AI to fix broken refs/empty files (default: 2)
            args: Command-line arguments
        """
        self.target_dirs = target_dirs or []
        self.skip_readme = skip_readme
        self.skip_claude_md = skip_claude_md
        self.max_depth = max_depth
        self.max_fix_retries = max_fix_retries
        self.args = args

        # Project root
        self.root = Path.cwd()

        # Directories to skip (well-known non-documentable)
        self.skip_dirs = DEFAULT_SKIP_DIRS.copy()

        # Initialize Claude API client (required for this workflow)
        self.claude_client = None
        try:
            import anthropic

            api_key = os.getenv("KEYCHAIN_ANTHROPIC_API_KEY")
            if api_key:
                self.claude_client = anthropic.Anthropic(api_key=api_key)
                print("✅ Claude API client initialized")
            else:
                raise ValueError(
                    "KEYCHAIN_ANTHROPIC_API_KEY environment variable is required."
                    "Context generation uses AI to determine which directories "
                    "warrant documentation and to generate appropriate content."
                )
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for context generation. "
                "Install with: pip install anthropic"
            )

        workflow_id = f"context-gen-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        super().__init__(
            workflow_name="context-generation",
            workflow_id=workflow_id,
            mode=ExecutionMode.AI_JSON_VALIDATION,
            enable_checkpoints=True,
        )

    def _define_steps(self) -> List[Dict[str, Any]]:
        """
        Define workflow steps.

        The workflow uses AI at key decision points:
        - Step 1: Scan directories (Mode 1 - pure Python)
        - Step 2: AI identifies which directories warrant documentation (Mode 2)
        - Step 3: Generate README.md for documentable directories (Mode 2 - AI)
        - Step 4: Generate/merge CLAUDE.md files (Mode 2 - AI + smart merge)
        - Step 5: Verify generated context
        - Step 6: Generate context index
        """
        steps = [
            {"id": "1-scan-directories", "name": "Scan Repository Structure"},
            {
                "id": "2-identify-documentable",
                "name": "AI: Identify Documentable Directories",
            },
        ]

        if not self.skip_readme:
            steps.append(
                {"id": "3-generate-readme", "name": "AI: Generate README.md Files"}
            )

        if not self.skip_claude_md:
            steps.append(
                {
                    "id": "4-generate-claude-md",
                    "name": "AI: Generate/Merge CLAUDE.md Files",
                }
            )

        steps.append({"id": "5-verify-context", "name": "Verify Generated Context"})
        steps.append({"id": "6-generate-index", "name": "Generate Context Index"})

        return steps

    def _execute_step(
        self, step: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step_id = step["id"]

        if step_id == "1-scan-directories":
            return self._step_1_scan_directories()
        elif step_id == "2-identify-documentable":
            return self._step_2_identify_documentable()
        elif step_id == "3-generate-readme":
            return self._step_3_generate_readme()
        elif step_id == "4-generate-claude-md":
            return self._step_4_generate_claude_md()
        elif step_id == "5-verify-context":
            return self._step_5_verify_context()
        elif step_id == "6-generate-index":
            return self._step_6_generate_index()
        else:
            raise ValueError(f"Unknown step: {step_id}")

    def _get_agent_sdk_wrapper(self):
        """Get or create Agent SDK wrapper for AI calls with codebase access."""
        if not hasattr(self, '_agent_sdk_wrapper') or self._agent_sdk_wrapper is None:
            try:
                from scripts.workflow_executor.agent_sdk import AgentSDKWrapper

                self._agent_sdk_wrapper = AgentSDKWrapper(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    working_directory=str(self.root),
                    tool_preset="implementation"  # Needs write access to generate CLAUDE.md and README.md files
                )
                return self._agent_sdk_wrapper
            except ImportError as e:
                # ALWAYS report import errors - no silent fallbacks
                from cli.console import print_error, print_info
                print_error(f"Agent SDK Import Failed: {e}")
                print_error("The Claude Agent SDK is not available. Context generation requires tool access to explore the codebase.")
                print_info("Install with: pip install claude-code-sdk")
                return None
            except Exception as e:
                from cli.console import print_error
                print_error(f"Agent SDK Initialization Failed: {e}")
                return None
        return self._agent_sdk_wrapper

    def _invoke_ai(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> str:
        """
        Invoke Claude API for AI-driven decisions.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_tokens: Maximum tokens in response

        Returns:
            The AI response text
        """
        # Try Agent SDK first (provides codebase access for context)
        sdk = self._get_agent_sdk_wrapper()
        if sdk:
            try:
                result = sdk.ask(
                    prompt=prompt,
                    system_prompt=system or "You are a documentation expert generating CLAUDE.md and README.md files."
                )
                return result.response
            except Exception as e:
                # ALWAYS report errors - no silent fallbacks to Anthropic API
                from cli.console import print_error, print_info
                print_error(f"Agent SDK Execution Failed: {e}")
                print_error("Context generation requires codebase access to analyze project structure and code.")
                print_info("This step will be skipped. Fix the Agent SDK issue to enable AI-assisted context generation.")
                raise RuntimeError(f"Agent SDK execution failed: {e}")

        # No Agent SDK available - report and fail (don't fall back to degraded functionality)
        from cli.console import print_error, print_info
        print_error("Context Generation Skipped: No Agent SDK available")
        print_error("Context generation requires codebase access. Cannot proceed without Agent SDK.")
        print_info("Install with: pip install claude-code-sdk")
        raise RuntimeError("Agent SDK not available - cannot generate context without codebase access")

    def _invoke_ai_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Invoke Claude API and parse JSON response.

        Args:
            prompt: The user prompt (should request JSON output)
            system: Optional system prompt
            max_tokens: Maximum tokens in response

        Returns:
            Parsed JSON as dictionary
        """
        import json

        response = self._invoke_ai(prompt, system, max_tokens)

        # Extract JSON from response (handle markdown code blocks)
        json_text = response
        if "```json" in response:
            match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                json_text = match.group(1)
        elif "```" in response:
            match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                json_text = match.group(1)

        return json.loads(json_text)

    def _strip_markdown_fences(self, content: str) -> str:
        """
        Strip markdown code fences from AI-generated content.

        AI sometimes wraps markdown content in code fences like:
        ```markdown
        ---
        keywords: ...
        ---
        # Title
        ```

        This strips those fences to get the raw markdown content.
        """
        if not content:
            return content

        content = content.strip()

        # Check for ```markdown or ``` at the start
        if content.startswith("```"):
            # Find the end of the opening fence line
            first_newline = content.find("\n")
            if first_newline == -1:
                return content  # Malformed, return as-is

            # Check if it ends with closing fence
            if content.rstrip().endswith("```"):
                # Strip opening fence line and closing fence
                inner = content[first_newline + 1 :].rstrip()
                if inner.endswith("```"):
                    inner = inner[:-3].rstrip()
                return inner

        return content

    def _analyze_directory(self, dir_path: Path) -> Dict[str, Any]:
        """Analyze a directory's contents."""
        files = []
        subdirs = []

        try:
            for item in dir_path.iterdir():
                # Skip hidden files (except .claude)
                if item.name.startswith(".") and item.name != ".claude":
                    continue

                # Skip ignored directories
                if item.name in self.skip_dirs:
                    continue

                if item.is_dir():
                    subdirs.append(item.name)
                elif item.is_file():
                    files.append(
                        {
                            "name": item.name,
                            "extension": item.suffix,
                            "size": item.stat().st_size,
                        }
                    )
        except PermissionError:
            pass  # Skip directories we can't read

        # Detect directory type
        dir_name = dir_path.name.lower()

        type_patterns = {
            "src": "source",
            "lib": "source",
            "app": "source",
            "pkg": "source",
            "tests": "tests",
            "test": "tests",
            "spec": "tests",
            "__tests__": "tests",
            "docs": "documentation",
            "documentation": "documentation",
            "api": "api",
            "apis": "api",
            "core": "core",
            "config": "configuration",
            "configs": "configuration",
            "scripts": "scripts",
            "bin": "scripts",
            ".claude": "claude_config",
        }

        dir_type = type_patterns.get(dir_name, "module")

        # Count file types
        py_files = len([f for f in files if f["extension"] == ".py"])
        js_files = len(
            [f for f in files if f["extension"] in [".js", ".ts", ".tsx", ".jsx"]]
        )
        go_files = len([f for f in files if f["extension"] == ".go"])

        primary_lang = "mixed"
        if py_files > js_files and py_files > go_files:
            primary_lang = "python"
        elif js_files > py_files and js_files > go_files:
            primary_lang = "javascript"
        elif go_files > 0:
            primary_lang = "go"

        relative_path = str(dir_path.relative_to(self.root))
        if relative_path == ".":
            relative_path = "root"

        return {
            "path": str(dir_path),
            "relative_path": relative_path,
            "type": dir_type,
            "files": files,
            "subdirs": subdirs,
            "primary_language": primary_lang,
            "has_readme": (dir_path / "README.md").exists(),
            "has_claude_md": (dir_path / "CLAUDE.md").exists(),
        }

    def _step_1_scan_directories(self) -> Dict[str, Any]:
        """
        Step 1: Scan all directories in repository (Mode 1 - Pure Python).

        Gathers directory metadata without filtering. AI will decide
        which directories warrant documentation in Step 2.
        """
        print("\n📁 Scanning repository structure...")

        all_directories = []

        # Always include root
        all_directories.append(self._analyze_directory(self.root))

        # Scan all subdirectories
        for item in self.root.rglob("*"):
            if not item.is_dir():
                continue

            try:
                relative = item.relative_to(self.root)
            except ValueError:
                continue

            # Check depth
            if len(relative.parts) > self.max_depth:
                continue

            # Skip well-known non-documentable directories
            if any(skip in relative.parts for skip in self.skip_dirs):
                continue

            # If target_dirs specified, only include those
            if self.target_dirs:
                if not any(
                    str(relative).startswith(target) for target in self.target_dirs
                ):
                    continue

            all_directories.append(self._analyze_directory(item))

        print(f"✓ Scanned {len(all_directories)} directories")

        return {
            "all_directories": all_directories,
            "count": len(all_directories),
        }

    def _step_2_identify_documentable(self) -> Dict[str, Any]:
        """
        Step 2: AI identifies which directories warrant documentation (Mode 2).

        The AI evaluates each directory based on:
        - Architectural significance (core modules, entry points)
        - Code complexity (non-trivial logic vs simple utilities)
        - Cohesion (logical unit vs random files)
        - Audience (would an AI agent need context for this?)
        """
        print("\n🤖 AI: Identifying documentable directories...")

        all_directories = self.step_evidence.get("1-scan-directories", {}).get(
            "all_directories", []
        )

        if not all_directories:
            print("  ⚠ No directories to analyze")
            return {"documentable": [], "count": 0}

        # Build directory summary for AI
        dir_summary = []
        for d in all_directories:
            rel_path = d["relative_path"]
            files = d.get("files", [])
            subdirs = d.get("subdirs", [])

            code_files = [
                f["name"] for f in files if f.get("extension") in CODE_EXTENSIONS
            ]
            other_files = [
                f["name"] for f in files if f.get("extension") not in CODE_EXTENSIONS
            ]

            dir_summary.append(
                {
                    "path": rel_path,
                    "type": d.get("type", "unknown"),
                    "code_files": code_files[:20],  # Limit for token efficiency
                    "other_files": other_files[:10],
                    "subdirs": subdirs[:10],
                    "has_readme": d.get("has_readme", False),
                    "has_claude_md": d.get("has_claude_md", False),
                    "primary_language": d.get("primary_language", "unknown"),
                }
            )

        # AI prompt
        system_prompt = """You are analyzing a software project to determine which directories
warrant documentation for AI coding agents.

Your task: Identify directories that should have CLAUDE.md (agent context) and README.md
(human documentation) files.

INCLUDE directories that are:
- Architecturally significant (core modules, main entry points, key subsystems)
- Contain non-trivial code logic that agents would need context for
- Represent cohesive functional units (not random utility collections)
- Would benefit AI agents working on the codebase

EXCLUDE directories that are:
- Asset folders (images, fonts, static resources)
- Generated/vendored code
- Trivial utility folders with 1-2 simple files
- Deep nested directories with no standalone significance

For directories with existing CLAUDE.md, include them for potential updates.

Respond with JSON only."""

        prompt = f"""Analyze this project structure and identify which directories warrant documentation.

Project directories:
```json
{yaml.dump(dir_summary, default_flow_style=False)}
```

Return a JSON object with this structure:
{{
    "documentable": [
        {{
            "path": "relative/path",
            "reason": "Brief reason why this directory warrants documentation",
            "priority": "high" | "medium" | "low"
        }}
    ],
    "excluded": [
        {{
            "path": "relative/path",
            "reason": "Brief reason for exclusion"
        }}
    ]
}}"""

        try:
            result = self._invoke_ai_json(prompt, system_prompt)
            documentable = result.get("documentable", [])
            excluded = result.get("excluded", [])

            print(f"  ✓ AI identified {len(documentable)} documentable directories")

            # Map back to full directory info
            documentable_paths = {d["path"] for d in documentable}
            documentable_dirs = []

            for d in all_directories:
                if d["relative_path"] in documentable_paths:
                    # Add AI reasoning
                    for doc in documentable:
                        if doc["path"] == d["relative_path"]:
                            d["ai_reason"] = doc.get("reason", "")
                            d["ai_priority"] = doc.get("priority", "medium")
                            break
                    documentable_dirs.append(d)

            # Print results
            for d in documentable_dirs:
                status = "✓ has docs" if d["has_claude_md"] else "○ needs docs"
                priority = d.get("ai_priority", "medium")
                rel_path = d["relative_path"] if d["relative_path"] != "root" else "."
                print(f"  [{priority}] {rel_path:<25} {status}")

            if excluded:
                print(
                    f"  ⏭ Excluded {len(excluded)} directories (not architecturally significant)"
                )

            return {
                "documentable": documentable_dirs,
                "excluded": excluded,
                "count": len(documentable_dirs),
            }

        except Exception as e:
            print(f"  ⚠ AI analysis failed: {e}")
            print("  ↩ Falling back to heuristic-based selection")

            # Fallback: use simple heuristics
            documentable_dirs = []
            for d in all_directories:
                files = d.get("files", [])
                code_files = [f for f in files if f.get("extension") in CODE_EXTENSIONS]

                # Include if: has 2+ code files OR has existing CLAUDE.md OR is a known structural dir
                if (
                    len(code_files) >= 2
                    or d.get("has_claude_md")
                    or d.get("type")
                    in ("core", "source", "tests", "api", "configuration")
                ):
                    d["ai_reason"] = "Heuristic: significant code or structure"
                    d["ai_priority"] = "medium"
                    documentable_dirs.append(d)

            return {
                "documentable": documentable_dirs,
                "excluded": [],
                "count": len(documentable_dirs),
            }

    def _generate_readme_content(self, dir_info: Dict[str, Any]) -> str:
        """Generate README.md content based on directory analysis."""
        dir_name = (
            Path(dir_info["relative_path"]).name
            if dir_info["relative_path"] != "root"
            else "Project"
        )
        files = dir_info["files"]
        subdirs = dir_info["subdirs"]
        dir_type = dir_info["type"]

        # Purpose based on directory type
        purposes = {
            "root": f"{dir_name} project root directory.",
            "source": "Contains the main source code implementation.",
            "tests": "Contains the test suite for quality assurance.",
            "api": "Contains API definitions, endpoints, and request handlers.",
            "documentation": "Contains project documentation and guides.",
            "configuration": "Contains configuration files and settings.",
            "scripts": "Contains utility and automation scripts.",
            "core": "Contains core framework functionality and base implementations.",
            "claude_config": "Contains Claude Code configuration and workflow state.",
            "module": f"Contains the {dir_name} module implementation.",
        }
        purpose = purposes.get(
            dir_type, f"Contains {dir_name} related code and resources."
        )

        # Build README content
        content = f"# {dir_name}\n\n"
        content += f"## Purpose\n\n{purpose}\n\n"

        # Key components - analyze actual files
        key_components = []
        for f in files:
            fname = f["name"]
            if fname == "__init__.py":
                continue
            elif fname.startswith("test_"):
                key_components.append(f"**{fname}**: Test file")
            elif f["extension"] == ".py":
                # Read first few lines to get docstring
                file_path = Path(dir_info["path"]) / fname
                try:
                    with open(file_path, "r", encoding="utf-8") as fp:
                        lines = fp.readlines()[:10]
                        # Look for docstring
                        for line in lines:
                            if '"""' in line or "'''" in line:
                                doc = line.strip('"""').strip("'''").strip()
                                if doc:
                                    key_components.append(f"**{fname}**: {doc}")
                                    break
                        else:
                            # No docstring found
                            base = fname.replace(".py", "").replace("_", " ").title()
                            key_components.append(f"**{fname}**: {base} implementation")
                except Exception:
                    pass
            elif f["extension"] in [".js", ".ts", ".tsx", ".jsx"]:
                base = fname.split(".")[0].replace("_", " ").replace("-", " ").title()
                key_components.append(f"**{fname}**: {base} module")
            elif fname in [
                "package.json",
                "pyproject.toml",
                "setup.py",
                "requirements.txt",
            ]:
                key_components.append(f"**{fname}**: Package configuration")
            elif fname == "Dockerfile":
                key_components.append(f"**{fname}**: Container configuration")

        if key_components:
            content += "## Key Components\n\n"
            for comp in key_components[:10]:
                content += f"- {comp}\n"
            content += "\n"

        # Subdirectories
        if subdirs:
            content += "## Subdirectories\n\n"
            for d in subdirs[:10]:
                content += f"- **{d}/**\n"
            content += "\n"

        # Structure
        structure_lines = []
        for d in subdirs[:10]:
            structure_lines.append(f"{d}/")
        for f in [fi["name"] for fi in files[:15]]:
            if f not in ["__init__.py", ".gitignore"]:
                structure_lines.append(f)

        if structure_lines:
            content += "## Structure\n\n```\n"
            content += "\n".join(structure_lines)
            content += "\n```\n"

        return content

    def _step_3_generate_readme(self) -> Dict[str, Any]:
        """
        Step 3: Generate README.md files using AI (Mode 2).

        AI generates comprehensive human-readable documentation for each
        documentable directory. Existing README.md files are skipped.
        """
        print("\n📄 AI: Generating README.md files...")

        directories = self.step_evidence.get("2-identify-documentable", {}).get(
            "documentable", []
        )

        generated = 0
        skipped = 0
        errors = 0

        for dir_info in directories:
            dir_path = Path(dir_info["path"])
            readme_path = dir_path / "README.md"
            rel_path = dir_info["relative_path"]

            if readme_path.exists():
                if self.skip_readme:
                    print(f"  ⏭ {rel_path}/README.md (exists)")
                else:
                    print(f"  ⏭ {rel_path}/README.md (exists, not overwriting)")
                skipped += 1
                continue

            # Generate README using AI
            try:
                content = self._generate_readme_with_ai(dir_info)

                if content and content.strip():
                    with open(readme_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"  ✓ {rel_path}/README.md")
                    generated += 1
                else:
                    print(f"  ⚠ {rel_path}/README.md (AI returned empty)")
                    skipped += 1

            except Exception as e:
                print(f"  ✗ {rel_path}/README.md (error: {e})")
                errors += 1

        print(
            f"\n✓ Generated {generated} README.md files ({skipped} skipped, {errors} errors)"
        )

        return {"generated": generated, "skipped": skipped, "errors": errors}

    def _generate_readme_with_ai(self, dir_info: Dict[str, Any]) -> str:
        """
        Generate README.md content using AI.

        The AI creates comprehensive human-readable documentation based on
        the directory's contents, purpose, and role in the project.
        """
        rel_path = dir_info["relative_path"]
        files = dir_info.get("files", [])
        subdirs = dir_info.get("subdirs", [])
        dir_type = dir_info.get("type", "module")
        ai_reason = dir_info.get("ai_reason", "")

        # Get file contents summaries for context
        code_files = [f for f in files if f.get("extension") in CODE_EXTENSIONS]
        file_summaries = []

        for f in code_files[:10]:  # Limit to 10 files for token efficiency
            file_path = Path(dir_info["path"]) / f["name"]
            try:
                with open(file_path, "r", encoding="utf-8") as fp:
                    content = fp.read(2000)  # First 2000 chars
                    file_summaries.append(
                        {
                            "name": f["name"],
                            "preview": content[:500],  # Even shorter for prompt
                        }
                    )
            except Exception:
                file_summaries.append(
                    {"name": f["name"], "preview": "(could not read)"}
                )

        system_prompt = """You are a technical documentation writer creating README.md files.

Write clear, comprehensive documentation that helps developers understand:
- What this directory/module does
- How to use it
- Key files and their purposes
- Any important patterns or conventions

Use proper Markdown formatting with headers, code blocks, and lists.
Be concise but thorough. Focus on practical information developers need."""

        prompt = f"""Generate a README.md for this directory:

Directory: {rel_path}
Type: {dir_type}
Purpose: {ai_reason}

Files:
{yaml.dump(file_summaries, default_flow_style=False)}

Subdirectories: {', '.join(subdirs) if subdirs else 'None'}

Generate a complete README.md with:
1. Title (# Directory Name)
2. Overview/Purpose section
3. Key Components (describe main files)
4. Usage examples if applicable
5. Related sections or dependencies

Return only the Markdown content, no code fences around the entire response."""

        response = self._invoke_ai(prompt, system_prompt, max_tokens=2000)
        return self._strip_markdown_fences(response)

    def _generate_claude_md_content(
        self, dir_info: Dict[str, Any], all_dirs: List[Dict[str, Any]]
    ) -> str:
        """Generate CLAUDE.md with YAML front matter."""
        relative_path = dir_info["relative_path"]
        dir_type = dir_info["type"]

        # Keywords based on directory type
        type_keywords = {
            "root": ["project", "overview", "framework"],
            "source": ["source", "code", "implementation", "feature"],
            "tests": ["test", "testing", "pytest", "coverage", "fixture"],
            "api": ["api", "endpoint", "route", "handler", "rest"],
            "documentation": ["docs", "documentation", "guide"],
            "configuration": ["config", "configuration", "settings"],
            "scripts": ["script", "automation", "build"],
            "core": ["core", "foundation", "base"],
            "claude_config": ["claude", "runtime", "workflow", "agent"],
            "module": ["module", "feature"],
        }

        keywords = type_keywords.get(dir_type, ["module"]).copy()

        # Add directory name to keywords
        dir_name = Path(relative_path).name if relative_path != "root" else ""
        if dir_name and dir_name not in keywords:
            keywords.append(dir_name.lower().replace("_", "-").replace(".", "-"))

        # Task types based on directory type
        type_task_types = {
            "root": ["any"],
            "source": ["implementation", "feature-development", "bug-fix"],
            "tests": ["testing", "quality-assurance"],
            "api": ["api-development"],
            "documentation": ["documentation"],
            "configuration": ["configuration"],
            "scripts": ["scripting", "automation"],
            "core": ["implementation", "architecture"],
            "claude_config": ["workflow", "agent-development"],
        }

        task_types = type_task_types.get(dir_type, ["implementation"])

        # Priority based on directory type
        type_priority = {
            "root": "high",
            "source": "high",
            "core": "high",
            "api": "high",
            "tests": "medium",
            "configuration": "medium",
            "documentation": "low",
        }

        priority = type_priority.get(dir_type, "medium")

        # Max tokens based on directory type
        type_max_tokens = {
            "root": 1500,
            "source": 800,
            "core": 800,
            "api": 800,
            "tests": 600,
            "documentation": 400,
        }

        max_tokens = type_max_tokens.get(dir_type, 600)

        # Build children list
        children = []
        for d in all_dirs:
            d_path = d["relative_path"]
            if d_path == relative_path or d_path == "root":
                continue

            # Check if this is a direct child
            if relative_path == "root":
                # Root: check if d_path has no slashes (direct child)
                if "/" not in d_path and "\\" not in d_path:
                    child_name = d_path
                    children.append(child_name)
            else:
                # Check if d_path is direct child of relative_path
                parent_parts = Path(relative_path).parts
                child_parts = Path(d_path).parts

                if (
                    len(child_parts) == len(parent_parts) + 1
                    and child_parts[: len(parent_parts)] == parent_parts
                ):
                    child_name = child_parts[-1]
                    children.append(child_name)

        # Build front matter
        front_matter = "---\n"
        front_matter += "context:\n"
        front_matter += f"  keywords: [{', '.join(keywords[:8])}]\n"
        front_matter += f"  task_types: [{', '.join(task_types[:3])}]\n"
        front_matter += f"  priority: {priority}\n"
        front_matter += f"  max_tokens: {max_tokens}\n"

        if children:
            front_matter += f"  children: [{', '.join(children[:10])}]\n"
        else:
            front_matter += "  children: []\n"

        front_matter += "  dependencies: []\n"
        front_matter += "---\n"

        # Minimal body - real docs in README.md
        dir_name = Path(relative_path).name if relative_path != "root" else "Project"

        body = f"# {dir_name}\n\n"
        body += "## Purpose\n\n"
        body += "See [README.md](README.md) for full documentation.\n\n"

        return front_matter + body

    def _generate_related_section(
        self, dir_info: Dict[str, Any], all_dirs: List[Dict[str, Any]]
    ) -> str:
        """
        Generate ## Related section content from directory dependencies.

        Finds related CLAUDE.md files based on:
        - Parent directory
        - Sibling directories
        - Dependencies from front matter (if known)
        """
        relative_path = dir_info["relative_path"]
        related_links = []

        # Parent directory
        if relative_path != "root":
            parent_path = Path(relative_path).parent
            if parent_path != Path("."):
                parent_claude = f"{parent_path}/CLAUDE.md"
                related_links.append(f"- [{parent_path}/CLAUDE.md]({parent_claude})")
            else:
                related_links.append("- [CLAUDE.md](../CLAUDE.md) - Project root")

        # Sibling directories (same parent)
        if relative_path != "root":
            current_parent = Path(relative_path).parent
            for d in all_dirs:
                d_path = d["relative_path"]
                if d_path == relative_path or d_path == "root":
                    continue
                d_parent = Path(d_path).parent
                if d_parent == current_parent:
                    sibling_name = Path(d_path).name
                    related_links.append(
                        f"- [{sibling_name}/CLAUDE.md](../{sibling_name}/CLAUDE.md)"
                    )

        # Limit to 5 related links
        if len(related_links) > 5:
            related_links = related_links[:5]

        return "\n".join(related_links) if related_links else "*No related contexts*"

    def _merge_claude_md_content(
        self,
        existing_content: str,
        new_front_matter: str,
        directory: Path,
        new_related_section: Optional[str] = None,
    ) -> str:
        """
        Smart merge of CLAUDE.md content.

        Uses ClaudeMdMerger for intelligent section-based merging:
        - Front matter: Always regenerated from code analysis
        - ## Related: Always regenerated from dependencies
        - ## Key *: Structure-aware merge (add/remove based on files)
        - Other sections: Preserved (respect user/agent edits)

        Args:
            existing_content: Current CLAUDE.md content
            new_front_matter: Regenerated front matter from code analysis
            directory: Path to the directory (for file detection)
            new_related_section: Regenerated ## Related content (optional)

        Returns:
            Merged CLAUDE.md content
        """
        merger = ClaudeMdMerger(directory, self.skip_dirs)
        return merger.merge(existing_content, new_front_matter, new_related_section)

    def _step_4_generate_claude_md(self) -> Dict[str, Any]:
        """
        Step 4: Generate/merge CLAUDE.md files using AI (Mode 2).

        For new files: AI generates agent-focused documentation with front matter.
        For existing files: Smart merge preserves user edits while updating structure.
        """
        print("\n📄 AI: Generating/merging CLAUDE.md files...")

        directories = self.step_evidence.get("2-identify-documentable", {}).get(
            "documentable", []
        )

        generated = 0
        merged = 0
        skipped = 0
        errors = 0

        for dir_info in directories:
            dir_path = Path(dir_info["path"])
            claude_path = dir_path / "CLAUDE.md"
            rel_path = dir_info["relative_path"]

            if claude_path.exists():
                # Merge mode: AI updates + smart section-based merge
                try:
                    with open(claude_path, "r", encoding="utf-8") as f:
                        existing_content = f.read()

                    # AI generates new front matter based on current code
                    new_front_matter = self._generate_front_matter_with_ai(
                        dir_info, directories
                    )

                    # Generate new ## Related section
                    new_related = self._generate_related_section(dir_info, directories)

                    # Smart merge: preserves user edits, updates structure
                    merged_content = self._merge_claude_md_content(
                        existing_content, new_front_matter, dir_path, new_related
                    )

                    with open(claude_path, "w", encoding="utf-8") as f:
                        f.write(merged_content)
                    print(f"  🔄 {rel_path}/CLAUDE.md (merged)")
                    merged += 1

                except Exception as e:
                    print(f"  ✗ {rel_path}/CLAUDE.md (merge error: {e})")
                    errors += 1
            else:
                # Create new file using AI
                try:
                    content = self._generate_claude_md_with_ai(dir_info, directories)

                    if content and content.strip():
                        with open(claude_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        print(f"  ✓ {rel_path}/CLAUDE.md")
                        generated += 1
                    else:
                        print(f"  ⚠ {rel_path}/CLAUDE.md (AI returned empty)")
                        skipped += 1

                except Exception as e:
                    print(f"  ✗ {rel_path}/CLAUDE.md (error: {e})")
                    errors += 1

        print(
            f"\n✓ Generated {generated} CLAUDE.md files ({merged} merged, {skipped} skipped, {errors} errors)"
        )

        return {
            "generated": generated,
            "merged": merged,
            "skipped": skipped,
            "errors": errors,
        }

    def _generate_front_matter_with_ai(
        self, dir_info: Dict[str, Any], all_dirs: List[Dict[str, Any]]
    ) -> str:
        """
        Generate YAML front matter using AI.

        The AI analyzes the directory to determine appropriate:
        - Keywords for context matching
        - Task types this directory is relevant for
        - Priority level
        - Token budget
        - Child/dependency relationships
        """
        rel_path = dir_info["relative_path"]
        files = dir_info.get("files", [])
        subdirs = dir_info.get("subdirs", [])
        dir_type = dir_info.get("type", "module")
        ai_reason = dir_info.get("ai_reason", "")

        code_files = [f["name"] for f in files if f.get("extension") in CODE_EXTENSIONS]

        system_prompt = """You are generating YAML front matter for CLAUDE.md files.
This metadata helps AI agents find relevant context for their tasks.

Generate appropriate:
- keywords: 5-8 terms that would match task descriptions
- task_types: 2-4 types of tasks this code is relevant for
- priority: high (core/critical), medium (important), low (supplementary)
- max_tokens: estimated token budget (400-1000 based on complexity)
- children: subdirectories that should be loaded together
- dependencies: other directories this depends on

Respond with YAML front matter only, starting with --- and ending with ---"""

        prompt = f"""Generate YAML front matter for:

Directory: {rel_path}
Type: {dir_type}
Purpose: {ai_reason}
Code files: {', '.join(code_files[:15])}
Subdirectories: {', '.join(subdirs[:10])}

Return only the YAML front matter block."""

        response = self._invoke_ai(prompt, system_prompt, max_tokens=500)

        # Ensure it starts and ends with ---
        if not response.strip().startswith("---"):
            response = "---\n" + response
        if not response.strip().endswith("---"):
            response = response.rstrip() + "\n---\n"

        return response

    def _generate_claude_md_with_ai(
        self, dir_info: Dict[str, Any], all_dirs: List[Dict[str, Any]]
    ) -> str:
        """
        Generate complete CLAUDE.md content using AI.

        Creates agent-focused documentation with:
        - YAML front matter for context loading
        - Purpose section (brief, problem-focused)
        - Key Components (files with descriptions)
        - Reference to README.md for full documentation
        """
        rel_path = dir_info["relative_path"]
        files = dir_info.get("files", [])
        subdirs = dir_info.get("subdirs", [])
        dir_type = dir_info.get("type", "module")
        ai_reason = dir_info.get("ai_reason", "")

        # Get file content previews for context
        code_files = [f for f in files if f.get("extension") in CODE_EXTENSIONS]
        file_previews = []

        for f in code_files[:8]:  # Limit for token efficiency
            file_path = Path(dir_info["path"]) / f["name"]
            try:
                with open(file_path, "r", encoding="utf-8") as fp:
                    content = fp.read(1500)
                    file_previews.append(
                        {
                            "name": f["name"],
                            "preview": content[:400],
                        }
                    )
            except Exception:
                file_previews.append({"name": f["name"], "preview": "(could not read)"})

        system_prompt = """You are creating CLAUDE.md files for AI coding agents.

CLAUDE.md is NOT a comprehensive README - it's a focused summary to help AI agents
quickly understand what this directory does and when they need its context.

Structure:
1. YAML front matter (keywords, task_types, priority, max_tokens, children, dependencies)
2. # Title
3. ## Purpose - 2-3 sentences on what problem this solves. Reference README.md for details.
4. ## Key Components - Brief description of each important file/module
5. ## Constraints for AI Agents - Critical rules or patterns agents must follow
6. ## Related - Links to related CLAUDE.md files

Keep it concise. Total should be 100-200 lines. Details go in README.md."""

        # Check if README exists for reference
        readme_exists = (Path(dir_info["path"]) / "README.md").exists()
        readme_note = (
            "See [README.md](README.md) for full documentation."
            if readme_exists
            else ""
        )

        prompt = f"""Generate CLAUDE.md for:

Directory: {rel_path}
Type: {dir_type}
Purpose: {ai_reason}
README exists: {readme_exists}

Files:
{yaml.dump(file_previews, default_flow_style=False)}

Subdirectories: {', '.join(subdirs) if subdirs else 'None'}

{f'Include this in Purpose section: {readme_note}' if readme_note else 'Note: No README.md exists yet.'}

Generate complete CLAUDE.md content with front matter and all sections."""

        response = self._invoke_ai(prompt, system_prompt, max_tokens=2500)
        return self._strip_markdown_fences(response)

    def _detect_issues(
        self, directories: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """
        Detect issues in generated CLAUDE.md files.

        Returns:
            Tuple of (issues, warnings, passed_count)
            Each issue/warning is a dict with: path, relative_path, type, message
        """
        issues = []
        warnings = []
        passed = 0

        for dir_info in directories:
            dir_path = Path(dir_info["path"])
            claude_path = dir_path / "CLAUDE.md"
            relative = dir_info["relative_path"]

            # Rule 1: CLAUDE.md must exist
            if not claude_path.exists():
                issues.append(
                    {
                        "path": str(claude_path),
                        "relative_path": relative,
                        "type": "missing",
                        "message": "CLAUDE.md not created",
                    }
                )
                continue

            try:
                with open(claude_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Rule 2: No null or empty files
                if content is None:
                    issues.append(
                        {
                            "path": str(claude_path),
                            "relative_path": relative,
                            "type": "null",
                            "message": "File is null",
                        }
                    )
                    continue

                if not content.strip():
                    issues.append(
                        {
                            "path": str(claude_path),
                            "relative_path": relative,
                            "type": "empty",
                            "message": "File is empty",
                        }
                    )
                    continue

                if len(content.strip()) < 50:
                    issues.append(
                        {
                            "path": str(claude_path),
                            "relative_path": relative,
                            "type": "too_short",
                            "message": f"File too short ({len(content)} chars)",
                        }
                    )
                    continue

                # Rule 3: Front matter must exist and be valid
                if not content.startswith("---"):
                    issues.append(
                        {
                            "path": str(claude_path),
                            "relative_path": relative,
                            "type": "no_front_matter",
                            "message": "Missing front matter",
                        }
                    )
                    continue

                match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
                if not match:
                    issues.append(
                        {
                            "path": str(claude_path),
                            "relative_path": relative,
                            "type": "malformed_front_matter",
                            "message": "Malformed front matter",
                        }
                    )
                    continue

                # Validate YAML front matter
                try:
                    yaml.safe_load(match.group(1))
                except yaml.YAMLError as e:
                    issues.append(
                        {
                            "path": str(claude_path),
                            "relative_path": relative,
                            "type": "invalid_yaml",
                            "message": f"Invalid YAML: {e}",
                        }
                    )
                    continue

                # Rule 4: Check for broken references
                broken_refs = self._find_broken_references(content, dir_path)
                if broken_refs:
                    for ref in broken_refs:
                        warnings.append(
                            {
                                "path": str(claude_path),
                                "relative_path": relative,
                                "type": "broken_ref",
                                "message": f"Broken reference: {ref}",
                                "ref": ref,
                            }
                        )
                    # Still counts as passed if only broken refs (can be fixed)
                    passed += 1
                else:
                    passed += 1

            except Exception as e:
                issues.append(
                    {
                        "path": str(claude_path),
                        "relative_path": relative,
                        "type": "read_error",
                        "message": str(e),
                    }
                )

        return issues, warnings, passed

    def _find_broken_references(
        self, content: str, dir_path: Path
    ) -> List[str]:
        """
        Find broken file references in CLAUDE.md content.

        Returns list of broken reference paths.
        """
        broken_refs = []

        # Pattern to find markdown links: [text](path)
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"

        for match in re.finditer(link_pattern, content):
            link_path = match.group(2)

            # Skip external URLs
            if link_path.startswith(("http://", "https://", "#", "mailto:")):
                continue

            # Resolve the path relative to the directory
            if link_path.startswith("../"):
                ref_path = (dir_path / link_path).resolve()
            else:
                ref_path = dir_path / link_path

            if not ref_path.exists():
                broken_refs.append(link_path)

        return broken_refs

    def _prompt_ai_to_fix_issues(
        self,
        issues: List[Dict[str, Any]],
        warnings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Prompt AI to fix detected issues in CLAUDE.md files.

        Args:
            issues: List of critical issues (empty, missing, invalid)
            warnings: List of warnings (broken refs)

        Returns:
            Dict with fix_attempts and fixes_applied counts
        """
        fix_attempts = 0
        fixes_applied = 0

        # Group issues by directory for efficient fixing
        dirs_to_fix = {}
        for issue in issues:
            rel_path = issue["relative_path"]
            if rel_path not in dirs_to_fix:
                dirs_to_fix[rel_path] = {"issues": [], "warnings": []}
            dirs_to_fix[rel_path]["issues"].append(issue)

        for warning in warnings:
            rel_path = warning["relative_path"]
            if rel_path not in dirs_to_fix:
                dirs_to_fix[rel_path] = {"issues": [], "warnings": []}
            dirs_to_fix[rel_path]["warnings"].append(warning)

        for rel_path, problems in dirs_to_fix.items():
            fix_attempts += 1
            dir_issues = problems["issues"]
            dir_warnings = problems["warnings"]

            # Find the directory info from step evidence
            documentable = self.step_evidence.get("2-identify-documentable", {}).get(
                "documentable", []
            )
            dir_info = next(
                (d for d in documentable if d["relative_path"] == rel_path), None
            )

            if not dir_info:
                print(f"  ⚠️  Cannot find directory info for {rel_path}, skipping")
                continue

            dir_path = Path(dir_info["path"])

            # Build fix prompt
            issue_descriptions = []
            for issue in dir_issues:
                issue_descriptions.append(f"- {issue['type']}: {issue['message']}")
            for warning in dir_warnings:
                issue_descriptions.append(
                    f"- broken_reference: {warning.get('ref', warning['message'])}"
                )

            system_prompt = """You are fixing CLAUDE.md documentation issues.

Generate a complete, valid CLAUDE.md file that:
1. Has valid YAML front matter with context section
2. Is at least 100 characters of meaningful content
3. Only references files that actually exist
4. Follows the standard structure: Purpose, Key Components, Usage, Related

If a README.md reference is requested but doesn't exist, DO NOT include a "See README.md" line.
Only reference files that are confirmed to exist."""

            # Read current content if file exists
            claude_path = dir_path / "CLAUDE.md"
            current_content = ""
            if claude_path.exists():
                try:
                    with open(claude_path, "r", encoding="utf-8") as f:
                        current_content = f.read()
                except Exception:
                    pass

            # Get file list
            files = []
            try:
                for item in dir_path.iterdir():
                    if item.is_file() and not item.name.startswith("."):
                        files.append(item.name)
            except Exception:
                pass

            prompt = f"""Fix the CLAUDE.md for directory: {rel_path}

Issues found:
{chr(10).join(issue_descriptions)}

Files in directory: {', '.join(files) if files else 'None found'}

{"Current content (has problems):" + chr(10) + current_content[:1000] if current_content else "No current content - generate new file."}

Generate the complete fixed CLAUDE.md content. Start with --- for front matter."""

            try:
                raw_content = self._invoke_ai(prompt, system_prompt, max_tokens=2500)
                fixed_content = self._strip_markdown_fences(raw_content)

                if fixed_content and fixed_content.strip().startswith("---"):
                    # Validate the fix
                    if len(fixed_content.strip()) >= 100:
                        with open(claude_path, "w", encoding="utf-8") as f:
                            f.write(fixed_content)
                        print(f"  ✅ AI fixed: {rel_path}/CLAUDE.md")
                        fixes_applied += 1
                    else:
                        print(f"  ⚠️  AI fix too short for {rel_path}, will retry")
                else:
                    print(f"  ⚠️  AI fix invalid for {rel_path}, will retry")

            except Exception as e:
                print(f"  ⚠️  AI fix failed for {rel_path}: {e}")

        return {"fix_attempts": fix_attempts, "fixes_applied": fixes_applied}

    def _step_5_verify_context(self) -> Dict[str, Any]:
        """
        Step 5: Verify and enforce rules on generated context files.

        Enforces:
        1. No CLAUDE.md files are null or empty (would crash Claude Code)
        2. All referenced files in CLAUDE.md actually exist
        3. Front matter is valid YAML

        Fix strategy:
        1. Detect issues
        2. Prompt AI to fix issues (retry up to max_fix_retries times)
        3. Fall back to auto-cleanup of broken references only after retries exhausted
        """
        print("\n🔍 Verifying generated context files...")

        directories = self.step_evidence.get("2-identify-documentable", {}).get(
            "documentable", []
        )

        if not directories:
            print("  No directories to verify")
            return {
                "passed": 0,
                "warnings": 0,
                "errors": 0,
                "ai_fix_attempts": 0,
                "ai_fixes_applied": 0,
                "auto_fixes_applied": 0,
                "total": 0,
            }

        # Initial detection
        issues, warnings, passed = self._detect_issues(directories)

        total_ai_attempts = 0
        total_ai_fixes = 0

        # AI fix retry loop
        retry = 0
        while (issues or warnings) and retry < self.max_fix_retries:
            retry += 1
            print(f"\n🤖 AI fix attempt {retry}/{self.max_fix_retries}...")

            fix_result = self._prompt_ai_to_fix_issues(issues, warnings)
            total_ai_attempts += fix_result["fix_attempts"]
            total_ai_fixes += fix_result["fixes_applied"]

            if fix_result["fixes_applied"] > 0:
                # Re-detect issues after fixes
                issues, warnings, passed = self._detect_issues(directories)

                if not issues and not warnings:
                    print(f"  ✅ All issues resolved by AI")
                    break
            else:
                print(f"  ⚠️  No fixes applied in this attempt")
                break

        # Fall back to auto-cleanup for remaining broken references
        auto_fixes = 0
        if warnings:
            print(f"\n🔧 Applying auto-cleanup for remaining broken references...")
            for warning in warnings:
                if warning["type"] == "broken_ref":
                    dir_path = Path(warning["path"]).parent
                    claude_path = Path(warning["path"])

                    try:
                        with open(claude_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        broken_refs, fixed_content = self._check_and_fix_references(
                            content, dir_path, warning["relative_path"]
                        )

                        if broken_refs:
                            with open(claude_path, "w", encoding="utf-8") as f:
                                f.write(fixed_content)
                            auto_fixes += 1

                    except Exception as e:
                        print(f"  ⚠️  Auto-fix failed for {warning['relative_path']}: {e}")

            if auto_fixes:
                print(f"  ✅ Auto-cleaned {auto_fixes} files")

        # Final detection after all fixes
        final_issues, final_warnings, final_passed = self._detect_issues(directories)

        # Print summary
        print(f"\n{'='*60}")
        print("Verification Summary")
        print(f"{'='*60}")
        print(f"  ✅ Passed: {final_passed}/{len(directories)}")
        print(f"  ⚠️  Warnings: {len(final_warnings)}")
        print(f"  ❌ Errors: {len(final_issues)}")
        print(f"  🤖 AI fix attempts: {total_ai_attempts}")
        print(f"  🤖 AI fixes applied: {total_ai_fixes}")
        print(f"  🔧 Auto-fixes applied: {auto_fixes}")

        if final_warnings:
            print("\n⚠️  Remaining Warnings:")
            for warning in final_warnings:
                print(f"  - {warning['relative_path']}: {warning['message']}")

        if final_issues:
            print("\n❌ Remaining Errors:")
            for issue in final_issues:
                print(f"  - {issue['relative_path']}: {issue['message']}")

        if not final_issues and not final_warnings:
            print("\n🎉 All documentation files generated successfully!")

        return {
            "passed": final_passed,
            "warnings": len(final_warnings),
            "errors": len(final_issues),
            "ai_fix_attempts": total_ai_attempts,
            "ai_fixes_applied": total_ai_fixes,
            "auto_fixes_applied": auto_fixes,
            "total": len(directories),
            "warning_details": [w["message"] for w in final_warnings],
            "error_details": [i["message"] for i in final_issues],
        }

    def _check_and_fix_references(
        self, content: str, dir_path: Path, relative_path: str
    ) -> Tuple[List[str], str]:
        """
        Check for broken file references in CLAUDE.md and fix them.

        Finds markdown links like [README.md](README.md) and verifies
        the referenced files exist. If not, removes or comments out
        the broken reference.

        Args:
            content: The CLAUDE.md content
            dir_path: Path to the directory containing CLAUDE.md
            relative_path: Relative path for logging

        Returns:
            Tuple of (list of broken refs found, fixed content)
        """
        broken_refs = []
        fixed_content = content

        # Pattern to find markdown links: [text](path)
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"

        for match in re.finditer(link_pattern, content):
            link_text = match.group(1)
            link_path = match.group(2)

            # Skip external URLs
            if link_path.startswith(("http://", "https://", "#", "mailto:")):
                continue

            # Resolve the path relative to the directory
            if link_path.startswith("../"):
                # Parent directory reference
                ref_path = (dir_path / link_path).resolve()
            else:
                # Same directory reference
                ref_path = dir_path / link_path

            # Check if file exists
            if not ref_path.exists():
                broken_refs.append(link_path)

                # Fix: Remove the broken reference line or replace with comment
                if "README.md" in link_path:
                    # For README references, remove the "See README.md" line entirely
                    fixed_content = re.sub(
                        rf"See \[README\.md\]\([^)]*README\.md[^)]*\)[^\n]*\n?",
                        "",
                        fixed_content,
                    )
                elif "CLAUDE.md" in link_path:
                    # For CLAUDE.md references in Related section, remove the line
                    fixed_content = re.sub(
                        rf"-\s*\[[^\]]*\]\({re.escape(link_path)}\)[^\n]*\n?",
                        "",
                        fixed_content,
                    )
                else:
                    # For other broken links, comment them out
                    original_link = match.group(0)
                    fixed_content = fixed_content.replace(
                        original_link, f"<!-- BROKEN: {original_link} -->"
                    )

        return broken_refs, fixed_content

    def _parse_front_matter(self, claude_path: Path) -> Optional[Dict[str, Any]]:
        """Parse YAML front matter from CLAUDE.md file."""
        try:
            with open(claude_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.startswith("---"):
                return None

            match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
            if not match:
                return None

            front_matter = yaml.safe_load(match.group(1))
            return front_matter.get("context", front_matter)

        except Exception:
            return None

    def _extract_root_summary(self) -> Dict[str, Any]:
        """Extract summary from root CLAUDE.md for the index."""
        root_claude = self.root / "CLAUDE.md"
        if not root_claude.exists():
            return {
                "path": "CLAUDE.md",
                "summary": "Project root",
                "keywords": ["project", "overview"],
            }

        front_matter = self._parse_front_matter(root_claude)

        # Try to get purpose from front matter or first paragraph
        purpose = "Project root"
        if front_matter:
            purpose = front_matter.get("purpose", purpose)

        keywords = (
            front_matter.get("keywords", ["project", "overview"])
            if front_matter
            else ["project", "overview"]
        )

        return {
            "path": "CLAUDE.md",
            "summary": (
                purpose[:200] if isinstance(purpose, str) else str(purpose)[:200]
            ),
            "keywords": keywords[:10] if isinstance(keywords, list) else [],
        }

    def _find_readme(self, directory: Path) -> Optional[str]:
        """Find README.md relative path for a directory."""
        readme_path = directory / "README.md"
        if readme_path.exists():
            try:
                return str(readme_path.relative_to(self.root))
            except ValueError:
                return None
        return None

    def _step_6_generate_index(self) -> Dict[str, Any]:
        """Step 6: Generate context-index.yaml from all CLAUDE.md files."""
        print("\n📑 Generating context index...")

        index = {
            "generated_at": datetime.now().isoformat(),
            "version": "1.0",
            "root": self._extract_root_summary(),
            "contexts": [],
            "keyword_index": {},
        }

        keyword_to_paths = defaultdict(list)
        indexed_count = 0
        skipped_count = 0

        # Scan all CLAUDE.md files
        for claude_file in self.root.rglob("CLAUDE.md"):
            # Skip .git and other hidden directories
            if ".git" in claude_file.parts:
                continue

            # Skip node_modules, venv, etc.
            if any(skip in claude_file.parts for skip in self.skip_dirs):
                continue

            # Skip root CLAUDE.md (handled separately in 'root' section)
            if claude_file.parent == self.root:
                continue

            front_matter = self._parse_front_matter(claude_file)
            if not front_matter:
                skipped_count += 1
                continue

            try:
                relative_path = str(claude_file.relative_to(self.root))
            except ValueError:
                skipped_count += 1
                continue

            context_entry = {
                "path": relative_path,
                "keywords": front_matter.get("keywords", [])[:10],
                "task_types": front_matter.get("task_types", [])[:5],
                "priority": front_matter.get("priority", "medium"),
                "max_tokens": front_matter.get("max_tokens", 600),
            }

            # Add readme reference if exists
            readme_path = self._find_readme(claude_file.parent)
            if readme_path:
                context_entry["readme"] = readme_path

            # Add children if present
            children = front_matter.get("children", [])
            if children:
                context_entry["children"] = children[:10]

            index["contexts"].append(context_entry)
            indexed_count += 1

            # Build keyword index for O(1) lookups
            for keyword in context_entry["keywords"]:
                if isinstance(keyword, str):
                    keyword_lower = keyword.lower()
                    if relative_path not in keyword_to_paths[keyword_lower]:
                        keyword_to_paths[keyword_lower].append(relative_path)

        # Convert defaultdict to regular dict for YAML serialization
        index["keyword_index"] = dict(keyword_to_paths)

        # Load config to get index path
        index_path = self.root / ".claude" / "context-index.yaml"
        try:
            from config.loader import load_config

            config_file = self.root / ".claude" / "config.yaml"
            if config_file.exists():
                config = load_config(str(config_file))
                if hasattr(config, "context_config") and hasattr(
                    config.context_config, "index_path"
                ):
                    index_path = self.root / config.context_config.index_path
        except Exception:
            pass  # Use default path if config loading fails

        # Ensure directory exists
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save index file
        with open(index_path, "w", encoding="utf-8") as f:
            yaml.dump(
                index, f, default_flow_style=False, sort_keys=False, allow_unicode=True
            )

        print(f"✓ Generated context index: {index_path}")
        print(f"  Contexts indexed: {indexed_count}")
        print(f"  Keywords mapped: {len(index['keyword_index'])}")
        if skipped_count > 0:
            print(f"  Skipped (no front matter): {skipped_count}")

        return {
            "index_path": str(index_path),
            "contexts_indexed": indexed_count,
            "keywords_mapped": len(index["keyword_index"]),
            "skipped": skipped_count,
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI-Driven Context Generation Workflow"
    )
    parser.add_argument(
        "--target-dirs",
        nargs="+",
        help="Target specific directories (default: scan all)",
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Skip README.md generation for existing files",
    )
    parser.add_argument(
        "--skip-claude-md",
        action="store_true",
        help="Skip CLAUDE.md generation (only regenerate index)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum directory depth to scan (default: 3)",
    )

    args = parser.parse_args()

    # Print header
    print("=" * 80)
    print("CONTEXT GENERATION WORKFLOW (AI-Driven)")
    print("=" * 80)
    print(f"\nMax Depth: {args.max_depth}")
    print("Mode: AI-Driven (Claude API required)")

    if args.target_dirs:
        print(f"Target Directories: {', '.join(args.target_dirs)}")

    if args.skip_readme:
        print("⏭ Skipping README.md generation for existing files")
    if args.skip_claude_md:
        print("⏭ Skipping CLAUDE.md generation (index only)")

    print()

    # Initialize and execute workflow
    workflow = ContextGenerationWorkflow(
        target_dirs=args.target_dirs,
        skip_readme=args.skip_readme,
        skip_claude_md=args.skip_claude_md,
        max_depth=args.max_depth,
        args=args,
    )

    try:
        result = workflow.execute()

        print("\n" + "=" * 80)
        print("✅ CONTEXT GENERATION COMPLETE")
        print("=" * 80)

        # Print summary
        scan = workflow.step_evidence.get("1-scan-directories", {})
        identify = workflow.step_evidence.get("2-identify-documentable", {})
        print(f"\nDirectories Scanned: {scan.get('count', 0)}")
        print(f"Documentable (AI): {identify.get('count', 0)}")

        if not args.skip_readme:
            readme = workflow.step_evidence.get("3-generate-readme", {})
            print(f"README.md Generated: {readme.get('generated', 0)}")

        if not args.skip_claude_md:
            claude_md = workflow.step_evidence.get("4-generate-claude-md", {})
            print(f"CLAUDE.md Generated: {claude_md.get('generated', 0)}")
            print(f"CLAUDE.md Merged: {claude_md.get('merged', 0)}")

        verification = workflow.step_evidence.get("5-verify-context", {})
        passed = verification.get("passed", 0)
        total = verification.get("total", 0)
        warnings = verification.get("warnings", 0)
        errors = verification.get("errors", 0)

        print(
            f"\nVerification: {passed}/{total} passed ({warnings} warnings, {errors} errors)"
        )

        index_gen = workflow.step_evidence.get("6-generate-index", {})
        if index_gen:
            print(
                f"Context Index: {index_gen.get('contexts_indexed', 0)} contexts, {index_gen.get('keywords_mapped', 0)} keywords"
            )
            print(f"Index saved to: {index_gen.get('index_path', 'N/A')}")

        print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Workflow failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
