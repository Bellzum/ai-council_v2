#!/usr/bin/env python3
"""
Centralized Argument Configuration for Workflow Scripts

Provides a standardized registry of command-line arguments with consistent
names, types, help text, and validation rules across all workflow scripts.

This module centralizes argument definitions to ensure:
- Consistent flag names across scripts
- Uniform help text
- Shared validation logic
- Easy maintenance and updates

Usage:
    from scripts.workflow_executor.argument_config import (
        WorkflowArgumentParser,
        StandardArguments
    )

    parser = WorkflowArgumentParser(
        description="My Workflow Script",
        standard_args=[
            StandardArguments.SPRINT,
            StandardArguments.NO_AI,
            StandardArguments.CONFIG
        ]
    )
    parser.add_custom_argument("--my-arg", help="Custom argument")
    args = parser.parse()
"""

import argparse
import re
from enum import Enum
from pathlib import Path
from typing import List, Optional, Any, Dict


class StandardArguments(Enum):
    """
    Registry of standard arguments used across workflow scripts.

    Each enum value maps to a tuple of (flags, kwargs) for argparse.add_argument().
    """

    # Sprint identification (standardized to SPRINT-XXX format)
    SPRINT = (
        ["--sprint"],
        {
            "type": str,
            "help": "Sprint identifier (format: 'SPRINT-XXX' or numeric like '8')",
            "metavar": "SPRINT",
        }
    )

    # Work item targeting
    WORK_ITEM_IDS = (
        ["--work-item-ids"],
        {
            "type": str,
            "help": "Comma-separated list of work item IDs to target (e.g., '1234,1235,1236')",
            "metavar": "IDS",
        }
    )

    # AI mode control (negated flag - default is AI enabled)
    NO_AI = (
        ["--no-ai"],
        {
            "action": "store_true",
            "help": "Disable AI agents (Mode 1: pure Python data collection only)",
        }
    )

    # Interactive mode control (negated flag - default is non-interactive)
    NO_INTERACTIVE = (
        ["--no-interactive"],
        {
            "action": "store_true",
            "help": "Disable interactive prompts (use defaults for all choices)",
        }
    )

    # Configuration file
    CONFIG = (
        ["--config"],
        {
            "type": str,
            "help": "Path to configuration file (default: .claude/config.yaml)",
            "default": ".claude/config.yaml",
            "metavar": "PATH",
        }
    )

    # Workflow identification
    WORKFLOW_ID = (
        ["--workflow-id"],
        {
            "type": str,
            "help": "Unique workflow ID (default: auto-generated timestamp-based ID)",
            "metavar": "ID",
        }
    )

    # State management
    NO_CHECKPOINTS = (
        ["--no-checkpoints"],
        {
            "action": "store_true",
            "help": "Disable state checkpointing (workflow state will not be saved)",
        }
    )

    # Dry run mode
    DRY_RUN = (
        ["--dry-run"],
        {
            "action": "store_true",
            "help": "Show what would be done without making changes",
        }
    )

    # Verbose output
    VERBOSE = (
        ["-v", "--verbose"],
        {
            "action": "store_true",
            "help": "Enable verbose output with detailed logging",
        }
    )

    # Sprint planning specific
    SPRINT_NUMBER = (
        ["--sprint-number"],
        {
            "type": int,
            "required": True,
            "help": "Sprint number (e.g., 8 for 'Sprint 8')",
            "metavar": "NUM",
        }
    )

    CAPACITY = (
        ["--capacity"],
        {
            "type": int,
            "required": True,
            "help": "Team capacity in story points for sprint planning",
            "metavar": "POINTS",
        }
    )

    # Sprint execution specific
    MAX_ITERATION_CYCLES = (
        ["--max-iteration-cycles"],
        {
            "type": int,
            "default": 10,
            "help": "Maximum number of execution cycles (0 = unlimited)",
            "metavar": "N",
        }
    )

    CONFIRM_START = (
        ["--confirm-start"],
        {
            "action": "store_true",
            "help": "Prompt for confirmation before starting execution",
        }
    )

    # Backlog grooming specific
    MAX_EPICS = (
        ["--max-epics"],
        {
            "type": int,
            "help": "Maximum number of EPICs to process (limits scope of grooming)",
            "metavar": "N",
        }
    )

    TARGET_IDS = (
        ["--target-ids"],
        {
            "nargs": '+',
            "type": int,
            "help": "Specific work item IDs to target (EPICs, Features, or Tasks)",
            "metavar": "ID",
        }
    )

    AUTO_FIX_HIERARCHY = (
        ["--auto-fix-hierarchy"],
        {
            "action": "store_true",
            "help": "Automatically fix hierarchy issues (orphan Features, empty EPICs)",
        }
    )

    # Deprecated arguments (for backward compatibility)
    USE_AI_DEPRECATED = (
        ["--use-ai"],
        {
            "action": "store_true",
            "help": "(DEPRECATED: Use --no-ai to disable AI instead) Enable AI agents",
            "dest": "use_ai_deprecated",
        }
    )

    INTERACTIVE_DEPRECATED = (
        ["--interactive"],
        {
            "action": "store_true",
            "help": "(DEPRECATED: Interactive is now default, use --no-interactive to disable) Enable interactive mode",
            "dest": "interactive_deprecated",
        }
    )


class ArgumentValidator:
    """
    Validation logic for standardized arguments.

    Provides static methods for validating common argument patterns
    including sprint names, work item IDs, and file paths.
    """

    @staticmethod
    def validate_sprint_format(sprint: str) -> str:
        """
        Validate and normalize sprint identifier.

        Accepts:
        - Numeric: "8" → "SPRINT-008"
        - SPRINT-XXX: "SPRINT-008" → "SPRINT-008"
        - Sprint N: "Sprint 8" → "SPRINT-008"

        Args:
            sprint: Sprint identifier in any accepted format

        Returns:
            str: Normalized sprint identifier in SPRINT-XXX format

        Raises:
            ValueError: If sprint format is invalid
        """
        if not sprint:
            raise ValueError("Sprint identifier cannot be empty")

        sprint = sprint.strip()

        # Check if empty after stripping
        if not sprint:
            raise ValueError("Sprint identifier cannot be empty")

        # Handle numeric format: "8" → "SPRINT-008"
        # Only accept ASCII digits to avoid confusion with Unicode fullwidth digits
        if sprint.isdigit() and sprint.isascii():
            try:
                sprint_num = int(sprint)
                return f"SPRINT-{sprint_num:03d}"
            except ValueError as e:
                # Handle int conversion errors (e.g., number too large)
                raise ValueError(
                    f"Invalid sprint format: '{sprint}'. "
                    "Expected formats: '8', 'Sprint 8', or 'SPRINT-008'"
                ) from e

        # Handle "Sprint N" format: "Sprint 8" → "SPRINT-008"
        match = re.match(r'^Sprint\s+(\d+)$', sprint, re.IGNORECASE)
        if match:
            try:
                sprint_num = int(match.group(1))
                return f"SPRINT-{sprint_num:03d}"
            except ValueError as e:
                # Handle int conversion errors (e.g., number too large)
                raise ValueError(
                    f"Invalid sprint format: '{sprint}'. "
                    "Expected formats: '8', 'Sprint 8', or 'SPRINT-008'"
                ) from e

        # Handle SPRINT-XXX format: "SPRINT-008" → "SPRINT-008"
        match = re.match(r'^SPRINT-(\d{3})$', sprint, re.IGNORECASE)
        if match:
            return sprint.upper()

        # Invalid format
        raise ValueError(
            f"Invalid sprint format: '{sprint}'. "
            "Expected formats: '8', 'Sprint 8', or 'SPRINT-008'"
        )

    @staticmethod
    def validate_work_item_ids(ids_str: str) -> List[int]:
        """
        Validate and parse comma-separated work item IDs.

        Args:
            ids_str: Comma-separated list of work item IDs (e.g., "1234,1235,1236")

        Returns:
            List[int]: Parsed and validated work item IDs

        Raises:
            ValueError: If any ID is invalid or not numeric
        """
        if not ids_str:
            return []

        ids_str = ids_str.strip()
        if not ids_str:
            return []

        parts = ids_str.split(',')
        work_item_ids = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check if it's a valid integer (handles negative numbers)
            try:
                work_item_id = int(part)
            except ValueError:
                raise ValueError(
                    f"Invalid work item ID: '{part}'. "
                    "Work item IDs must be numeric."
                )

            # Work tracking systems (Azure DevOps, Jira, GitHub) use IDs starting from 1
            if work_item_id <= 0:
                raise ValueError(
                    f"Invalid work item ID: {work_item_id}. "
                    "Work item IDs must be positive integers (>= 1)."
                )

            work_item_ids.append(work_item_id)

        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for work_item_id in work_item_ids:
            if work_item_id not in seen:
                seen.add(work_item_id)
                unique_ids.append(work_item_id)

        return unique_ids

    @staticmethod
    def validate_config_path(config_path: str) -> Path:
        """
        Validate configuration file path.

        Args:
            config_path: Path to configuration file

        Returns:
            Path: Validated Path object

        Raises:
            ValueError: If config file doesn't exist or is not a file
        """
        path = Path(config_path)
        if not path.exists():
            raise ValueError(
                f"Configuration file not found: {config_path}\n"
                "Run 'trustable-ai init' to create default configuration."
            )
        if not path.is_file():
            raise ValueError(
                f"Configuration path is not a file: {config_path}\n"
                "Expected a YAML configuration file."
            )
        return path

    @staticmethod
    def validate_positive_int(value: int, name: str) -> int:
        """
        Validate that an integer value is positive.

        Args:
            value: Integer value to validate
            name: Name of the parameter (for error messages)

        Returns:
            int: The validated value

        Raises:
            ValueError: If value is not positive
        """
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer, got: {value}")
        return value

    @staticmethod
    def validate_non_negative_int(value: int, name: str) -> int:
        """
        Validate that an integer value is non-negative (0 or positive).

        Args:
            value: Integer value to validate
            name: Name of the parameter (for error messages)

        Returns:
            int: The validated value

        Raises:
            ValueError: If value is negative
        """
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got: {value}")
        return value


class WorkflowArgumentParser:
    """
    Standardized argument parser for workflow scripts.

    Provides a consistent interface for creating argparse parsers with
    standardized arguments. Supports both standard and custom arguments,
    with built-in validation and backward compatibility.

    Example:
        parser = WorkflowArgumentParser(
            description="Sprint Planning Workflow",
            standard_args=[
                StandardArguments.SPRINT_NUMBER,
                StandardArguments.CAPACITY,
                StandardArguments.NO_AI,
                StandardArguments.CONFIG
            ]
        )
        parser.add_custom_argument("--my-flag", action="store_true")
        args = parser.parse()
    """

    def __init__(
        self,
        description: str,
        standard_args: Optional[List[StandardArguments]] = None,
        add_deprecated_aliases: bool = True
    ):
        """
        Initialize workflow argument parser.

        Args:
            description: Description for the argument parser
            standard_args: List of StandardArguments to include
            add_deprecated_aliases: If True, add deprecated argument aliases
        """
        self.parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        self.standard_args = standard_args or []
        self.add_deprecated_aliases = add_deprecated_aliases

        # Add standard arguments
        for std_arg in self.standard_args:
            flags, kwargs = std_arg.value
            self.parser.add_argument(*flags, **kwargs)

        # Add deprecated aliases if requested
        if self.add_deprecated_aliases:
            if StandardArguments.NO_AI in self.standard_args:
                flags, kwargs = StandardArguments.USE_AI_DEPRECATED.value
                self.parser.add_argument(*flags, **kwargs)

            if StandardArguments.NO_INTERACTIVE in self.standard_args:
                flags, kwargs = StandardArguments.INTERACTIVE_DEPRECATED.value
                self.parser.add_argument(*flags, **kwargs)

    def add_custom_argument(self, *args, **kwargs):
        """
        Add a custom argument not in the standard registry.

        Args:
            *args: Positional arguments for argparse.add_argument()
            **kwargs: Keyword arguments for argparse.add_argument()
        """
        self.parser.add_argument(*args, **kwargs)

    def parse(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse arguments and apply validation.

        Args:
            args: Optional list of arguments to parse (defaults to sys.argv)

        Returns:
            argparse.Namespace: Parsed and validated arguments

        Raises:
            SystemExit: If validation fails
        """
        parsed_args = self.parser.parse_args(args)

        # Handle deprecated arguments
        if hasattr(parsed_args, 'use_ai_deprecated') and parsed_args.use_ai_deprecated:
            print("WARNING: --use-ai is deprecated. AI is now enabled by default.")
            print("         Use --no-ai to disable AI instead.")

        if hasattr(parsed_args, 'interactive_deprecated') and parsed_args.interactive_deprecated:
            print("WARNING: --interactive is deprecated. Interactive mode is now default.")
            print("         Use --no-interactive to disable interactive prompts.")

        # Apply validation
        try:
            self._validate_arguments(parsed_args)
        except ValueError as e:
            self.parser.error(str(e))

        return parsed_args

    def _validate_arguments(self, args: argparse.Namespace):
        """
        Validate parsed arguments.

        Args:
            args: Parsed arguments namespace

        Raises:
            ValueError: If validation fails
        """
        # Validate sprint format
        if StandardArguments.SPRINT in self.standard_args:
            if hasattr(args, 'sprint') and args.sprint:
                args.sprint = ArgumentValidator.validate_sprint_format(args.sprint)

        # Validate sprint number
        if StandardArguments.SPRINT_NUMBER in self.standard_args:
            if hasattr(args, 'sprint_number') and args.sprint_number is not None:
                ArgumentValidator.validate_positive_int(args.sprint_number, "Sprint number")

        # Validate capacity
        if StandardArguments.CAPACITY in self.standard_args:
            if hasattr(args, 'capacity') and args.capacity is not None:
                ArgumentValidator.validate_positive_int(args.capacity, "Team capacity")

        # Validate work item IDs
        if StandardArguments.WORK_ITEM_IDS in self.standard_args:
            if hasattr(args, 'work_item_ids') and args.work_item_ids:
                args.work_item_ids = ArgumentValidator.validate_work_item_ids(args.work_item_ids)

        # Validate config path (only if not default and file should exist)
        if StandardArguments.CONFIG in self.standard_args:
            if hasattr(args, 'config') and args.config and args.config != ".claude/config.yaml":
                # Only validate non-default config paths
                ArgumentValidator.validate_config_path(args.config)

        # Validate max iteration cycles
        if StandardArguments.MAX_ITERATION_CYCLES in self.standard_args:
            if hasattr(args, 'max_iteration_cycles'):
                ArgumentValidator.validate_non_negative_int(
                    args.max_iteration_cycles,
                    "Max iteration cycles"
                )

        # Validate max epics
        if StandardArguments.MAX_EPICS in self.standard_args:
            if hasattr(args, 'max_epics') and args.max_epics is not None:
                ArgumentValidator.validate_positive_int(args.max_epics, "Max EPICs")

        # Validate target IDs
        if StandardArguments.TARGET_IDS in self.standard_args:
            if hasattr(args, 'target_ids') and args.target_ids:
                for work_item_id in args.target_ids:
                    ArgumentValidator.validate_positive_int(work_item_id, "Target work item ID")
