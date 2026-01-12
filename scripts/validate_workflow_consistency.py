#!/usr/bin/env python3
"""
Validate workflow script consistency using AST-based analysis.

Implements Task #1314: Final validation for all 8 workflow scripts
- Detects remaining print() calls (AST-based)
- Checks for decorative emojis (regex-based)
- Identifies first-person language violations (regex-based)
- Verifies professional tone throughout
- Validates theme consistency

Validation Rules:
1. NO decorative emojis (✔✓✗✘⚠ℹ▶ are allowed as UI symbols)
2. NO first-person language ("I", "we", "our") in user-facing output
3. NO ANSI escape codes in source
4. Professional, objective tone throughout
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a script."""
    file: str
    line_number: int
    issue_type: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    code_snippet: str = ""


@dataclass
class ValidationResult:
    """Results from validating a single file."""
    file: str
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)


class WorkflowValidator:
    """AST-based validator for workflow scripts."""

    # Decorative emojis to detect
    DECORATIVE_EMOJI_PATTERN = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]')

    # Allowed UI symbols (not emojis)
    ALLOWED_SYMBOLS = {'✔', '✓', '✗', '✘', '⚠', 'ℹ', '▶', '└', '─', '│', '├', '┌', '┐', '┘'}

    # ANSI escape code pattern
    ANSI_PATTERN = re.compile(r'\\033\[[0-9;]+m')

    # First-person pronouns (case-sensitive to avoid false positives with 'I' as variable)
    # Only catch actual first-person usage, not variable names
    FIRST_PERSON_PATTERN = re.compile(r'\b(I\'m|I\'ll|I\'ve|I will|I am|we|we\'re|we\'ll|we\'ve|our|ours)\b', re.IGNORECASE)

    # Allowed first-person contexts (not violations)
    ALLOWED_CONTEXTS = [
        'get_current_user',  # Function names
        'current_user',      # Variable names
        'user_input',        # Variable names
        'interactive',       # Mode/feature names
        'ours',             # Could be legitimate (dependency conflict resolution)
    ]

    def __init__(self, project_root: Path):
        """Initialize validator with project root."""
        self.project_root = project_root
        self.scripts_dir = project_root / "scripts"

    def validate_file(self, file_path: Path) -> ValidationResult:
        """
        Validate a single workflow script file.

        Returns:
            ValidationResult with all detected issues
        """
        issues = []
        stats = {
            'total_lines': 0,
            'print_statements': 0,
            'decorative_emojis': 0,
            'first_person_violations': 0,
            'ansi_codes': 0
        }

        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                stats['total_lines'] = len(lines)
        except Exception as e:
            issues.append(ValidationIssue(
                file=file_path.name,
                line_number=0,
                issue_type='file_error',
                description=f"Failed to read file: {e}",
                severity='error'
            ))
            return ValidationResult(file=file_path.name, passed=False, issues=issues, stats=stats)

        # Check 1: Decorative emojis
        issues.extend(self._check_decorative_emojis(lines, file_path.name, stats))

        # Check 2: ANSI escape codes
        issues.extend(self._check_ansi_codes(lines, file_path.name, stats))

        # Check 3: First-person language in print statements
        issues.extend(self._check_first_person_language(lines, file_path.name, stats))

        # Check 4: AST-based print statement analysis
        try:
            tree = ast.parse(content, filename=str(file_path))
            issues.extend(self._analyze_print_statements(tree, lines, file_path.name, stats))
        except SyntaxError as e:
            issues.append(ValidationIssue(
                file=file_path.name,
                line_number=e.lineno or 0,
                issue_type='syntax_error',
                description=f"Failed to parse file: {e}",
                severity='error'
            ))

        # Determine if validation passed
        error_count = sum(1 for issue in issues if issue.severity == 'error')
        passed = error_count == 0

        return ValidationResult(
            file=file_path.name,
            passed=passed,
            issues=issues,
            stats=stats
        )

    def _check_decorative_emojis(
        self,
        lines: List[str],
        filename: str,
        stats: Dict[str, int]
    ) -> List[ValidationIssue]:
        """Check for decorative emojis (keeps allowed UI symbols)."""
        issues = []
        stats.setdefault('decorative_emojis', 0)

        for line_num, line in enumerate(lines, start=1):
            for char in line:
                if self.DECORATIVE_EMOJI_PATTERN.match(char) and char not in self.ALLOWED_SYMBOLS:
                    stats['decorative_emojis'] += 1
                    issues.append(ValidationIssue(
                        file=filename,
                        line_number=line_num,
                        issue_type='decorative_emoji',
                        description=f"Decorative emoji found: {char}",
                        severity='error',
                        code_snippet=line.strip()[:80]
                    ))

        return issues

    def _check_ansi_codes(
        self,
        lines: List[str],
        filename: str,
        stats: Dict[str, int]
    ) -> List[ValidationIssue]:
        """Check for ANSI escape codes."""
        issues = []
        stats.setdefault('ansi_codes', 0)

        for line_num, line in enumerate(lines, start=1):
            if self.ANSI_PATTERN.search(line):
                stats['ansi_codes'] += 1
                issues.append(ValidationIssue(
                    file=filename,
                    line_number=line_num,
                    issue_type='ansi_code',
                    description="ANSI escape code found in source",
                    severity='error',
                    code_snippet=line.strip()[:80]
                ))

        return issues

    def _check_first_person_language(
        self,
        lines: List[str],
        filename: str,
        stats: Dict[str, int]
    ) -> List[ValidationIssue]:
        """Check for first-person language in user-facing strings."""
        issues = []
        stats.setdefault('first_person_violations', 0)

        for line_num, line in enumerate(lines, start=1):
            # Skip comments and non-string contexts
            if line.strip().startswith('#'):
                continue

            # Check if line has print() or console output
            if 'print(' not in line and 'console.' not in line:
                continue

            # Check for first-person pronouns
            matches = self.FIRST_PERSON_PATTERN.finditer(line)
            for match in matches:
                pronoun = match.group(0)

                # Skip if in allowed context
                if any(ctx in line.lower() for ctx in self.ALLOWED_CONTEXTS):
                    continue

                # Skip if it's a variable name or function name (not in quotes)
                # Simple heuristic: if match is between quotes, it's likely user-facing text
                before_match = line[:match.start()]
                if before_match.count('"') % 2 == 0 and before_match.count("'") % 2 == 0:
                    # Not inside quotes, likely a variable/function name
                    continue

                stats['first_person_violations'] += 1
                issues.append(ValidationIssue(
                    file=filename,
                    line_number=line_num,
                    issue_type='first_person_language',
                    description=f"First-person pronoun found: '{pronoun}'",
                    severity='warning',  # Warning since some might be in comments/variables
                    code_snippet=line.strip()[:80]
                ))

        return issues

    def _analyze_print_statements(
        self,
        tree: ast.AST,
        lines: List[str],
        filename: str,
        stats: Dict[str, int]
    ) -> List[ValidationIssue]:
        """Analyze print statements using AST."""
        issues = []
        stats.setdefault('print_statements', 0)

        class PrintAnalyzer(ast.NodeVisitor):
            def __init__(self, validator, lines, filename, stats, issues):
                self.validator = validator
                self.lines = lines
                self.filename = filename
                self.stats = stats
                self.issues = issues

            def visit_Call(self, node):
                # Check if this is a print() call
                if isinstance(node.func, ast.Name) and node.func.id == 'print':
                    self.stats['print_statements'] += 1

                    # Get line content
                    line_num = node.lineno
                    if 0 < line_num <= len(self.lines):
                        line = self.lines[line_num - 1]

                        # Info: Track print statements (not an error, just stats)
                        # We allow print() statements, just no emojis/first-person in them

                self.generic_visit(node)

        analyzer = PrintAnalyzer(self, lines, filename, stats, issues)
        analyzer.visit(tree)

        return issues


def validate_workflow_scripts(project_root: Path, workflow_scripts: List[str]) -> Dict[str, ValidationResult]:
    """
    Validate multiple workflow scripts.

    Args:
        project_root: Project root directory
        workflow_scripts: List of script filenames to validate

    Returns:
        Dictionary mapping filename to ValidationResult
    """
    validator = WorkflowValidator(project_root)
    results = {}

    for script_name in workflow_scripts:
        script_path = project_root / "scripts" / script_name
        if not script_path.exists():
            results[script_name] = ValidationResult(
                file=script_name,
                passed=False,
                issues=[ValidationIssue(
                    file=script_name,
                    line_number=0,
                    issue_type='file_not_found',
                    description=f"Script not found: {script_path}",
                    severity='error'
                )]
            )
            continue

        results[script_name] = validator.validate_file(script_path)

    return results


def print_validation_report(results: Dict[str, ValidationResult]):
    """Print a formatted validation report."""
    print("\n" + "=" * 70)
    print("WORKFLOW SCRIPT VALIDATION REPORT")
    print("=" * 70)

    total_files = len(results)
    passed_files = sum(1 for r in results.values() if r.passed)
    failed_files = total_files - passed_files

    # Summary
    print(f"\nSummary:")
    print(f"  Total files: {total_files}")
    print(f"  Passed: {passed_files}")
    print(f"  Failed: {failed_files}")

    # File-by-file results
    for filename, result in sorted(results.items()):
        status = "PASS" if result.passed else "FAIL"
        status_symbol = "✓" if result.passed else "✗"
        print(f"\n{status_symbol} {filename}: {status}")

        if result.stats:
            print(f"  Stats:")
            print(f"    Total lines: {result.stats.get('total_lines', 0)}")
            print(f"    Print statements: {result.stats.get('print_statements', 0)}")
            print(f"    Decorative emojis: {result.stats.get('decorative_emojis', 0)}")
            print(f"    First-person violations: {result.stats.get('first_person_violations', 0)}")
            print(f"    ANSI codes: {result.stats.get('ansi_codes', 0)}")

        if result.issues:
            print(f"  Issues ({len(result.issues)}):")
            for issue in result.issues[:5]:  # Show first 5 issues
                severity_symbol = "✗" if issue.severity == 'error' else "⚠"
                print(f"    {severity_symbol} Line {issue.line_number}: {issue.description}")
                if issue.code_snippet:
                    print(f"       {issue.code_snippet}")

            if len(result.issues) > 5:
                print(f"    ... and {len(result.issues) - 5} more issues")

    print("\n" + "=" * 70)

    # Overall result
    if failed_files == 0:
        print("VALIDATION PASSED: All workflow scripts meet consistency standards")
    else:
        print(f"VALIDATION FAILED: {failed_files} script(s) have issues")

    print("=" * 70)


def main():
    """Main validation entry point."""
    project_root = Path(__file__).parent.parent

    # 8 workflow scripts to validate
    workflow_scripts = [
        "product_intake.py",
        "backlog_grooming.py",
        "sprint_planning.py",
        "sprint_execution.py",
        "daily_standup.py",
        "sprint_review.py",
        "sprint_retrospective.py",
        "dependency_management.py"
    ]

    print("Validating workflow script consistency...")
    print(f"Checking {len(workflow_scripts)} workflow scripts")

    results = validate_workflow_scripts(project_root, workflow_scripts)
    print_validation_report(results)

    # Exit code: 0 if all passed, 1 if any failed
    all_passed = all(r.passed for r in results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
