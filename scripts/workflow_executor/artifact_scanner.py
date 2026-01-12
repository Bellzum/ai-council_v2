"""
Artifact Scanner Module

Encapsulates pattern-based file scanning logic for the artifact hygiene workflow.
Provides reusable scanning functionality that can be used independently or as
part of the full workflow.
"""

import fnmatch
import glob
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class ArtifactInfo:
    """Information about a discovered artifact."""
    path: str
    filename: str
    size: int
    modified: datetime
    is_empty: bool
    category: Optional[str] = None
    destination: Optional[str] = None
    confidence: str = "unknown"


@dataclass
class ScanResult:
    """Result of an artifact scan."""
    artifacts: List[ArtifactInfo] = field(default_factory=list)
    by_pattern: Dict[str, int] = field(default_factory=dict)
    scanned_at: datetime = field(default_factory=datetime.now)

    @property
    def total_count(self) -> int:
        return len(self.artifacts)


# Default categorization rules
DEFAULT_CATEGORIZATION_RULES = {
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

# Files that should never be moved
DEFAULT_EXCLUSIONS = {
    "README.md", "CLAUDE.md", "VISION.md", "CHANGELOG.md", "LICENSE.md", "LICENSE",
    "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md"
}


class ArtifactScanner:
    """
    Scanner for finding and categorizing project artifacts.

    Usage:
        scanner = ArtifactScanner()
        result = scanner.scan(".")
        for artifact in result.artifacts:
            print(f"{artifact.filename} -> {artifact.category}")
    """

    def __init__(
        self,
        categorization_rules: Optional[Dict] = None,
        exclusions: Optional[Set[str]] = None,
        file_extensions: Optional[List[str]] = None
    ):
        """
        Initialize the artifact scanner.

        Args:
            categorization_rules: Custom categorization rules (default: DEFAULT_CATEGORIZATION_RULES)
            exclusions: Files to never include (default: DEFAULT_EXCLUSIONS)
            file_extensions: File extensions to scan (default: [".md", ".txt"])
        """
        self.categorization_rules = categorization_rules or DEFAULT_CATEGORIZATION_RULES
        self.exclusions = exclusions or DEFAULT_EXCLUSIONS
        self.file_extensions = file_extensions or [".md", ".txt"]

    def scan(
        self,
        target_dir: str = ".",
        recursive: bool = False
    ) -> ScanResult:
        """
        Scan a directory for artifacts.

        Args:
            target_dir: Directory to scan
            recursive: If True, scan subdirectories (default: False)

        Returns:
            ScanResult with found artifacts
        """
        result = ScanResult()
        pattern_counts: Dict[str, int] = {}

        # Build glob patterns
        for ext in self.file_extensions:
            if recursive:
                pattern = os.path.join(target_dir, "**", f"*{ext}")
                files = glob.glob(pattern, recursive=True)
            else:
                pattern = os.path.join(target_dir, f"*{ext}")
                files = glob.glob(pattern)

            for filepath in files:
                filename = os.path.basename(filepath)

                # Skip excluded files
                if filename in self.exclusions:
                    continue

                # Skip files in hidden directories
                if any(part.startswith('.') for part in Path(filepath).parts[:-1]):
                    continue

                # Get file metadata
                try:
                    stat = os.stat(filepath)
                    artifact = ArtifactInfo(
                        path=filepath,
                        filename=filename,
                        size=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime),
                        is_empty=stat.st_size == 0
                    )

                    # Categorize by pattern
                    self._categorize_artifact(artifact, pattern_counts)
                    result.artifacts.append(artifact)

                except OSError:
                    # Skip files we can't access
                    continue

        result.by_pattern = pattern_counts
        return result

    def _categorize_artifact(self, artifact: ArtifactInfo, pattern_counts: Dict[str, int]) -> None:
        """
        Categorize an artifact based on pattern matching.

        Args:
            artifact: Artifact to categorize (modified in place)
            pattern_counts: Counter dict to update
        """
        for category, rules in self.categorization_rules.items():
            for pattern in rules["patterns"]:
                if fnmatch.fnmatch(artifact.filename, pattern):
                    artifact.category = category
                    artifact.destination = rules["destination"]
                    artifact.confidence = "rule"
                    pattern_counts[category] = pattern_counts.get(category, 0) + 1
                    return

        # No match found
        artifact.category = "unknown"
        pattern_counts["unknown"] = pattern_counts.get("unknown", 0) + 1

    def scan_multiple(self, target_dirs: List[str]) -> ScanResult:
        """
        Scan multiple directories and combine results.

        Args:
            target_dirs: List of directories to scan

        Returns:
            Combined ScanResult
        """
        combined = ScanResult()

        for target_dir in target_dirs:
            result = self.scan(target_dir)
            combined.artifacts.extend(result.artifacts)

            for category, count in result.by_pattern.items():
                combined.by_pattern[category] = combined.by_pattern.get(category, 0) + count

        return combined

    def get_movable_artifacts(self, result: ScanResult) -> List[ArtifactInfo]:
        """
        Filter artifacts that have a known destination.

        Args:
            result: Scan result to filter

        Returns:
            List of artifacts with destinations
        """
        return [a for a in result.artifacts if a.destination is not None]

    def get_unknown_artifacts(self, result: ScanResult) -> List[ArtifactInfo]:
        """
        Filter artifacts with unknown category.

        Args:
            result: Scan result to filter

        Returns:
            List of unknown artifacts
        """
        return [a for a in result.artifacts if a.category == "unknown"]


def quick_scan(target_dir: str = ".") -> ScanResult:
    """
    Quick utility function for scanning without creating scanner instance.

    Args:
        target_dir: Directory to scan

    Returns:
        ScanResult with found artifacts
    """
    scanner = ArtifactScanner()
    return scanner.scan(target_dir)


if __name__ == "__main__":
    # Quick test when run directly
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Scanning: {target}")

    result = quick_scan(target)
    print(f"\nFound {result.total_count} artifacts:")

    for category, count in sorted(result.by_pattern.items()):
        print(f"  {category}: {count}")

    print("\nArtifacts:")
    for artifact in result.artifacts[:10]:
        status = "→" if artifact.destination else "?"
        dest = artifact.destination or "unknown"
        print(f"  {status} {artifact.filename} -> {dest}")

    if result.total_count > 10:
        print(f"  ... and {result.total_count - 10} more")
