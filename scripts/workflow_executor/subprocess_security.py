"""
Subprocess Security Wrapper

Provides secure subprocess execution with:
- Command allowlist enforcement
- NEVER uses shell=True
- Comprehensive audit logging
- Timeout protection

CRITICAL SECURITY REQUIREMENTS:
1. NEVER use shell=True (prevents command injection)
2. Always validate commands against allowlist
3. Log all executions to audit trail
4. Use subprocess.run() with explicit argument lists
"""

import subprocess
from pathlib import Path
from typing import List, Set, Optional, Dict, Any
from datetime import datetime
import json
import sys


class SubprocessSecurityError(Exception):
    """Raised when subprocess security violation is detected."""
    pass


class SecureSubprocess:
    """
    Secure subprocess execution wrapper.

    Enforces security constraints:
    - Command allowlist (only approved commands can run)
    - No shell=True (prevents injection attacks)
    - Audit logging (all executions logged)
    - Timeout protection (prevent runaway processes)

    Example:
        executor = SecureSubprocess(
            allowed_commands={"pytest", "git", "docker"}
        )

        result = executor.execute_command(
            command=["pytest", "tests/", "-v"],
            timeout=300
        )

        if result["returncode"] == 0:
            print("Tests passed!")
    """

    def __init__(
        self,
        allowed_commands: Optional[Set[str]] = None,
        audit_log_dir: Optional[Path] = None,
        strict_mode: bool = True
    ):
        """
        Initialize secure subprocess executor.

        Args:
            allowed_commands: Set of allowed command names (e.g., {"pytest", "git"})
            audit_log_dir: Directory for audit logs (defaults to .claude/audit)
            strict_mode: If True, raise error on empty allowlist (default: True)
        """
        self.allowed_commands = allowed_commands or set()
        self.audit_log_dir = audit_log_dir or Path(".claude/audit")
        self.strict_mode = strict_mode
        self.audit_log_dir.mkdir(parents=True, exist_ok=True)

        # Audit log file (one per day)
        today = datetime.now().strftime("%Y%m%d")
        self.audit_log_file = self.audit_log_dir / f"subprocess-{today}.log"

    def execute_command(
        self,
        command: List[str],
        timeout: int = 300,
        capture_output: bool = True,
        check: bool = False,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Execute command securely.

        Args:
            command: Command as list of strings (e.g., ["pytest", "tests/"])
            timeout: Timeout in seconds
            capture_output: Whether to capture stdout/stderr
            check: Whether to raise exception on non-zero exit
            cwd: Working directory for command
            env: Environment variables

        Returns:
            Dict with:
                - returncode: Exit code
                - stdout: Standard output (if captured)
                - stderr: Standard error (if captured)
                - duration: Execution time in seconds

        Raises:
            SubprocessSecurityError: If command not allowed or invalid
            subprocess.TimeoutExpired: If command times out
            subprocess.CalledProcessError: If check=True and command fails
        """
        # Validate command
        self._validate_command(command)

        # Log execution start
        execution_id = self._log_execution_start(command, cwd)

        start_time = datetime.now()

        try:
            # Execute command (NEVER use shell=True!)
            result = subprocess.run(
                command,
                timeout=timeout,
                capture_output=capture_output,
                check=check,
                cwd=cwd,
                env=env,
                shell=False,  # CRITICAL: Never use shell=True
            )

            duration = (datetime.now() - start_time).total_seconds()

            # Prepare result dict
            result_dict = {
                "returncode": result.returncode,
                "duration": duration,
                "stdout": result.stdout.decode("utf-8") if capture_output else None,
                "stderr": result.stderr.decode("utf-8") if capture_output else None,
            }

            # Log success
            self._log_execution_end(execution_id, result_dict, success=True)

            return result_dict

        except subprocess.TimeoutExpired as e:
            duration = (datetime.now() - start_time).total_seconds()

            # Log timeout
            self._log_execution_end(
                execution_id,
                {"error": "timeout", "duration": duration},
                success=False
            )

            raise

        except subprocess.CalledProcessError as e:
            duration = (datetime.now() - start_time).total_seconds()

            # Log failure
            self._log_execution_end(
                execution_id,
                {
                    "returncode": e.returncode,
                    "duration": duration,
                    "error": "non_zero_exit"
                },
                success=False
            )

            raise

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()

            # Log unexpected error
            self._log_execution_end(
                execution_id,
                {"error": str(e), "duration": duration},
                success=False
            )

            raise

    def _validate_command(self, command: List[str]) -> None:
        """
        Validate command against security constraints.

        Args:
            command: Command list

        Raises:
            SubprocessSecurityError: If command invalid or not allowed
        """
        # Command must be non-empty list
        if not command or not isinstance(command, list):
            raise SubprocessSecurityError("Command must be non-empty list")

        # All elements must be strings
        if not all(isinstance(arg, str) for arg in command):
            raise SubprocessSecurityError("All command arguments must be strings")

        # Extract command name (first element)
        command_name = command[0]

        # Check for empty allowlist
        if not self.allowed_commands:
            if self.strict_mode:
                raise SubprocessSecurityError(
                    "Empty command allowlist detected (security risk).\n"
                    "\n"
                    "An empty allowlist bypasses all command validation, allowing\n"
                    "arbitrary command execution including command injection attacks.\n"
                    "\n"
                    "Solutions:\n"
                    "  1. Populate allowed_commands with approved command patterns\n"
                    "  2. Set strict_mode=False for testing/development only\n"
                    "\n"
                    "Example allowlist:\n"
                    "  allowed_commands = {'git', 'pytest', 'python'}\n"
                )
            else:
                # Non-strict mode: Print warning but allow
                print(
                    "⚠️  WARNING: Empty command allowlist - all commands allowed\n"
                    "   This is a security risk. Only use in testing/development.",
                    file=sys.stderr
                )
                return  # Allow command

        # Check against allowlist (non-empty)
        # Handle full paths (e.g., /usr/bin/pytest -> pytest)
        base_command = Path(command_name).name

        if base_command not in self.allowed_commands:
            raise SubprocessSecurityError(
                f"Command '{base_command}' not in allowlist. "
                f"Allowed: {sorted(self.allowed_commands)}"
            )

        # Reject shell metacharacters in command name
        shell_metacharacters = {"|", "&", ";", "$", "`", "(", ")", "<", ">", "\n"}
        if any(char in command_name for char in shell_metacharacters):
            raise SubprocessSecurityError(
                f"Command name contains shell metacharacters: {command_name}"
            )

    def _log_execution_start(
        self,
        command: List[str],
        cwd: Optional[str]
    ) -> str:
        """
        Log command execution start.

        Args:
            command: Command list
            cwd: Working directory

        Returns:
            Execution ID for correlation
        """
        execution_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

        log_entry = {
            "execution_id": execution_id,
            "timestamp": datetime.now().isoformat(),
            "event": "execution_start",
            "command": command,
            "cwd": cwd,
        }

        self._append_to_audit_log(log_entry)

        return execution_id

    def _log_execution_end(
        self,
        execution_id: str,
        result: Dict[str, Any],
        success: bool
    ) -> None:
        """
        Log command execution end.

        Args:
            execution_id: Execution ID from start
            result: Execution result
            success: Whether execution succeeded
        """
        log_entry = {
            "execution_id": execution_id,
            "timestamp": datetime.now().isoformat(),
            "event": "execution_end",
            "success": success,
            "result": result,
        }

        self._append_to_audit_log(log_entry)

    def _append_to_audit_log(self, log_entry: Dict[str, Any]) -> None:
        """
        Append entry to audit log.

        Args:
            log_entry: Log entry dict
        """
        with open(self.audit_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + "\n")

    def add_allowed_command(self, command: str) -> None:
        """
        Add command to allowlist.

        Args:
            command: Command name to allow
        """
        self.allowed_commands.add(command)

    def remove_allowed_command(self, command: str) -> None:
        """
        Remove command from allowlist.

        Args:
            command: Command name to disallow
        """
        self.allowed_commands.discard(command)

    def is_command_allowed(self, command: str) -> bool:
        """
        Check if command is allowed.

        Args:
            command: Command name

        Returns:
            True if allowed (or no allowlist configured)
        """
        if not self.allowed_commands:
            return True  # No allowlist = all allowed

        return command in self.allowed_commands

    def get_audit_log_entries(
        self,
        execution_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get audit log entries.

        Args:
            execution_id: Optional filter by execution ID
            limit: Maximum number of entries to return

        Returns:
            List of log entries (most recent first)
        """
        if not self.audit_log_file.exists():
            return []

        entries = []

        with open(self.audit_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())

                    # Filter by execution ID if specified
                    if execution_id and entry.get("execution_id") != execution_id:
                        continue

                    entries.append(entry)

                except json.JSONDecodeError:
                    continue

        # Return most recent first
        return entries[-limit:][::-1]


# Default secure executor for common commands
def get_default_executor() -> SecureSubprocess:
    """
    Get default secure subprocess executor with common allowed commands.

    Returns:
        SecureSubprocess with common commands allowed
    """
    return SecureSubprocess(
        allowed_commands={
            "pytest",
            "python",
            "python3",
            "git",
            "docker",
            "npm",
            "node",
            "pip",
            "black",
            "ruff",
            "mypy",
        }
    )


# Example usage
if __name__ == "__main__":
    # Create secure executor
    executor = SecureSubprocess(
        allowed_commands={"echo", "pytest", "git"}
    )

    print("Testing secure subprocess execution...")

    # Test 1: Allowed command
    try:
        result = executor.execute_command(
            command=["echo", "Hello, World!"],
            timeout=5
        )
        print(f"✓ Allowed command executed: returncode={result['returncode']}")
        print(f"  Output: {result['stdout'].strip()}")
    except Exception as e:
        print(f"✗ Allowed command failed: {e}")

    # Test 2: Disallowed command (should fail)
    try:
        result = executor.execute_command(
            command=["rm", "-rf", "/"],
            timeout=5
        )
        print(f"✗ SECURITY FAILURE: Disallowed command executed!")
    except SubprocessSecurityError as e:
        print(f"✓ Disallowed command blocked: {e}")

    # Test 3: Shell injection attempt (should fail)
    try:
        result = executor.execute_command(
            command=["echo", "test", "|", "cat", "/etc/passwd"],
            timeout=5
        )
        print(f"✗ SECURITY FAILURE: Shell injection not detected!")
    except Exception as e:
        print(f"✓ Shell injection blocked (command not in allowlist)")

    # Test 4: View audit log
    print("\nAudit log entries:")
    entries = executor.get_audit_log_entries(limit=5)
    for entry in entries:
        print(f"  - {entry['timestamp']}: {entry['event']} - {entry.get('command', 'N/A')}")

    print(f"\n✓ Full audit log: {executor.audit_log_file}")
