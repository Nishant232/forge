from __future__ import annotations

from abc import ABC, abstractmethod


class ApprovalGate(ABC):
    """Interface for requesting human approval before a sensitive action."""

    @abstractmethod
    def request(self, action: str, description: str) -> bool:
        """Return True if the action is approved, False to deny."""
        ...


class AutoApprove(ApprovalGate):
    """Always approves. Used in automated / testing contexts."""

    def request(self, action: str, description: str) -> bool:
        return True


class CLIApprove(ApprovalGate):
    """Prompts the user on stdin for approval of a sensitive action."""

    def request(self, action: str, description: str) -> bool:
        print(f"\n[APPROVAL REQUIRED] {action}")
        print(f"  {description}")
        try:
            response = input("Allow? [y/N] ").strip().lower()
        except EOFError:
            return False
        return response in ("y", "yes")
