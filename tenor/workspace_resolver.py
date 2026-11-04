"""
Pre-Flight Workspace Path Resolver & Dual-Tier Cache Subsystem (workspace_resolver.py)
====================================================================================

Resolves the canonical project workspace path before command/tool execution:
- Evaluates user prompt, OS environment facts, and known directory roots.
- Fast heuristic shortcuts for absolute paths and existing directories.
- Pre-flight lightweight LLM call (cheapest/fastest model) for ambiguous/relative requests.
- Automatic typo correction (e.g. 'imgprc/imgenc' -> 'imgprc/imenc') via sibling inspection.
- Dual-tier caching:
    * Task-scoped cache: Locked for the entire lifecycle of an active task.
    * Turn-scoped cache: TTL of N turns for casual conversational commands.
- Explicit agent override tool: 'workspace.set_path'.
- Eliminates stdout regex scraping and random CWD drift.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Common top-level scan directories relative to USERPROFILE
COMMON_USER_ROOTS = [
    "Documents",
    "Downloads",
    "Desktop",
    "Projects",
    "source",
    "repos",
    "workspace",
]


# Common stop-words that should not be used as isolated folder search tokens
GENERIC_QUERY_WORDS = {
    "project", "finish", "complete", "implement", "build", "create", "start",
    "debug", "script", "folder", "directory", "dir", "path", "test", "tests",
    "work", "workspace", "code", "repo", "repository", "task", "subtask"
}

def _get_user_home() -> Path:
    """Returns canonical user home directory across Windows and POSIX."""
    home_env = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if home_env and os.path.isdir(home_env):
        return Path(home_env).resolve()
    return Path(os.path.expanduser("~")).resolve()


class WorkspaceResolver:
    """
    Intelligent workspace path resolver with fast heuristics,
    cheap-model grounding, and dual-tier caching.
    """

    def __init__(self, llm_client: Optional[Any] = None) -> None:
        self.user_home: Path = _get_user_home()
        self.llm_client = llm_client

    def _find_candidate_folders(self, query: str, max_depth: int = 3) -> List[Path]:
        """Scans common user roots for folders matching query tokens."""
        raw_tokens = [t.lower() for t in re.findall(r"\b[A-Za-z0-9_-]{3,}\b", query)]
        # Filter out generic stop-words unless all tokens are generic
        tokens = [t for t in raw_tokens if t not in GENERIC_QUERY_WORDS]
        if not tokens:
            tokens = raw_tokens
        if not tokens:
            return []

        candidates: List[Path] = []
        scan_roots = [self.user_home / r for r in COMMON_USER_ROOTS if (self.user_home / r).is_dir()]

        for root in scan_roots:
            try:
                for current_root, dirnames, _ in os.walk(str(root)):
                    # Avoid scanning deep into vendor or cache directories
                    dirnames[:] = [
                        d for d in dirnames
                        if not d.startswith(".")
                        and d.lower() not in (
                            "node_modules", "venv", ".venv", "__pycache__", "build", "dist",
                            "appdata", "local settings", "program files"
                        )
                    ]
                    c_path = Path(current_root)
                    base_lower = c_path.name.lower()

                    # Match against tokens
                    if any(tok in base_lower for tok in tokens):
                        candidates.append(c_path)
                        if len(candidates) >= 15:
                            return candidates

                    # Depth limit relative to root
                    rel_parts = c_path.relative_to(root).parts
                    if len(rel_parts) >= max_depth:
                        dirnames.clear()
            except (PermissionError, OSError):
                continue

        return candidates

    def _correct_typo_in_path(self, target_path: Path) -> Optional[Path]:
        """
        If target_path does not exist, checks parent directory siblings
        to correct typos (e.g. 'Downloads/imgprc/imgenc' -> 'Downloads/imgprc/imenc').
        """
        if target_path.exists():
            return target_path

        parent = target_path.parent
        target_name = target_path.name.lower()

        # Check if parent exists
        if parent.exists() and parent.is_dir():
            try:
                siblings = [d.name for d in parent.iterdir() if d.is_dir()]
                matches = difflib.get_close_matches(target_name, siblings, n=1, cutoff=0.55)
                if matches:
                    corrected = parent / matches[0]
                    if corrected.is_dir():
                        logger.info(f"[WorkspaceResolver] Corrected folder typo: '{target_path.name}' -> '{matches[0]}'")
                        return corrected
            except Exception:
                pass

        # Also check grandparent if parent had a typo
        grandparent = parent.parent
        if grandparent.exists() and grandparent.is_dir():
            try:
                parent_name = parent.name.lower()
                parent_siblings = [d.name for d in grandparent.iterdir() if d.is_dir()]
                p_matches = difflib.get_close_matches(parent_name, parent_siblings, n=1, cutoff=0.55)
                if p_matches:
                    corr_parent = grandparent / p_matches[0]
                    corr_target = corr_parent / target_path.name
                    if corr_target.is_dir():
                        return corr_target
                    # Check child of corrected parent
                    c_siblings = [d.name for d in corr_parent.iterdir() if d.is_dir()]
                    c_matches = difflib.get_close_matches(target_name, c_siblings, n=1, cutoff=0.55)
                    if c_matches:
                        return corr_parent / c_matches[0]
            except Exception:
                pass

        return None

    def resolve_workspace(
        self,
        user_input: str,
        current_workspace: str = "",
        working_memory: Optional[Any] = None,
        force_refresh: bool = False,
    ) -> Tuple[str, str]:
        """
        Determines the canonical project workspace directory.
        
        Returns:
            Tuple of (resolved_absolute_path, scope) where scope is "task" or "turn".
        """
        clean_input = str(user_input).strip()

        # 1. Check existing cache in working_memory if valid and not forcing refresh
        if not force_refresh and working_memory:
            cached = working_memory.get_workspace_path()
            if cached and os.path.isdir(cached):
                return cached, working_memory.workspace_cache_scope

        # If current_workspace is provided and valid
        if not force_refresh and current_workspace and os.path.isdir(current_workspace):
            return os.path.abspath(current_workspace), "task"

        # 2. Fast Heuristic: Look for explicit path pattern in user input
        # Matches: C:\...\foo, C:/.../foo, ~/..., Downloads/foo, imgprc/imenc
        path_matches = re.findall(
            r'(?:[a-zA-Z]:[\\/][\w\-./\\]+)|(?:~[\\/][\w\-./\\]+)|(?:[\w\-]+[\\/][\w\-./\\]+)',
            clean_input
        )

        # Roots to test for relative paths
        test_roots = [
            self.user_home / "Downloads",
            self.user_home / "Documents",
            self.user_home / "Desktop",
            self.user_home / "Projects",
            self.user_home,
        ]

        for raw_match in path_matches:
            raw_clean = raw_match.strip().strip("'\"").rstrip(".,;")
            candidate_paths: List[Path] = []

            if raw_clean.startswith("~"):
                candidate_paths.append((self.user_home / raw_clean[2:]).resolve())
            elif os.path.isabs(raw_clean):
                candidate_paths.append(Path(raw_clean).resolve())
            else:
                for root in test_roots:
                    if root.is_dir():
                        candidate_paths.append((root / raw_clean).resolve())

            for cand in candidate_paths:
                corrected = self._correct_typo_in_path(cand)
                if corrected and corrected.is_dir():
                    logger.info(f"[WorkspaceResolver] Fast heuristic resolved: '{corrected}'")
                    scope = "task" if any(w in clean_input.lower() for w in ("finish", "implement", "build", "create", "fix", "code", "run", "project", "work on")) else "turn"
                    return str(corrected), scope

        # 3. Candidate scan in common roots
        candidates = self._find_candidate_folders(clean_input)
        if candidates:
            # If exactly one strong match or first matches high-confidence query words
            first_cand = candidates[0]
            if first_cand.is_dir():
                scope = "task" if any(w in clean_input.lower() for w in ("finish", "implement", "build", "create", "fix", "code", "project")) else "turn"
                return str(first_cand), scope

        # 4. Pre-Flight Lightweight LLM Call (if llm_client available)
        if self.llm_client:
            try:
                resolved_llm = self._resolve_via_fast_llm(clean_input, candidates)
                if resolved_llm and os.path.isdir(resolved_llm):
                    scope = "task" if any(w in clean_input.lower() for w in ("finish", "implement", "build", "create", "fix", "code", "project")) else "turn"
                    return resolved_llm, scope
            except Exception as e:
                logger.warning(f"[WorkspaceResolver] Fast LLM resolution fallback: {e}")

        # 5. Default Fallback: user home or current project directory
        default_dir = os.getcwd()
        return default_dir, "turn"

    def _resolve_via_fast_llm(self, user_input: str, candidate_paths: List[Path]) -> Optional[str]:
        """Invokes fast/cheapest LLM to identify the intended project path."""
        candidates_str = "\n".join(f"- {str(p)}" for p in candidate_paths[:8]) or "None found"
        facts = (
            f"OS: {'Windows' if os.name == 'nt' else 'POSIX'}, "
            f"User Home: {str(self.user_home)}, "
            f"CWD: {os.getcwd()}"
        )

        prompt = (
            f"The user wants to work on a task.\n\n"
            f"User Request: {user_input}\n"
            f"Environment Facts: {facts}\n"
            f"Discovered Candidate Folders:\n{candidates_str}\n\n"
            f"Determine the exact, absolute directory path where this project lives. "
            f"If there is a small typo in the folder name (e.g. imgenc instead of imenc), correct it to the actual path. "
            f"Return ONLY valid JSON with keys: 'path' (string) and 'confidence' (float). "
            f"Do not include any explanation or extra text."
        )

        system_prompt = "You are a fast workspace path resolution engine. Output ONLY valid JSON with 'path' and 'confidence'."

        # Call llm_client
        if hasattr(self.llm_client, "complete_json"):
            res = self.llm_client.complete_json(prompt=prompt, system_prompt=system_prompt)
            if isinstance(res, dict) and "path" in res:
                cand_path = Path(res["path"]).resolve()
                corrected = self._correct_typo_in_path(cand_path)
                if corrected and corrected.is_dir():
                    return str(corrected)
                if cand_path.is_dir():
                    return str(cand_path)

        elif hasattr(self.llm_client, "complete"):
            res_str = self.llm_client.complete(prompt=prompt, system_prompt=system_prompt)
            match = re.search(r'\{.*\}', res_str, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if "path" in data:
                    cand_path = Path(data["path"]).resolve()
                    corrected = self._correct_typo_in_path(cand_path)
                    if corrected and corrected.is_dir():
                        return str(corrected)
                    if cand_path.is_dir():
                        return str(cand_path)

        return None


# ============================================================================
# Tool Definition: workspace.set_path
# ============================================================================

def make_workspace_set_path_tool_def() -> Dict[str, Any]:
    """Returns the function calling schema for the workspace.set_path tool."""
    return {
        "type": "function",
        "function": {
            "name": "workspace.set_path",
            "description": "Explicitly set or change the active project workspace directory for all subsequent tools and shell commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The absolute or relative directory path to anchor as the project workspace root."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation why the workspace path is being changed or anchored."
                    }
                },
                "required": ["path"]
            }
        }
    }


def execute_workspace_set_path(
    args: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    working_memory: Optional[Any] = None
) -> Dict[str, Any]:
    """Executor implementation for workspace.set_path."""
    raw_path = str(args.get("path", "")).strip().strip("'\"")
    if not raw_path:
        return {"ok": False, "error": "No path provided to workspace.set_path"}

    p = Path(raw_path).expanduser().resolve()
    if not p.exists():
        # Try typo correction
        resolver = WorkspaceResolver()
        corrected = resolver._correct_typo_in_path(p)
        if corrected and corrected.is_dir():
            p = corrected
        else:
            return {"ok": False, "error": f"Directory does not exist: {raw_path}"}

    if not p.is_dir():
        return {"ok": False, "error": f"Path is a file, not a directory: {p}"}

    abs_path_str = str(p)

    # 1. Update working memory
    wm = working_memory or (context.get("working_memory") if context else None)
    if wm and hasattr(wm, "set_workspace_path"):
        wm.set_workspace_path(abs_path_str, scope="task")

    # 2. Update context dictionary
    if context:
        context["workspace_dir"] = abs_path_str

    logger.info(f"[WorkspaceResolver] Explicitly anchored workspace to: {abs_path_str}")
    return {
        "ok": True,
        "workspace_path": abs_path_str,
        "message": f"Successfully anchored project workspace to: {abs_path_str}"
    }
