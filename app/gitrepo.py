"""Git plumbing for the vault clone on the server.

The server's clone is never edited by hand. The only local changes are the
append-only writes, and those are committed and pushed immediately, inside
the same lock the periodic pull uses.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}


class GitError(RuntimeError):
    pass


class WriteConflict(GitError):
    pass


def _redact(s: str) -> str:
    return re.sub(r"(https?://)[^@/\s]+@", r"\1***@", s)


class GitRepo:
    def __init__(self, path: Path, url: str, branch: str, author_name: str, author_email: str):
        self.path = path
        self.url = url
        self.branch = branch
        self.author_name = author_name
        self.author_email = author_email

    def _git(self, *args: str, check: bool = True, cwd: Path | None = None) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=cwd or self.path, capture_output=True, text=True, timeout=120,
            env=_ENV,
        )
        if check and proc.returncode != 0:
            raise GitError(_redact(f"git {' '.join(args[:2])} failed: {proc.stderr.strip() or proc.stdout.strip()}"))
        return proc.stdout.strip()

    # ---- setup -----------------------------------------------------------
    def ensure_clone(self) -> None:
        if not self.url:
            if (self.path / ".git").exists():
                return
            raise GitError("VAULT_REPO_URL is not set")
        if not (self.path / ".git").exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            log.info("cloning vault into %s", self.path)
            self._git("clone", "--branch", self.branch, self.url, str(self.path), cwd=self.path.parent)
        else:
            self._git("remote", "set-url", "origin", self.url)
        self._git("config", "user.name", self.author_name)
        self._git("config", "user.email", self.author_email)
        self._git("config", "pull.rebase", "true")
        self._git("config", "core.quotepath", "false")

    def head(self) -> str:
        return self._git("rev-parse", "HEAD")

    # ---- sync ------------------------------------------------------------
    def pull(self) -> bool:
        """Bring the clone up to date with origin. Returns True if HEAD moved."""
        before = self.head()
        if self._git("status", "--porcelain"):
            # Should never happen (writes commit atomically). Don't let a stray
            # change block syncing forever.
            log.warning("vault clone had uncommitted changes; discarding them")
            self._git("reset", "--hard", "HEAD")
            self._git("clean", "-fd")
        self._git("fetch", "origin", self.branch)
        remote = f"origin/{self.branch}"
        ahead = int(self._git("rev-list", "--count", f"{remote}..HEAD") or 0)
        if ahead:
            # A previous push failed. Replay our commits on top and push again.
            self._rebase_or_abort(remote)
            self._git("push", "origin", f"HEAD:{self.branch}")
        else:
            self._git("merge", "--ff-only", remote)
        return self.head() != before

    def _rebase_or_abort(self, remote: str) -> None:
        try:
            self._git("rebase", remote)
        except GitError as e:
            self._git("rebase", "--abort", check=False)
            self._git("reset", "--hard", remote)
            raise WriteConflict(
                "Your laptop changed the same lines at the same time; the server's change was dropped. "
                f"Details: {e}"
            ) from e

    def commit_and_push(self, rel_paths: list[str], message: str, attempts: int = 3) -> str:
        self._git("add", "--", *rel_paths)
        if not self._git("status", "--porcelain", "--", *rel_paths):
            return self.head()
        self._git("commit", "-m", message)
        remote = f"origin/{self.branch}"
        for i in range(attempts):
            proc = subprocess.run(
                ["git", "push", "origin", f"HEAD:{self.branch}"], cwd=self.path,
                capture_output=True, text=True, timeout=120,
                env=_ENV,
            )
            if proc.returncode == 0:
                return self.head()
            log.info("push rejected (attempt %s), rebasing: %s", i + 1, _redact(proc.stderr.strip()))
            self._git("fetch", "origin", self.branch)
            self._rebase_or_abort(remote)
        raise GitError("push kept failing; the commit is kept locally and will be pushed on the next sync")

    # ---- history ---------------------------------------------------------
    def history(self, rel_path: str, limit: int = 20) -> list[dict]:
        out = self._git(
            "log", f"-n{limit}", "--follow", "--format=%h%x1f%aI%x1f%an%x1f%s", "--shortstat", "--", rel_path,
        )
        entries: list[dict] = []
        for line in out.splitlines():
            if "\x1f" in line:
                h, date, author, subject = line.split("\x1f", 3)
                entries.append({"commit": h, "date": date, "author": author, "message": subject})
            elif line.strip() and entries:
                entries[-1]["change"] = line.strip()
        return entries
