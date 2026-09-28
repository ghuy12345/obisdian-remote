"""The Brain: keeps the index in sync with the vault and answers every query.

REST and MCP are both thin layers over this class.
"""
from __future__ import annotations

import hashlib
import logging
import os
import posixpath
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .config import Settings
from .embeddings import Embedder, EmbeddingError
from .gitrepo import GitError, GitRepo
from .graph import VaultGraph
from .parser import parse_note
from .store import Store, embed_text

log = logging.getLogger(__name__)


class NoteNotFound(LookupError):
    def __init__(self, name: str, suggestions: list[str]):
        self.name, self.suggestions = name, suggestions
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        super().__init__(f"No note called '{name}'.{hint}")


class WriteRejected(ValueError):
    pass


_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def _norm_heading(h: str) -> str:
    return re.sub(r"\s+", " ", h.strip().lstrip("#").strip()).lower()


def _find_section(lines: list[str], heading: str) -> tuple[int, int]:
    """(index of heading line, index where the section ends). Ignores headings in code blocks."""
    want = _norm_heading(heading)
    found: list[tuple[int, int]] = []
    all_headings: list[str] = []
    in_code = False
    for i, line in enumerate(lines):
        if line.strip().startswith(("```", "~~~")):
            in_code = not in_code
            continue
        m = None if in_code else _HEADING.match(line)
        if m:
            all_headings.append(m.group(2))
            if _norm_heading(m.group(2)) == want:
                found.append((i, len(m.group(1))))
    if not found:
        listed = ", ".join(all_headings[:30]) or "none"
        raise WriteRejected(f"No heading '{heading}' in this note. Headings: {listed}")
    if len(found) > 1:
        raise WriteRejected(f"The heading '{heading}' appears {len(found)} times; use vault_edit_note instead.")
    start, level = found[0]
    in_code = False
    for j in range(start + 1, len(lines)):
        if lines[j].strip().startswith(("```", "~~~")):
            in_code = not in_code
            continue
        m = None if in_code else _HEADING.match(lines[j])
        if m and len(m.group(1)) <= level:
            return start, j
    return start, len(lines)


def _diff_size(old: str, new: str) -> tuple[int, int]:
    """(chars removed, chars added), ignoring the shared prefix and suffix."""
    p = 0
    while p < min(len(old), len(new)) and old[p] == new[p]:
        p += 1
    s = 0
    while s < min(len(old), len(new)) - p and old[-1 - s] == new[-1 - s]:
        s += 1
    return len(old) - p - s, len(new) - p - s


@dataclass
class Vectors:
    meta: list[dict]
    mat: np.ndarray | None


class Brain:
    def __init__(self, settings: Settings):
        self.s = settings
        self.store = Store(settings.db_path)
        self.repo = GitRepo(settings.vault_dir, settings.vault_repo_url, settings.vault_branch,
                            settings.git_author_name, settings.git_author_email)
        self.embedder = (
            Embedder(settings.embeddings_base_url, settings.embeddings_api_key,
                     settings.embeddings_model, settings.embeddings_batch_size)
            if settings.embeddings_enabled else None
        )
        self.vault_lock = threading.RLock()    # git + files
        self.embed_lock = threading.Lock()
        self.graph = VaultGraph([], [])
        self.vectors = Vectors([], None)
        self.last_sync: float | None = None
        self.last_error: str | None = None
        self.last_embed_error: str | None = None
        self.ready = False

    # =====================================================================
    # Sync + indexing
    # =====================================================================
    def startup(self) -> None:
        with self.vault_lock:
            self.repo.ensure_clone()
            self._index_vault()
            self._rebuild_graph()
        self._load_vectors()
        self.ready = True
        self.embed_pending()

    def sync(self) -> dict:
        """Pull, reindex what changed, embed new chunks. Safe to call any time."""
        stats: dict = {"pulled": False}
        try:
            with self.vault_lock:
                if self.s.vault_repo_url or (self.s.vault_dir / ".git").exists():
                    stats["pulled"] = self.repo.pull()
                stats.update(self._index_vault())
                if stats["pulled"] or any(stats.get(k) for k in ("added", "updated", "deleted")):
                    self._rebuild_graph()
            self.last_sync = time.time()
            self.last_error = None
        except GitError as e:
            self.last_error = str(e)
            log.error("sync failed: %s", e)
            stats["error"] = str(e)
        stats["embedded"] = self.embed_pending()
        return stats

    def _walk(self):
        root = self.s.vault_dir
        ignore = self.s.ignore_dir_set
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ignore and not d.startswith(".")]
            for f in filenames:
                if f.endswith(".md"):
                    full = Path(dirpath) / f
                    yield full.relative_to(root).as_posix(), full

    def _index_vault(self) -> dict:
        known = self.store.file_states()
        seen: set[str] = set()
        added = updated = 0
        for rel, full in self._walk():
            seen.add(rel)
            st = full.stat()
            prev = known.get(rel)
            if prev and prev[0] == st.st_mtime and prev[1] == st.st_size:
                continue
            text = full.read_text(encoding="utf-8", errors="replace")
            h = hashlib.sha256(text.encode()).hexdigest()
            if prev and prev[2] == h:
                self.store.touch(rel, st.st_mtime, st.st_size)
                continue
            self._index_text(rel, text, st.st_mtime, st.st_size)
            if prev:
                updated += 1
            else:
                added += 1
        deleted = [p for p in known if p not in seen]
        for p in deleted:
            self.store.delete_note(p)
        if added or updated or deleted:
            log.info("indexed: %s added, %s updated, %s deleted", added, updated, len(deleted))
        return {"added": added, "updated": updated, "deleted": len(deleted)}

    def _index_text(self, rel: str, text: str, mtime: float, size: int) -> None:
        note = parse_note(rel, text, self.s.chunk_chars, self.s.chunk_overlap)
        hashes = [
            hashlib.sha256(embed_text(note.title, c.heading, c.text).encode()).hexdigest() for c in note.chunks
        ]
        self.store.upsert_note(note, mtime, size, hashes)

    def _index_file(self, rel: str) -> None:
        full = self.s.vault_dir / rel
        st = full.stat()
        self._index_text(rel, full.read_text(encoding="utf-8", errors="replace"), st.st_mtime, st.st_size)

    def _rebuild_graph(self) -> None:
        self.graph = VaultGraph(self.store.all_notes(), self.store.all_links())

    def embed_pending(self) -> int:
        if not self.embedder or not self.embed_lock.acquire(blocking=False):
            return 0
        try:
            pending = self.store.chunks_missing_embeddings(self.s.embeddings_model)
            done = 0
            batch = self.s.embeddings_batch_size * 4
            for i in range(0, len(pending), batch):
                part = pending[i : i + batch]
                vecs = self.embedder.embed([t for _, t in part])
                self.store.save_embeddings(self.s.embeddings_model, [(h, v) for (h, _), v in zip(part, vecs)])
                done += len(part)
            if done:
                self.store.prune_embeddings(self.s.embeddings_model)
                log.info("embedded %s chunks", done)
            if done or self.vectors.mat is None:
                self._load_vectors()
            self.last_embed_error = None
            return done
        except EmbeddingError as e:
            self.last_embed_error = str(e)
            log.error("embedding failed (will retry next sync): %s", e)
            self._load_vectors()
            return 0
        finally:
            self.embed_lock.release()

    def _load_vectors(self) -> None:
        meta, mat = self.store.load_chunk_vectors(self.s.embeddings_model)
        self.vectors = Vectors(meta, mat)

    # =====================================================================
    # Lookup helpers
    # =====================================================================
    def resolve(self, name: str) -> str:
        name = name.strip()
        if name.startswith("[[") and name.endswith("]]"):
            name = name[2:-2].split("|")[0].split("#")[0]
        hits = self.graph.resolver.find(name)
        if hits:
            return hits[0]
        sugg = [self.graph.title(r["path"]) for r in self.store.fts(name, 5)]
        raise NoteNotFound(name, sugg)

    def _brief(self, path: str) -> dict:
        n = self.graph.notes.get(path, {})
        return {"path": path, "title": self.graph.title(path), "tags": n.get("tags", [])}

    def _read_text(self, path: str) -> str:
        return (self.s.vault_dir / path).read_text(encoding="utf-8", errors="replace")

    # =====================================================================
    # Read queries
    # =====================================================================
    def read_note(self, name: str, max_chars: int = 20000) -> dict:
        path = self.resolve(name)
        row = self.store.note_row(path) or {}
        text = self._read_text(path)
        return {
            "path": path,
            "title": self.graph.title(path),
            "frontmatter": row.get("frontmatter", {}),
            "tags": row.get("tags", []),
            "aliases": row.get("aliases", []),
            "word_count": row.get("word_count"),
            "content": text[:max_chars],
            "truncated": len(text) > max_chars,
            "outlinks": self.graph.outlinks(path),
            "backlink_count": self.graph.g.in_degree(path),
        }

    def list_notes(self, folder: str | None = None, tag: str | None = None,
                   sort: str = "recent", limit: int = 50) -> list[dict]:
        notes = self.graph.notes.values()
        if folder:
            f = folder.strip("/") + "/"
            notes = [n for n in notes if n["path"].startswith(f)]
        if tag:
            t = tag.lstrip("#").lower()
            notes = [n for n in notes if any(x.lower() == t or x.lower().startswith(t + "/") for x in n["tags"])]
        key = {"recent": lambda n: -n["mtime"], "title": lambda n: n["title"].lower(),
               "size": lambda n: -n["word_count"]}.get(sort, lambda n: -n["mtime"])
        return [
            {"path": n["path"], "title": n["title"], "tags": n["tags"], "words": n["word_count"],
             "modified": time.strftime("%Y-%m-%d %H:%M", time.gmtime(n["mtime"]))}
            for n in sorted(notes, key=key)[:limit]
        ]

    # ---- search ---------------------------------------------------------
    def _semantic_chunks(self, text: str) -> np.ndarray | None:
        if not self.embedder or self.vectors.mat is None:
            return None
        q = self.embedder.embed([text])[0]
        q = q / (np.linalg.norm(q) or 1.0)
        return self.vectors.mat @ q

    def _aggregate(self, scores: np.ndarray, limit: int, exclude: set[str] = frozenset(),
                   folder: str | None = None) -> list[dict]:
        best: dict[str, tuple[float, int]] = {}
        for i in np.argsort(-scores)[: max(limit * 20, 200)]:
            p = self.vectors.meta[i]["path"]
            if p in exclude or (folder and not p.startswith(folder.strip("/") + "/")):
                continue
            if p not in best:
                best[p] = (float(scores[i]), int(i))
            if len(best) >= limit:
                break
        out = []
        for p, (score, i) in best.items():
            m = self.vectors.meta[i]
            out.append({**self._brief(p), "score": round(score, 4), "section": m["heading"],
                        "excerpt": m["text"][:400]})
        return out

    def search(self, query: str, mode: str = "hybrid", limit: int = 10, folder: str | None = None) -> dict:
        notice = None
        kw = []
        if mode in ("keyword", "hybrid"):
            kw = self.store.fts(query, limit * 3)
            if folder:
                kw = [r for r in kw if r["path"].startswith(folder.strip("/") + "/")]
        sem = []
        if mode in ("semantic", "hybrid"):
            try:
                scores = self._semantic_chunks(query)
            except EmbeddingError as e:
                scores, notice = None, f"Semantic search unavailable right now ({e}); keyword results only."
            if scores is None and notice is None:
                notice = "Semantic search is not configured or not indexed yet; keyword results only."
            if scores is not None:
                sem = self._aggregate(scores, limit * 3, folder=folder)

        # Reciprocal rank fusion: robust, no score calibration needed.
        fused: dict[str, dict] = {}
        for rank, r in enumerate(kw):
            d = fused.setdefault(r["path"], {**self._brief(r["path"]), "rrf": 0.0, "matched": []})
            d["rrf"] += 1 / (60 + rank)
            d["matched"].append("keyword")
            d.setdefault("excerpt", r["snippet"])
        for rank, r in enumerate(sem):
            d = fused.setdefault(r["path"], {**self._brief(r["path"]), "rrf": 0.0, "matched": []})
            d["rrf"] += 1 / (60 + rank)
            d["matched"].append("semantic")
            d["section"] = r["section"]
            if "excerpt" not in d or mode == "semantic":
                d["excerpt"] = r["excerpt"]
        results = sorted(fused.values(), key=lambda x: -x["rrf"])[:limit]
        for r in results:
            r["score"] = round(r.pop("rrf") * 1000, 2)
        return {"query": query, "mode": mode, "results": results, **({"notice": notice} if notice else {})}

    def related(self, name: str, limit: int = 10, include_linked: bool = False) -> dict:
        """Notes semantically close to this one. By default excludes notes already linked
        either way, so what's left are connections you haven't made yet."""
        path = self.resolve(name)
        if not self.embedder or self.vectors.mat is None:
            return {"note": self.graph.title(path), "results": [],
                    "notice": "Semantic search is not configured or not indexed yet."}
        idx = [i for i, m in enumerate(self.vectors.meta) if m["path"] == path]
        if not idx:
            return {"note": self.graph.title(path), "results": [], "notice": "This note is not embedded yet."}
        centroid = self.vectors.mat[idx].mean(axis=0)
        centroid /= np.linalg.norm(centroid) or 1.0
        exclude = {path}
        if not include_linked:
            exclude |= set(self.graph.ug.neighbors(path))
        return {"note": self.graph.title(path),
                "results": self._aggregate(self.vectors.mat @ centroid, limit, exclude=exclude)}

    # ---- graph ----------------------------------------------------------
    def backlinks(self, name: str) -> dict:
        path = self.resolve(name)
        return {"note": self.graph.title(path), "path": path, "backlinks": self.graph.backlinks(path)}

    def neighbors(self, name: str, hops: int = 2, limit: int = 50) -> dict:
        path = self.resolve(name)
        return {"note": self.graph.title(path), "hops": hops, "neighbors": self.graph.neighbors(path, hops, limit)}

    def find_path(self, a: str, b: str) -> dict:
        pa, pb = self.resolve(a), self.resolve(b)
        p = self.graph.shortest_path(pa, pb)
        if p is None:
            return {"from": self.graph.title(pa), "to": self.graph.title(pb), "connected": False}
        return {"from": self.graph.title(pa), "to": self.graph.title(pb), "connected": True,
                "length": len(p["steps"]), **p}

    def hubs(self, limit: int = 20, folder: str | None = None) -> list[dict]:
        return self.graph.hubs(limit, folder)

    def clusters(self, min_size: int = 3, limit: int = 20) -> list[dict]:
        return [{k: v for k, v in c.items() if k != "paths"} for c in self.graph.clusters(min_size)[:limit]]

    def gaps(self, limit: int = 30) -> dict:
        """Unwritten notes you keep linking to, plus notes nothing links to."""
        return {"unresolved_links": self.graph.unresolved_summary(limit),
                "orphans": [self.graph.title(p) for p in self.graph.orphans()[:limit]]}

    def history(self, name: str, limit: int = 20) -> dict:
        path = self.resolve(name)
        with self.vault_lock:
            return {"note": self.graph.title(path), "path": path, "commits": self.repo.history(path, limit)}

    # ---- the one-call context tool -------------------------------------
    def get_context(self, topic: str, max_chars: int = 6000) -> dict:
        focus = None
        matches: list[dict] = []
        try:
            focus = self.resolve(topic)
        except NoteNotFound:
            res = self.search(topic, "hybrid", 6)
            matches = res["results"]
            if matches:
                focus = matches[0]["path"]
        if not focus:
            return {"topic": topic, "found": False,
                    "message": "Nothing in the vault matches this topic yet."}

        text = self._read_text(focus)
        g = self.graph
        linked = set(g.ug.neighbors(focus))
        two_hop = [n for n in g.neighbors(focus, 2, 40) if n["hops"] == 2][:12]
        pr = g.pagerank()
        cluster_paths = [p for p in g.cluster_of(focus) if p != focus and p not in linked]
        cluster = [g.title(p) for p in sorted(cluster_paths, key=lambda p: -pr.get(p, 0))]
        out = {
            "topic": topic,
            "found": True,
            "note": {"path": focus, "title": g.title(focus), "tags": g.notes[focus]["tags"],
                     "content": text[:max_chars], "truncated": len(text) > max_chars},
            "links_to": g.outlinks(focus),
            "linked_from": g.backlinks(focus),
            "two_hops_away": two_hop,
            "same_cluster_not_linked": cluster[:10],
            "unwritten_links": g.unresolved_from(focus),
            "hub_score": round(pr.get(focus, 0) * 1000, 3),
        }
        rel = self.related(focus, 8)
        out["semantically_related_not_linked"] = rel.get("results", [])
        if rel.get("notice"):
            out["notice"] = rel["notice"]
        if len(matches) > 1:
            out["other_matches"] = [{"title": m["title"], "path": m["path"]} for m in matches[1:]]
        return out

    # =====================================================================
    # Creating and appending
    # =====================================================================
    def _check_path(self, path: str, allowed: list[str]) -> str:
        p = posixpath.normpath(path.strip().lstrip("/"))
        if not p.endswith(".md"):
            p += ".md"
        if p.startswith("..") or "/../" in p or p.startswith("/"):
            raise WriteRejected("Path must stay inside the vault.")
        parts = p.split("/")
        if any(x in self.s.ignore_dir_set or x.startswith(".") for x in parts[:-1]):
            raise WriteRejected("Can't write into hidden or ignored folders.")
        if allowed and not any(p.startswith(d) for d in allowed):
            raise WriteRejected(f"Writes are only allowed in: {', '.join(allowed)}")
        return p

    def _guard(self, content: str) -> None:
        if not self.s.writes_enabled:
            raise WriteRejected("Writes are disabled on this server (WRITES_ENABLED=false).")
        if not content.strip():
            raise WriteRejected("Content is empty.")
        if len(content) > self.s.max_write_chars:
            raise WriteRejected(f"Content is longer than {self.s.max_write_chars} characters.")

    def _after_write(self, rel: str) -> None:
        self._index_file(rel)
        self._rebuild_graph()
        threading.Thread(target=self.embed_pending, daemon=True).start()

    def create_note(self, path: str, content: str, frontmatter: dict | None = None) -> dict:
        self._guard(content)
        rel = self._check_path(path, self.s.create_dirs)
        with self.vault_lock:
            self.repo.pull()
            full = self.s.vault_dir / rel
            if full.exists():
                raise WriteRejected(f"'{rel}' already exists. Use append_to_note instead.")
            full.parent.mkdir(parents=True, exist_ok=True)
            body = content if content.endswith("\n") else content + "\n"
            if frontmatter:
                fm = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()
                body = f"---\n{fm}\n---\n\n{body}"
            try:
                full.write_text(body, encoding="utf-8")
                commit = self.repo.commit_and_push([rel], f"vault-brain: create {rel}")
            except Exception:
                if full.exists() and not self.repo._git("ls-files", "--", rel):
                    full.unlink()
                raise
            self._after_write(rel)
        return {"created": rel, "title": posixpath.splitext(posixpath.basename(rel))[0], "commit": commit}

    def append_to_note(self, name: str, content: str, heading: str | None = None) -> dict:
        self._guard(content)
        with self.vault_lock:
            self.repo.pull()
            self._index_vault()
            self._rebuild_graph()
            rel = self._check_path(self.resolve(name), self.s.append_dirs)
            full = self.s.vault_dir / rel
            current = full.read_text(encoding="utf-8", errors="replace")
            addition = content.strip("\n") + "\n"
            if heading:
                addition = f"## {heading.strip().lstrip('#').strip()}\n\n{addition}"
            sep = "" if current.endswith("\n\n") or not current else ("\n" if current.endswith("\n") else "\n\n")
            try:
                with full.open("a", encoding="utf-8") as f:
                    f.write(sep + addition)
                commit = self.repo.commit_and_push([rel], f"vault-brain: append to {rel}")
            except Exception:
                self.repo._git("checkout", "--", rel, check=False)
                raise
            self._after_write(rel)
        return {"appended_to": rel, "title": self.graph.title(rel), "chars": len(addition), "commit": commit}

    # =====================================================================
    # In-place edits
    # =====================================================================
    def _modify(self, name: str, change, verb: str) -> dict:
        """Pull, apply change(current_text) -> new_text, commit, push. All inside the vault lock,
        so the edit always applies to the latest version from the laptop."""
        if not self.s.writes_enabled or not self.s.edits_enabled:
            raise WriteRejected("Editing existing notes is disabled on this server (EDITS_ENABLED=false).")
        with self.vault_lock:
            self.repo.pull()
            self._index_vault()
            self._rebuild_graph()
            rel = self._check_path(self.resolve(name), self.s.edit_dirs)
            full = self.s.vault_dir / rel
            current = full.read_text(encoding="utf-8", errors="replace")
            new = change(current)
            if new == current:
                return {"edited": rel, "title": self.graph.title(rel), "changed": False}
            removed, added = _diff_size(current, new)
            if removed > self.s.max_edit_delete_chars:
                raise WriteRejected(
                    f"This edit would remove {removed} characters, more than the {self.s.max_edit_delete_chars} "
                    "allowed in one edit. Make smaller edits."
                )
            try:
                full.write_text(new, encoding="utf-8")
                commit = self.repo.commit_and_push([rel], f"vault-brain: {verb} {rel}")
            except Exception:
                self.repo._git("checkout", "--", rel, check=False)
                raise
            self._after_write(rel)
        return {"edited": rel, "title": self.graph.title(rel), "changed": True,
                "chars_removed": removed, "chars_added": added, "commit": commit}

    def edit_note(self, name: str, old_text: str, new_text: str, replace_all: bool = False) -> dict:
        """Replace an exact piece of text. It must match exactly once unless replace_all."""
        if not old_text:
            raise WriteRejected("old_text is empty. Copy the exact text to change from vault_read_note.")
        if len(new_text) > self.s.max_write_chars:
            raise WriteRejected(f"new_text is longer than {self.s.max_write_chars} characters.")

        def change(current: str) -> str:
            n = current.count(old_text)
            if n == 0:
                raise WriteRejected(
                    "old_text was not found in the note. It must match exactly, including spaces and line "
                    "breaks. The note may also have changed on the laptop; read it again and retry."
                )
            if n > 1 and not replace_all:
                raise WriteRejected(
                    f"old_text appears {n} times. Include more surrounding text so it's unique, "
                    "or set replace_all to change every occurrence."
                )
            return current.replace(old_text, new_text)

        return self._modify(name, change, "edit")

    def update_section(self, name: str, heading: str, content: str, mode: str = "replace") -> dict:
        """Replace, append to, or prepend to the body under a heading. The heading line itself is kept."""
        if mode not in ("replace", "append", "prepend"):
            raise WriteRejected("mode must be replace, append or prepend.")
        if mode != "replace" and not content.strip():
            raise WriteRejected("Content is empty.")
        if len(content) > self.s.max_write_chars:
            raise WriteRejected(f"Content is longer than {self.s.max_write_chars} characters.")

        def change(current: str) -> str:
            lines = current.split("\n")
            start, end = _find_section(lines, heading)
            body = lines[start + 1 : end]
            while body and not body[-1].strip():  # keep the blank line(s) before the next heading
                body.pop()
            trailing = lines[start + 1 + len(body) : end]
            new_block = content.strip("\n").split("\n") if content.strip() else []
            if mode == "replace":
                body = ([""] + new_block) if new_block else []
            elif mode == "append":
                body = body + ([""] if body and body[-1].strip() else []) + new_block
            else:  # prepend
                lead = body[1:] if body and not body[0].strip() else body
                body = [""] + new_block + ([""] + lead if lead else [])
            if end < len(lines) and not trailing:
                trailing = [""]
            return "\n".join(lines[: start + 1] + body + trailing + lines[end:])

        return self._modify(name, change, f"update section '{heading}' in")

    # =====================================================================
    def status(self) -> dict:
        counts = self.store.counts()
        head = None
        try:
            head = self.repo.head()[:10]
        except GitError:
            pass
        return {
            "ready": self.ready,
            "head": head,
            "last_sync": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.last_sync)) if self.last_sync else None,
            "last_error": self.last_error,
            "embeddings": {
                "enabled": bool(self.embedder),
                "model": self.s.embeddings_model if self.embedder else None,
                "vectors_loaded": len(self.vectors.meta),
                "last_error": self.last_embed_error,
            },
            "graph": {"nodes": self.graph.g.number_of_nodes(), "edges": self.graph.g.number_of_edges(),
                      "unresolved_targets": len(self.graph.unresolved)},
            "writes_enabled": self.s.writes_enabled,
            "edits_enabled": self.s.writes_enabled and self.s.edits_enabled,
            **counts,
        }
