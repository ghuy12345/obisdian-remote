"""SQLite storage: notes, links, chunks, embedding cache, full-text index.

Everything here is derived from the vault and can be deleted and rebuilt at
any time. Git is the source of truth.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from .parser import ParsedNote

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS notes (
    path TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    hash TEXT NOT NULL,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    frontmatter TEXT NOT NULL,
    tags TEXT NOT NULL,
    aliases TEXT NOT NULL,
    headings TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    errors TEXT NOT NULL,
    indexed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS links (
    src TEXT NOT NULL,
    target TEXT NOT NULL,
    kind TEXT NOT NULL,
    heading TEXT,
    alias TEXT,
    context TEXT NOT NULL,
    section TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS links_src ON links(src);
CREATE TABLE IF NOT EXISTS chunks (
    path TEXT NOT NULL,
    ord INTEGER NOT NULL,
    heading TEXT NOT NULL,
    text TEXT NOT NULL,
    hash TEXT NOT NULL,
    PRIMARY KEY (path, ord)
);
CREATE INDEX IF NOT EXISTS chunks_hash ON chunks(hash);
CREATE TABLE IF NOT EXISTS embeddings (
    hash TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,
    PRIMARY KEY (hash, model)
);
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    path UNINDEXED, title, headings, body,
    tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    # ---- meta -----------------------------------------------------------
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self.lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    # ---- notes ----------------------------------------------------------
    def file_states(self) -> dict[str, tuple[float, int, str]]:
        with self.lock:
            rows = self.conn.execute("SELECT path, mtime, size, hash FROM notes").fetchall()
        return {r["path"]: (r["mtime"], r["size"], r["hash"]) for r in rows}

    def touch(self, path: str, mtime: float, size: int) -> None:
        with self.lock, self.conn:
            self.conn.execute("UPDATE notes SET mtime=?, size=? WHERE path=?", (mtime, size, path))

    def upsert_note(self, note: ParsedNote, mtime: float, size: int, chunk_hashes: list[str]) -> None:
        with self.lock, self.conn:
            c = self.conn
            self._delete(c, note.path)
            c.execute(
                "INSERT INTO notes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    note.path, note.title, note.content_hash, mtime, size,
                    json.dumps(note.frontmatter, ensure_ascii=False),
                    json.dumps(note.tags, ensure_ascii=False),
                    json.dumps(note.aliases, ensure_ascii=False),
                    json.dumps(note.headings, ensure_ascii=False),
                    note.word_count,
                    json.dumps(note.errors),
                    time.time(),
                ),
            )
            c.executemany(
                "INSERT INTO links VALUES (?,?,?,?,?,?,?)",
                [(note.path, l.target, l.kind, l.heading, l.alias, l.context, l.section) for l in note.links],
            )
            c.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?)",
                [(note.path, ch.ord, ch.heading, ch.text, h) for ch, h in zip(note.chunks, chunk_hashes)],
            )
            c.execute(
                "INSERT INTO notes_fts(path, title, headings, body) VALUES (?,?,?,?)",
                (note.path, note.title, " ".join(note.headings), note.body),
            )

    def delete_note(self, path: str) -> None:
        with self.lock, self.conn:
            self._delete(self.conn, path)

    @staticmethod
    def _delete(c: sqlite3.Connection, path: str) -> None:
        for table, col in (("notes", "path"), ("links", "src"), ("chunks", "path"), ("notes_fts", "path")):
            c.execute(f"DELETE FROM {table} WHERE {col}=?", (path,))

    def all_notes(self) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT path, title, tags, aliases, frontmatter, word_count, mtime FROM notes"
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("tags", "aliases", "frontmatter"):
                d[k] = json.loads(d[k])
            out.append(d)
        return out

    def note_row(self, path: str) -> dict | None:
        with self.lock:
            r = self.conn.execute("SELECT * FROM notes WHERE path=?", (path,)).fetchone()
        if not r:
            return None
        d = dict(r)
        for k in ("tags", "aliases", "frontmatter", "headings", "errors"):
            d[k] = json.loads(d[k])
        return d

    def all_links(self) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute("SELECT * FROM links").fetchall()]

    # ---- full text --------------------------------------------------------
    def fts(self, query: str, limit: int) -> list[dict]:
        terms = [t for t in "".join(ch if ch.isalnum() else " " for ch in query).split() if t]
        if not terms:
            return []
        # OR the terms together, each as a prefix match; bm25 ranks by relevance.
        match = " OR ".join(f'"{t}"*' for t in terms)
        with self.lock:
            rows = self.conn.execute(
                """SELECT path, bm25(notes_fts, 0, 10.0, 4.0, 1.0) AS score,
                          snippet(notes_fts, 3, '**', '**', ' … ', 24) AS snippet
                   FROM notes_fts WHERE notes_fts MATCH ? ORDER BY score LIMIT ?""",
                (match, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- chunks + embeddings ---------------------------------------------
    def chunks_missing_embeddings(self, model: str) -> list[tuple[str, str]]:
        """(hash, text) for chunks with no cached vector for this model."""
        with self.lock:
            rows = self.conn.execute(
                """SELECT DISTINCT c.hash, n.title, c.heading, c.text FROM chunks c
                   JOIN notes n ON n.path = c.path
                   LEFT JOIN embeddings e ON e.hash = c.hash AND e.model = ?
                   WHERE e.hash IS NULL""",
                (model,),
            ).fetchall()
        return [(r["hash"], embed_text(r["title"], r["heading"], r["text"])) for r in rows]

    def save_embeddings(self, model: str, items: list[tuple[str, np.ndarray]]) -> None:
        with self.lock, self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)",
                [(h, model, v.shape[0], v.astype(np.float32).tobytes()) for h, v in items],
            )

    def load_chunk_vectors(self, model: str) -> tuple[list[dict], np.ndarray | None]:
        with self.lock:
            rows = self.conn.execute(
                """SELECT c.path, c.ord, c.heading, c.text, e.vec FROM chunks c
                   JOIN embeddings e ON e.hash = c.hash AND e.model = ?
                   ORDER BY c.path, c.ord""",
                (model,),
            ).fetchall()
        if not rows:
            return [], None
        meta = [{"path": r["path"], "ord": r["ord"], "heading": r["heading"], "text": r["text"]} for r in rows]
        mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return meta, mat / norms

    def prune_embeddings(self, model: str) -> int:
        """Drop cached vectors no chunk uses any more."""
        with self.lock, self.conn:
            cur = self.conn.execute(
                "DELETE FROM embeddings WHERE model=? AND hash NOT IN (SELECT hash FROM chunks)", (model,)
            )
            return cur.rowcount

    def counts(self) -> dict:
        with self.lock:
            q = lambda s: self.conn.execute(s).fetchone()[0]  # noqa: E731
            return {
                "notes": q("SELECT COUNT(*) FROM notes"),
                "links": q("SELECT COUNT(*) FROM links"),
                "chunks": q("SELECT COUNT(*) FROM chunks"),
                "embedded_chunks": q(
                    "SELECT COUNT(DISTINCT c.hash) FROM chunks c JOIN embeddings e ON e.hash=c.hash"
                ),
            }


def embed_text(title: str, heading: str, text: str) -> str:
    header = title if not heading else f"{title} > {heading}"
    return f"{header}\n\n{text}"
