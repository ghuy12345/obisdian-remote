"""MCP tools. Each one is a thin async wrapper over a Brain method."""
from __future__ import annotations

from typing import Annotated, Callable, Literal

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field

from .engine import Brain

INSTRUCTIONS = """\
This server is Joshua's Obsidian vault (his second brain), indexed as a link graph
plus semantic embeddings. Start with vault_get_context for any topic: it returns
the best-matching note plus everything connected to it in one call. Use
vault_search when you need a list of matches, vault_read_note for full text, and
the graph tools (backlinks, neighbors, find_path, hubs, clusters, gaps) to explore
structure. Note names can be titles, aliases or paths. Writes are append-only:
create new notes or append to existing ones; nothing is ever overwritten or deleted.
Link other notes with [[Note Title]] wikilinks when writing so the graph stays connected.
"""

READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)

Note = Annotated[str, Field(description="Note title, alias or path, e.g. 'Hook Frameworks' or 'Creative Strategy/Hooks.md'")]

_brain: Brain | None = None


def set_brain(b: Brain) -> None:
    global _brain
    _brain = b


async def _run(fn: Callable, *args, **kwargs):
    if _brain is None or not _brain.ready:
        raise RuntimeError("The vault index is still starting up. Try again in a minute.")
    return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))


def build_mcp(allowed_hosts: list[str]) -> FastMCP:
    security = (
        TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=allowed_hosts,
                                  allowed_origins=[f"https://{h}" for h in allowed_hosts])
        if allowed_hosts else TransportSecuritySettings(enable_dns_rebinding_protection=False)
    )
    mcp = FastMCP("vault-brain", instructions=INSTRUCTIONS, stateless_http=True, json_response=True,
                  transport_security=security)

    @mcp.tool(annotations=READ)
    async def vault_get_context(
        topic: Annotated[str, Field(description="A note name or a free-text topic, e.g. 'awareness levels' or 'Men Mansion'")],
        max_chars: Annotated[int, Field(ge=500, le=40000, description="Max characters of the focus note's content")] = 6000,
    ) -> dict:
        """Everything the vault knows about a topic in one call: the best-matching note's content,
        what it links to, what links to it (with the sentence each link sits in), notes two hops away,
        notes in the same cluster, and semantically similar notes that aren't linked yet.
        Use this first whenever a topic comes up."""
        return await _run(_brain.get_context, topic, max_chars)

    @mcp.tool(annotations=READ)
    async def vault_search(
        query: Annotated[str, Field(description="What to look for, in natural language or keywords")],
        mode: Annotated[Literal["hybrid", "keyword", "semantic"], Field(description="hybrid combines keyword and meaning (default)")] = "hybrid",
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
        folder: Annotated[str | None, Field(description="Only search inside this folder, e.g. 'Creative Strategy'")] = None,
    ) -> dict:
        """Search the vault. Returns matching notes with the most relevant excerpt from each."""
        return await _run(_brain.search, query, mode, limit, folder)

    @mcp.tool(annotations=READ)
    async def vault_read_note(
        note: Note,
        max_chars: Annotated[int, Field(ge=500, le=100000)] = 20000,
    ) -> dict:
        """Full content of one note plus its frontmatter, tags and outgoing links."""
        return await _run(_brain.read_note, note, max_chars)

    @mcp.tool(annotations=READ)
    async def vault_list_notes(
        folder: Annotated[str | None, Field(description="Folder prefix to list")] = None,
        tag: Annotated[str | None, Field(description="Tag without #, nested tags included")] = None,
        sort: Annotated[Literal["recent", "title", "size"], Field()] = "recent",
        limit: Annotated[int, Field(ge=1, le=500)] = 50,
    ) -> list[dict]:
        """List notes, filtered by folder and/or tag. 'recent' sorts by last modified."""
        return await _run(_brain.list_notes, folder, tag, sort, limit)

    @mcp.tool(annotations=READ)
    async def vault_backlinks(note: Note) -> dict:
        """Notes that link to this note, with the line each link appears in."""
        return await _run(_brain.backlinks, note)

    @mcp.tool(annotations=READ)
    async def vault_neighbors(
        note: Note,
        hops: Annotated[int, Field(ge=1, le=4, description="How many link-steps out to go")] = 2,
        limit: Annotated[int, Field(ge=1, le=200)] = 50,
    ) -> dict:
        """Notes within N links of this one (in either direction), closest and most central first,
        with the notes you pass through to get there."""
        return await _run(_brain.neighbors, note, hops, limit)

    @mcp.tool(annotations=READ)
    async def vault_find_path(from_note: Note, to_note: Note) -> dict:
        """Shortest chain of links connecting two notes, with the sentence behind each link.
        Good for 'how does X relate to Y?'."""
        return await _run(_brain.find_path, from_note, to_note)

    @mcp.tool(annotations=READ)
    async def vault_related(
        note: Note,
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
        include_linked: Annotated[bool, Field(description="Also include notes already linked")] = False,
    ) -> dict:
        """Notes whose content is similar in meaning to this note. By default excludes notes already
        linked, so the results are connections that haven't been made yet."""
        return await _run(_brain.related, note, limit, include_linked)

    @mcp.tool(annotations=READ)
    async def vault_hubs(
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
        folder: Annotated[str | None, Field()] = None,
    ) -> list[dict]:
        """The most central notes in the vault (PageRank over links): the core concepts everything else hangs off."""
        return await _run(_brain.hubs, limit, folder)

    @mcp.tool(annotations=READ)
    async def vault_clusters(
        min_size: Annotated[int, Field(ge=2, le=50)] = 3,
        limit: Annotated[int, Field(ge=1, le=50)] = 20,
    ) -> list[dict]:
        """Topic clusters detected from link density (independent of folders), each with its most central notes."""
        return await _run(_brain.clusters, min_size, limit)

    @mcp.tool(annotations=READ)
    async def vault_gaps(limit: Annotated[int, Field(ge=1, le=200)] = 30) -> dict:
        """Where the vault is thin: links to notes that don't exist yet (ranked by how often they're
        referenced) and notes nothing links to."""
        return await _run(_brain.gaps, limit)

    @mcp.tool(annotations=READ)
    async def vault_note_history(note: Note, limit: Annotated[int, Field(ge=1, le=100)] = 20) -> dict:
        """Git history of a note: when it changed and by how much. Shows how thinking on a topic evolved."""
        return await _run(_brain.history, note, limit)

    @mcp.tool(annotations=READ)
    async def vault_status() -> dict:
        """Index health: note/link/chunk counts, last sync, embedding status, errors."""
        if _brain is None:
            return {"ready": False}
        return await anyio.to_thread.run_sync(_brain.status)

    @mcp.tool(annotations=WRITE)
    async def vault_create_note(
        path: Annotated[str, Field(description="Vault-relative path, e.g. 'Inbox/Hook ideas.md'. Must not exist yet.")],
        content: Annotated[str, Field(description="Markdown body. Use [[wikilinks]] to connect to existing notes.")],
        frontmatter: Annotated[dict | None, Field(description="Optional YAML frontmatter, e.g. {'tags': ['hooks']}")] = None,
    ) -> dict:
        """Create a new note. Fails if the path already exists (never overwrites). Synced to the laptop via git within about a minute."""
        return await _run(_brain.create_note, path, content, frontmatter)

    @mcp.tool(annotations=WRITE)
    async def vault_append_to_note(
        note: Note,
        content: Annotated[str, Field(description="Markdown to add at the end of the note")],
        heading: Annotated[str | None, Field(description="Optional '## heading' to put above the new content")] = None,
    ) -> dict:
        """Append to the end of an existing note. Existing content is never changed."""
        return await _run(_brain.append_to_note, note, content, heading)

    return mcp
