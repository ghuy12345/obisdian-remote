# Vault Brain

Your Obsidian vault as an always-on, queryable brain: a link graph + semantic
search + append-only writes, exposed as a REST API and an MCP server for
Claude. One container, one SQLite file, git as the only source of truth.

```
Laptop (Obsidian + Obsidian Git, every 1 min)  <->  GitHub  <->  Vault Brain (Coolify)
                                                                   ├─ pulls every 60s, reindexes what changed
                                                                   ├─ SQLite: notes, links, chunks, FTS, vectors
                                                                   ├─ /api/*            REST (Bearer token)
                                                                   └─ /<secret>/mcp     MCP for Claude
```

The index is derived data. Delete `/data/index.db` and it rebuilds from the
vault (embeddings are re-bought for a few cents). Moving to Postgres later is a
reindex, not a migration.

## What Claude can do with it

| Tool | What it answers |
|---|---|
| `vault_get_context` | **Start here.** Best note for a topic + what it links to, what links to it (with the sentence), 2-hop notes, same-cluster notes, semantically similar notes that aren't linked yet, and unwritten links. |
| `vault_search` | Hybrid keyword + meaning search, optional folder filter |
| `vault_read_note` | Full note with frontmatter, tags, outlinks |
| `vault_list_notes` | By folder / tag, sorted by recent, title or size |
| `vault_backlinks` | Who links here, with context lines |
| `vault_neighbors` | Everything within N hops, and which notes you pass through |
| `vault_find_path` | Shortest chain of links between two notes ("how does X relate to Y?") |
| `vault_related` | Similar-in-meaning notes you haven't linked yet |
| `vault_hubs` | Most central notes (PageRank) |
| `vault_clusters` | Topic clusters from link density, independent of folders |
| `vault_gaps` | Notes you link to but haven't written, and orphan notes |
| `vault_note_history` | Git history of a note: how your thinking evolved |
| `vault_status` | Index health |
| `vault_create_note` | New note; fails if it exists (never overwrites) |
| `vault_append_to_note` | Adds to the end of a note |
| `vault_update_section` | Rewrites, or adds to the start/end of, the part under one heading |
| `vault_edit_note` | Replaces an exact piece of text anywhere (mid-paragraph, frontmatter, ...) |

Note names can be titles, aliases (`aliases:` in frontmatter) or paths.
Link resolution matches Obsidian: path, then file name, then alias;
`[[Note|alias]]`, `[[Note#Heading]]`, `![[embeds]]` and relative markdown
links all work; links inside code blocks are ignored.

## How writes stay safe with your laptop sync

Every write runs inside one lock: `git pull` → change → commit → push, so it
always applies to the newest version your laptop pushed. If the push is
rejected because your laptop pushed in between, it rebases and retries.

- **Edits match exact text.** `vault_edit_note` only changes text it finds
  exactly once in the latest version. If you've rewritten that line on the
  laptop, the edit fails with a message instead of overwriting you.
- **Different lines merge.** You editing one paragraph while Claude edits
  another in the same minute is fine; git merges them.
- **Same line, same minute:** your laptop's version wins, the server's change
  is dropped, and the tool returns an error saying so.
- **Everything is a commit.** Any edit can be undone from git history
  (`vault_note_history` shows it). Notes are never deleted.
- **Size guard.** One edit can't remove more than `MAX_EDIT_DELETE_CHARS`
  (5000 by default).

Tip: `WRITE_CREATE_DIRS=Inbox` keeps new notes in one folder; `EDITS_ENABLED=false`
turns off in-place edits and leaves create + append.

## Deploy on Coolify

1. **GitHub token for the vault.** GitHub → Settings → Developer settings →
   Fine-grained tokens → only select the vault repo → Contents: *Read and write*.
2. **New resource** in Coolify → *Docker Compose* (or *Dockerfile*) from this repo.
3. **Environment variables** (see `.env.example`):
   - `VAULT_REPO_URL=https://x-access-token:<token>@github.com/ghuy12345/JoshOS-Remote-Obsidian.git`
   - `API_TOKEN` → `openssl rand -hex 32`
   - `MCP_SECRET` → `openssl rand -hex 24`
   - `ALLOWED_HOSTS=brain.apps.sebaagency.com`
   - `EMBEDDINGS_API_KEY` → your OpenRouter key
   - `GIT_AUTHOR_EMAIL` → an email GitHub shows on the server's commits
4. **Domain**: set it on the service, port `8000`. Coolify handles TLS.
5. **Persistent storage**: the compose file declares a volume at `/data`
   (vault clone + index). Keep it.
6. Deploy. `https://brain.apps.sebaagency.com/health` shows `"ready": true` once the first
   index is built. A few thousand notes take well under a minute, plus the
   one-off embedding pass.

## Connect Claude

**Claude Desktop / claude.ai:** Settings → Connectors → Add custom connector →
URL `https://brain.apps.sebaagency.com/<MCP_SECRET>/mcp`. The secret in the path is the
password, so keep that URL private.

**Claude Code:** `claude mcp add --transport http vault https://brain.apps.sebaagency.com/<MCP_SECRET>/mcp`

## REST API

All routes need `Authorization: Bearer <API_TOKEN>`. Interactive docs at
`/api/docs`.

```
GET  /api/context?topic=awareness levels
GET  /api/search?q=hooks for cold traffic&mode=hybrid&folder=Creative Strategy
GET  /api/note?name=Hooks
GET  /api/backlinks?name=Hooks
GET  /api/neighbors?name=Hooks&hops=2
GET  /api/path?a=Offer Architecture&b=Hooks
GET  /api/related?name=Men Mansion
GET  /api/hubs   /api/clusters   /api/gaps   /api/history?name=Hooks
GET  /api/notes?folder=Clients&tag=copywriting&sort=recent
POST /api/notes   {"path": "Inbox/Idea.md", "content": "...", "frontmatter": {"tags": ["idea"]}}
POST /api/append  {"note": "Men Mansion", "content": "...", "heading": "2026-09-26"}
POST /api/edit    {"note": "Hooks", "old_text": "exact text", "new_text": "replacement"}
POST /api/section {"note": "Hooks", "heading": "Examples", "content": "...", "mode": "replace"}
POST /api/sync    force a pull + reindex now
GET  /api/status
```

## Settings

| Variable | Default | |
|---|---|---|
| `SYNC_INTERVAL_SECONDS` | 60 | How often to pull |
| `EMBEDDINGS_BASE_URL` | OpenRouter | Any OpenAI-compatible `/embeddings` API |
| `EMBEDDINGS_MODEL` | `openai/text-embedding-3-small` | Changing it re-embeds everything |
| `WRITES_ENABLED` | true | Set false for read-only |
| `EDITS_ENABLED` | true | Allow in-place edits (edit_note, update_section) |
| `WRITE_CREATE_DIRS` / `WRITE_APPEND_DIRS` / `WRITE_EDIT_DIRS` | (anywhere) | Comma-separated folder allowlists |
| `MAX_EDIT_DELETE_CHARS` | 5000 | Largest deletion one edit may make |
| `IGNORE_DIRS` | `.obsidian,.git,.trash,node_modules` | Hidden folders are always skipped |
| `CHUNK_CHARS` / `CHUNK_OVERLAP` | 1500 / 200 | Semantic chunk size |

Without `EMBEDDINGS_API_KEY` everything except semantic search still works;
tools say so in a `notice` field. If the embeddings API is down, keyword
and graph queries keep working and embedding retries on the next sync.

## Develop

```
pip install -r requirements.txt pytest
python -m pytest -q          # builds a throwaway vault + git remote per test
cp .env.example .env         # set DATA_DIR=./data for local runs
uvicorn app.main:app --reload
```

## Layout

```
app/parser.py     markdown → frontmatter, links (with context + section), tags, chunks
app/store.py      SQLite schema, FTS5, embedding cache
app/graph.py      Obsidian-style link resolution, networkx graph queries
app/embeddings.py OpenAI-compatible embeddings client with retries
app/gitrepo.py    clone / pull / commit / push with rebase + conflict handling
app/engine.py     Brain: sync loop, all queries, append-only writes
app/mcp_tools.py  MCP tools (thin wrappers over Brain)
app/main.py       FastAPI: REST + MCP mount + background sync thread
```
